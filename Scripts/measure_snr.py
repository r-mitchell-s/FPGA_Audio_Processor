# snr_measurement.py
import pyvisa
import numpy as np
import time
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import signal

# Create top-level output directory in ../Documents relative to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.normpath(os.path.join(script_dir, "..", "Documents", "snr_measurements"))
os.makedirs(output_dir, exist_ok=True)

print(f"Data will be saved to: {output_dir}")

# Connect to scope
rm = pyvisa.ResourceManager()
resources = rm.list_resources()

if not resources:
    print("No oscilloscope found!")
    exit()

scope_resource = resources[0]
print(f"Connecting to {scope_resource}...")
scope = rm.open_resource(scope_resource)
scope.timeout = 30000  # 30 seconds timeout

def setup_oscilloscope(amplitude_range=2.0):
    """Setup the oscilloscope with specified amplitude range"""
    print("Setting up oscilloscope...")
    scope.write("*RST")  # Reset the scope
    time.sleep(2)  # Wait for reset to complete

    # Explicitly enable both channels
    scope.write("SELECT:CH1 ON")  # Enable Channel 1 (input signal)
    time.sleep(0.5)
    scope.write("SELECT:CH2 ON")  # Enable Channel 2 (output signal)
    time.sleep(0.5)
    
    # Configure vertical settings based on expected amplitude range
    # Add 20% extra headroom to avoid clipping
    volts_per_div = (amplitude_range * 1.2) / 6  # Aim for signal to use about 6 divisions with headroom
    scope.write(f"CH1:SCALE {volts_per_div}")  # Set volts/div for input
    time.sleep(0.5)
    scope.write(f"CH2:SCALE {volts_per_div}")  # Set volts/div for output
    time.sleep(0.5)

    # Configure horizontal timebase - capture multiple cycles of the test signal
    scope.write("HORIZONTAL:SCALE 0.0005")  # 500µs/div (good for 1kHz)
    time.sleep(0.5)
    scope.write("HORIZONTAL:RECORDLENGTH 10000")  # Set record length
    time.sleep(0.5)

    # Configure trigger
    scope.write("TRIGGER:A:TYPE EDGE")
    time.sleep(0.5)
    scope.write("TRIGGER:A:EDGE:SOURCE CH1")
    time.sleep(0.5)
    scope.write("TRIGGER:A:EDGE:SLOPE RISE")
    time.sleep(0.5)
    scope.write("TRIGGER:A:LEVEL 0.1")  # Set trigger level
    time.sleep(0.5)

def safe_query(command, max_attempts=3):
    """Query the scope with retry logic"""
    for attempt in range(max_attempts):
        try:
            return scope.query(command)
        except pyvisa.errors.VisaIOError as e:
            print(f"Query attempt {attempt+1} failed: {e}")
            if attempt < max_attempts - 1:
                print("Retrying...")
                time.sleep(1)
            else:
                print(f"Failed to query {command} after {max_attempts} attempts")
                raise

def parse_frequency(freq_str):
    """Parse frequency string to a float value in Hz"""
    try:
        freq_str = freq_str.strip().lower()
        if "khz" in freq_str:
            # Remove the 'khz' and convert to Hz
            freq_val = float(freq_str.replace("khz", "")) * 1000
        elif "hz" in freq_str:
            # Remove the 'hz'
            freq_val = float(freq_str.replace("hz", ""))
        else:
            # Try direct conversion, assume Hz
            freq_val = float(freq_str)
        return freq_val
    except ValueError:
        print(f"Warning: Couldn't parse frequency '{freq_str}'. Using 1kHz as default.")
        return 1000.0

def measure_noise_floor():
    """Measure system noise floor with no input signal"""
    print("\n--- Measuring System Noise Floor ---")
    print("Please disconnect the signal generator or set its output to minimum.")
    input("Press Enter when ready...")
    
    # Setup oscilloscope with high sensitivity for noise floor measurement
    # Use a much more sensitive scale for noise measurement
    print("Setting up oscilloscope for noise floor measurement...")
    scope.write("*RST")  # Reset the scope
    time.sleep(2)
    
    # Configure for maximum sensitivity
    scope.write("CH1:SCALE 0.002")  # 2mV/div for high sensitivity
    time.sleep(0.5)
    scope.write("CH2:SCALE 0.002")  # 2mV/div for high sensitivity
    time.sleep(0.5)
    
    # Make sure both channels are ON
    scope.write("SELECT:CH1 ON")
    time.sleep(0.5)
    scope.write("SELECT:CH2 ON")
    time.sleep(0.5)
    
    # Use AC coupling to remove DC offsets
    scope.write("CH1:COUPLING AC")
    time.sleep(0.5)
    scope.write("CH2:COUPLING AC")  # Changed from DC to AC for consistency
    time.sleep(0.5)
    
    # Set position to center
    scope.write("CH1:POSITION 0")
    time.sleep(0.5)
    scope.write("CH2:POSITION 0")
    time.sleep(0.5)
    
    # Capture with longer record length but not too long to avoid timeouts
    scope.write("HORIZONTAL:RECORDLENGTH 20000")  # Reduced from 50000
    time.sleep(0.5)
    
    # Force acquisition
    scope.write("ACQUIRE:STATE STOP")
    time.sleep(0.5)
    scope.write("ACQUIRE:STATE RUN")
    time.sleep(0.5)
    scope.write("TRIGGER FORCE")
    time.sleep(2)
    
    # Get noise data from both channels
    waveform_data = {"Time": [], "CH1": [], "CH2": []}
    
    # Process Channel 1 (Input)
    print("Getting input channel noise data...")
    scope.write("DATA:SOURCE CH1")
    time.sleep(0.5)
    scope.write("DATA:ENCDG ASCII")
    time.sleep(0.5)
    record_length = int(safe_query("HORIZONTAL:RECORDLENGTH?").strip())
    scope.write(f"DATA:START 1")
    time.sleep(0.5)
    scope.write(f"DATA:STOP {record_length}")
    time.sleep(0.5)
    
    # Get CH1 scaling
    x_incr = float(safe_query("WFMPRE:XINCR?").strip())
    x_zero = float(safe_query("WFMPRE:XZERO?").strip())
    y_mult1 = float(safe_query("WFMPRE:YMULT?").strip())
    y_zero1 = float(safe_query("WFMPRE:YZERO?").strip())
    y_offset1 = float(safe_query("WFMPRE:YOFF?").strip())
    
    # Get CH1 data
    ch1_raw = safe_query("CURVE?")
    ch1_values = ch1_raw.split(',')
    
    # Process Channel 2 (Output)
    print("Getting output channel noise data...")
    scope.write("DATA:SOURCE CH2")
    time.sleep(1)  # Increase wait time
    
    # Get CH2 scaling - with extra retries and timeout
    try:
        scope.timeout = 10000  # Increase timeout for potentially slow operations
        y_mult2 = float(safe_query("WFMPRE:YMULT?", max_attempts=5).strip())
        y_zero2 = float(safe_query("WFMPRE:YZERO?", max_attempts=5).strip())
        y_offset2 = float(safe_query("WFMPRE:YOFF?", max_attempts=5).strip())
        
        # Get CH2 data with increased timeout
        ch2_raw = safe_query("CURVE?", max_attempts=5)
        ch2_values = ch2_raw.split(',')
    except Exception as e:
        print(f"Error reading CH2 data: {e}")
        print("Using placeholder data for CH2 (zeroes)")
        ch2_values = ["0"] * len(ch1_values)
        y_mult2 = 1.0
        y_zero2 = 0.0
        y_offset2 = 0.0
    finally:
        scope.timeout = 30000  # Reset timeout to original value
    
    # Convert to voltages
    time_axis = [x_zero + i * x_incr for i in range(len(ch1_values))]
    ch1_voltages = [(float(val) - y_offset1) * y_mult1 + y_zero1 for val in ch1_values]
    
    # Make sure CH2 data matches length of CH1
    if len(ch2_values) >= len(ch1_values):
        ch2_voltages = [(float(val) - y_offset2) * y_mult2 + y_zero2 for val in ch2_values[:len(ch1_values)]]
    else:
        # Pad CH2 with zeros if it's shorter
        ch2_voltages = [(float(val) - y_offset2) * y_mult2 + y_zero2 for val in ch2_values]
        ch2_voltages.extend([0] * (len(ch1_values) - len(ch2_values)))
    
    # Calculate RMS noise
    input_noise_rms = np.sqrt(np.mean(np.array(ch1_voltages)**2))
    output_noise_rms = np.sqrt(np.mean(np.array(ch2_voltages)**2))
    
    print(f"Input channel noise floor: {input_noise_rms*1000:.3f} mVrms")
    print(f"Output channel noise floor: {output_noise_rms*1000:.3f} mVrms")
    
    # Calculate noise spectrum
    window = signal.windows.blackman(len(ch1_voltages))
    ch1_windowed = np.array(ch1_voltages) * window
    ch2_windowed = np.array(ch2_voltages) * window
    
    ch1_fft = np.fft.rfft(ch1_windowed)
    ch2_fft = np.fft.rfft(ch2_windowed)
    
    ch1_magnitude = np.abs(ch1_fft)
    ch2_magnitude = np.abs(ch2_fft)
    
    sample_rate = 1.0 / x_incr
    freq_axis = np.fft.rfftfreq(len(ch1_voltages), 1/sample_rate)
    
    # Create noise floor directory to save data
    noise_floor_dir = os.path.join(output_dir, "noise_floor")
    os.makedirs(noise_floor_dir, exist_ok=True)
    
    # Save noise floor data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    noise_file = os.path.join(noise_floor_dir, f"noise_floor_{timestamp}.csv")
    
    noise_data = {
        "Time": time_axis,
        "CH1_noise": ch1_voltages,
        "CH2_noise": ch2_voltages
    }
    
    pd.DataFrame(noise_data).to_csv(noise_file, index=False)
    
    # Generate noise floor spectrum plot
    plt.figure(figsize=(10, 6))
    plt.plot(freq_axis, 20*np.log10(ch1_magnitude + 1e-15), label="Input Channel Noise")
    plt.plot(freq_axis, 20*np.log10(ch2_magnitude + 1e-15), label="Output Channel Noise")
    plt.title("System Noise Floor Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
    plt.legend()
    plt.xscale("log")
    
    # Save the plot
    plot_file = os.path.join(noise_floor_dir, f"noise_floor_spectrum_{timestamp}.png")
    plt.savefig(plot_file)
    plt.close()
    
    print(f"Saved noise floor data to {noise_file}")
    print(f"Saved noise floor spectrum plot to {plot_file}")
    
    # Return noise floor measurements
    return {
        "input_noise_rms": input_noise_rms,
        "output_noise_rms": output_noise_rms,
        "input_spectrum": ch1_magnitude,
        "output_spectrum": ch2_magnitude,
        "frequencies": freq_axis
    }

def calculate_corrected_snr(measured_snr_db, noise_floor_db):
    """Calculate corrected SNR accounting for measurement system noise floor"""
    # Convert from dB to linear power ratio
    measured_snr_linear = 10**(measured_snr_db / 10)
    noise_floor_linear = 10**(noise_floor_db / 10)
    
    # If noise floor is significantly better than measured SNR, return measured value
    if noise_floor_linear > 10 * measured_snr_linear:
        return measured_snr_db
    
    # Calculate what the SNR would be without measurement system noise
    # Using the formula: 1/SNR_true = 1/SNR_measured - 1/SNR_noise_floor
    if measured_snr_linear <= noise_floor_linear:
        # Measured SNR is limited by noise floor
        corrected_snr_linear = float('inf')  # Or some very large value
    else:
        corrected_snr_linear = 1 / ((1/measured_snr_linear) - (1/noise_floor_linear))
    
    # Convert back to dB
    corrected_snr_db = 10 * np.log10(corrected_snr_linear)
    
    return corrected_snr_db

def capture_waveforms(signal_frequency):
    """Capture waveform data from both channels"""
    try:
        # Adjust horizontal scale based on the signal frequency
        frequency = parse_frequency(signal_frequency)
        if frequency > 0:
            # Calculate a good timebase setting - aim for 8-10 cycles
            cycles_to_capture = 10
            timebase = (1.0 / frequency) * cycles_to_capture / 10  # 10 divisions on screen
            # Limit timebase to reasonable values
            timebase = min(max(timebase, 0.00005), 0.05)  # Between 50µs and 50ms per division
            
            print(f"Setting timebase to {timebase*1000:.3f} ms/div for {frequency} Hz signal")
            scope.write(f"HORIZONTAL:SCALE {timebase}")
            time.sleep(0.5)
        
        # Force a new acquisition for each measurement
        scope.write("ACQUIRE:STATE STOP")
        time.sleep(0.5)
        scope.write("ACQUIRE:MODE SAMPLE")  # Use sample mode for spectral analysis
        time.sleep(0.5)
        scope.write("ACQUIRE:STOPAFTER SEQUENCE")
        time.sleep(0.5)
        
        # Clear any previous acquisitions and force a new trigger
        scope.write("ACQUIRE:STATE RUN")
        time.sleep(0.5)
        
        # Wait for acquisition to complete
        print("Waiting for trigger...")
        # Query acquisition state until complete or timeout
        timeout_start = time.time()
        max_wait = 8  # seconds
        while time.time() - timeout_start < max_wait:
            acq_state = safe_query("ACQUIRE:STATE?").strip()
            if acq_state == "0":  # Acquisition complete
                break
            time.sleep(0.5)
        
        if time.time() - timeout_start >= max_wait:
            # Force trigger if waiting too long
            print("Forcing trigger...")
            scope.write("TRIGGER FORCE")
            time.sleep(1)
        
        # Get data from both channels
        waveform_data = {"Time": [], "CH1": [], "CH2": []}
        
        # Process Channel 1 (Input)
        print("Getting Channel 1 data...")
        scope.write("DATA:SOURCE CH1")
        time.sleep(0.5)
        scope.write("DATA:ENCDG ASCII")
        time.sleep(0.5)
        record_length = int(safe_query("HORIZONTAL:RECORDLENGTH?").strip())
        scope.write(f"DATA:START 1")
        time.sleep(0.5)
        scope.write(f"DATA:STOP {record_length}")
        time.sleep(0.5)
        
        # Get CH1 scaling
        x_incr = float(safe_query("WFMPRE:XINCR?").strip())
        x_zero = float(safe_query("WFMPRE:XZERO?").strip())
        y_mult1 = float(safe_query("WFMPRE:YMULT?").strip())
        y_zero1 = float(safe_query("WFMPRE:YZERO?").strip())
        y_offset1 = float(safe_query("WFMPRE:YOFF?").strip())
        
        # Get CH1 data
        ch1_raw = safe_query("CURVE?")
        ch1_values = ch1_raw.split(',')
        
        # Process Channel 2 (Output)
        print("Getting Channel 2 data...")
        scope.write("DATA:SOURCE CH2")
        time.sleep(0.5)
        
        # Get CH2 scaling
        y_mult2 = float(safe_query("WFMPRE:YMULT?").strip())
        y_zero2 = float(safe_query("WFMPRE:YZERO?").strip())
        y_offset2 = float(safe_query("WFMPRE:YOFF?").strip())
        
        # Get CH2 data
        ch2_raw = safe_query("CURVE?")
        ch2_values = ch2_raw.split(',')
        
        # Convert to voltages
        time_axis = [x_zero + i * x_incr for i in range(len(ch1_values))]
        ch1_voltages = [(float(val) - y_offset1) * y_mult1 + y_zero1 for val in ch1_values]
        
        # Make sure CH2 data matches length of CH1
        if len(ch2_values) >= len(ch1_values):
            ch2_voltages = [(float(val) - y_offset2) * y_mult2 + y_zero2 for val in ch2_values[:len(ch1_values)]]
        else:
            # Pad CH2 with zeros if it's shorter
            ch2_voltages = [(float(val) - y_offset2) * y_mult2 + y_zero2 for val in ch2_values]
            ch2_voltages.extend([0] * (len(ch1_values) - len(ch2_values)))
        
        sample_rate = 1.0 / x_incr  # Calculate sample rate for FFT
        
        return {
            "Time": time_axis,
            "CH1": ch1_voltages,
            "CH2": ch2_voltages,
            "SampleRate": sample_rate
        }
        
    except Exception as e:
        print(f"Error capturing data: {e}")
        import traceback
        traceback.print_exc()
        return {"Time": [], "CH1": [], "CH2": [], "SampleRate": 0}

def calculate_snr_improved(data, signal_freq, noise_floor_data=None):
    """Calculate SNR with improved methodology for high SNR systems"""
    try:
        # Extract data and sample rate
        input_signal = np.array(data["CH1"])
        output_signal = np.array(data["CH2"])
        sample_rate = data["SampleRate"]
        
        if len(input_signal) < 1000 or len(output_signal) < 1000:
            print("Insufficient data for SNR calculation")
            return None
        
        # Use Blackman-Harris window for better spectral leakage suppression
        window = signal.windows.blackmanharris(len(input_signal))
        input_windowed = input_signal * window
        output_windowed = output_signal * window
        
        # Compute FFTs with zero padding for better frequency resolution
        n_fft = 2**int(np.ceil(np.log2(len(input_signal))) + 1)  # Next power of 2, doubled
        input_fft = np.fft.rfft(input_windowed, n=n_fft)
        output_fft = np.fft.rfft(output_windowed, n=n_fft)
        
        # Calculate magnitudes
        input_magnitude = np.abs(input_fft)
        output_magnitude = np.abs(output_fft)
        
        # Calculate frequency axis
        freq_axis = np.fft.rfftfreq(n_fft, 1/sample_rate)
        
        # More precise location of fundamental frequency
        # Look in a narrow range around the expected frequency
        target_freq = parse_frequency(signal_freq)
        freq_range = 0.1 * target_freq  # Look within ±10% of expected frequency
        
        # Find index range for expected frequency
        lower_idx = np.searchsorted(freq_axis, target_freq - freq_range)
        upper_idx = np.searchsorted(freq_axis, target_freq + freq_range)
        
        # Find peak in this range
        input_peak_idx = lower_idx + np.argmax(input_magnitude[lower_idx:upper_idx])
        output_peak_idx = lower_idx + np.argmax(output_magnitude[lower_idx:upper_idx])
        
        # Get the actual frequencies found
        input_freq = freq_axis[input_peak_idx]
        output_freq = freq_axis[output_peak_idx]
        
        print(f"Signal frequency: Target={target_freq}Hz, Input={input_freq:.1f}Hz, Output={output_freq:.1f}Hz")
        
        # Calculate exact number of cycles in the signal
        input_cycles = input_freq * (len(input_signal) / sample_rate)
        print(f"Captured approximately {input_cycles:.1f} cycles of the input signal")
        
        # Extract signal power (at fundamental)
        input_signal_power = input_magnitude[input_peak_idx]**2
        output_signal_power = output_magnitude[output_peak_idx]**2
        
        # More sophisticated noise calculation - exclude wider bands around harmonics
        input_mask = np.ones_like(input_magnitude, dtype=bool)
        output_mask = np.ones_like(output_magnitude, dtype=bool)
        
        # Exclude more of the DC component
        dc_width = int(n_fft/1000)  # More aggressive DC exclusion
        input_mask[:dc_width] = False
        output_mask[:dc_width] = False
        
        # Exclude wider bands around harmonics
        harmonic_width_percent = 0.05  # 5% of frequency for each harmonic
        
        # Calculate exact harmonic frequencies and exclude bands around them
        for h in range(1, 10):  # Consider up to 10 harmonics
            h_freq_in = input_freq * h
            h_freq_out = output_freq * h
            
            # Calculate width in bins based on percentage of frequency
            width_in_bins = int(harmonic_width_percent * h_freq_in * n_fft / sample_rate)
            
            # Convert frequencies to FFT bin indices
            h_idx_in = int(h_freq_in * n_fft / sample_rate)
            h_idx_out = int(h_freq_out * n_fft / sample_rate)
            
            # Skip if beyond FFT range
            if h_idx_in < len(input_mask):
                # Mask a window around each harmonic
                left_idx = max(0, h_idx_in - width_in_bins)
                right_idx = min(len(input_mask), h_idx_in + width_in_bins + 1)
                input_mask[left_idx:right_idx] = False
            
            if h_idx_out < len(output_mask):
                left_idx = max(0, h_idx_out - width_in_bins)
                right_idx = min(len(output_mask), h_idx_out + width_in_bins + 1)
                output_mask[left_idx:right_idx] = False
        
        # Calculate noise power (using median instead of mean for robustness against outliers)
        # Use 99th percentile to exclude any residual spikes
        input_noise_values = input_magnitude[input_mask]**2
        output_noise_values = output_magnitude[output_mask]**2
        
        # Sort the values and use the median as a robust estimate
        input_noise_power = np.median(input_noise_values) if len(input_noise_values) > 0 else 1e-15
        output_noise_power = np.median(output_noise_values) if len(output_noise_values) > 0 else 1e-15
        
        # Calculate raw SNR in dB
        input_snr_db = 10 * np.log10(input_signal_power / input_noise_power)
        output_snr_db = 10 * np.log10(output_signal_power / output_noise_power)
        
        print(f"Raw SNR values - Input: {input_snr_db:.2f} dB, Output: {output_snr_db:.2f} dB")
        
        # If we have noise floor data, calculate corrected SNR
        corrected_input_snr_db = input_snr_db
        corrected_output_snr_db = output_snr_db
        
        if noise_floor_data is not None:
            # Calculate noise floor in dB relative to signal
            input_noise_floor_power = np.median(noise_floor_data["input_spectrum"]**2)
            output_noise_floor_power = np.median(noise_floor_data["output_spectrum"]**2)
            
            # Normalize to same scale as the signal measurement
            input_signal_rms = np.sqrt(np.mean(input_signal**2))
            output_signal_rms = np.sqrt(np.mean(output_signal**2))
            
            input_noise_floor_db = 10 * np.log10(input_noise_floor_power * (input_signal_rms / noise_floor_data["input_noise_rms"])**2 / input_signal_power)
            output_noise_floor_db = 10 * np.log10(output_noise_floor_power * (output_signal_rms / noise_floor_data["output_noise_rms"])**2 / output_signal_power)
            
            # Calculate corrected SNR
            corrected_input_snr_db = calculate_corrected_snr(input_snr_db, input_noise_floor_db)
            corrected_output_snr_db = calculate_corrected_snr(output_snr_db, output_noise_floor_db)
            
            print(f"Corrected SNR values - Input: {corrected_input_snr_db:.2f} dB, Output: {corrected_output_snr_db:.2f} dB")
        
        # THD calculation with improved precision
        input_harmonics_power = 0
        output_harmonics_power = 0
        
        # More precise harmonic measurement
        for h in range(2, 10):  # 2nd to 9th harmonics
            h_freq_in = input_freq * h
            h_freq_out = output_freq * h
            
            # Check if harmonic is within bandwidth
            if h_freq_in < sample_rate/2 and h_freq_out < sample_rate/2:
                # Find closest bin to harmonic frequency
                h_idx_in = int(round(h_freq_in * n_fft / sample_rate))
                h_idx_out = int(round(h_freq_out * n_fft / sample_rate))
                
                # Use quadratic interpolation for more precise harmonic amplitude
                if 0 < h_idx_in < len(input_magnitude)-1:
                    y0, y1, y2 = input_magnitude[h_idx_in-1], input_magnitude[h_idx_in], input_magnitude[h_idx_in+1]
                    h_amp_in = y1 + 0.5 * ((y0 - y2) / (2*y1 - y0 - y2)) * (y2 - y0)
                    input_harmonics_power += h_amp_in**2
                
                if 0 < h_idx_out < len(output_magnitude)-1:
                    y0, y1, y2 = output_magnitude[h_idx_out-1], output_magnitude[h_idx_out], output_magnitude[h_idx_out+1]
                    h_amp_out = y1 + 0.5 * ((y0 - y2) / (2*y1 - y0 - y2)) * (y2 - y0)
                    output_harmonics_power += h_amp_out**2
        
        # Calculate THD in dB
        input_thd_db = 10 * np.log10(input_harmonics_power / input_signal_power) if input_signal_power > 0 else -100
        output_thd_db = 10 * np.log10(output_harmonics_power / output_signal_power) if output_signal_power > 0 else -100
        
        # Store results
        result = {
            "Input_SNR_dB": input_snr_db,
            "Output_SNR_dB": output_snr_db,
            "Corrected_Input_SNR_dB": corrected_input_snr_db,
            "Corrected_Output_SNR_dB": corrected_output_snr_db,
            "Input_THD_dB": input_thd_db,
            "Output_THD_dB": output_thd_db,
            "Frequencies": freq_axis,
            "Input_FFT": input_magnitude,
            "Output_FFT": output_magnitude,
            "Input_Peak_Idx": input_peak_idx,
            "Output_Peak_Idx": output_peak_idx,
            "Input_Frequency": input_freq,
            "Output_Frequency": output_freq,
            "FFT_Resolution": sample_rate / n_fft,
            "Input_Mask": input_mask,
            "Output_Mask": output_mask
        }
        
        return result
    
    except Exception as e:
        print(f"Error calculating SNR: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_snr_plots(data, snr_results, trial, effect_config, signal_dir, signal_info):
    """Generate verification plots for SNR calculation"""
    try:
        # Save plots in a dedicated plots folder within the signal directory
        plots_dir = os.path.join(signal_dir, "verification_plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. Time Domain Plot
        plt.figure(figsize=(12, 12))
        
        # Calculate peak-to-peak voltages for display
        ch1_pp = np.max(data["CH1"]) - np.min(data["CH1"])
        ch2_pp = np.max(data["CH2"]) - np.min(data["CH2"])
        
        # Time domain subplot
        plt.subplot(3, 1, 1)
        plt.plot(data["Time"], data["CH1"], label="Input (CH1)")
        plt.plot(data["Time"], data["CH2"], label="Output (CH2)")
        plt.title(f"Time Domain - {effect_config} - {signal_info['shape']} {signal_info['frequency']} - Trial {trial}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (V)")
        # Add voltage annotations
        plt.annotate(f"Input Vpp: {ch1_pp:.3f}V", xy=(0.05, 0.9), xycoords='axes fraction')
        plt.annotate(f"Output Vpp: {ch2_pp:.3f}V", xy=(0.05, 0.8), xycoords='axes fraction')
        plt.grid(True)
        plt.legend()
        
        # Linear frequency domain subplot
        plt.subplot(3, 1, 2)
        
        # Plot FFT magnitude in linear scale
        if snr_results:
            freq = snr_results["Frequencies"]
            input_mag = snr_results["Input_FFT"]
            output_mag = snr_results["Output_FFT"]
            
            # Only show up to 20kHz (audio range)
            cutoff_idx = np.searchsorted(freq, 20000)
            freq = freq[:cutoff_idx]
            input_mag = input_mag[:cutoff_idx]
            output_mag = output_mag[:cutoff_idx]
            
            plt.plot(freq, input_mag, label="Input Spectrum")
            plt.plot(freq, output_mag, label="Output Spectrum")
            
            # Mark the fundamental and harmonics
            input_peak_freq = snr_results["Input_Frequency"]
            output_peak_freq = snr_results["Output_Frequency"]
            
            plt.axvline(x=input_peak_freq, color='green', linestyle='--', alpha=0.5)
            plt.axvline(x=output_peak_freq, color='red', linestyle='--', alpha=0.5)
            
        plt.title("Frequency Domain (Linear)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.grid(True)
        plt.legend()
        
        # dB frequency domain subplot
        plt.subplot(3, 1, 3)
        
        # Plot FFT magnitude in dB scale with deeper noise floor
        if snr_results:
            input_mag_db = 20 * np.log10(input_mag + 1e-15)  # Lower noise floor (1e-15)
            output_mag_db = 20 * np.log10(output_mag + 1e-15)
            
            plt.plot(freq, input_mag_db, label=f"Input (SNR: {snr_results['Input_SNR_dB']:.1f} dB)")
            plt.plot(freq, output_mag_db, label=f"Output (SNR: {snr_results['Output_SNR_dB']:.1f} dB)")
            
            # Mark the fundamental
            plt.axvline(x=input_peak_freq, color='green', linestyle='--', alpha=0.5)
            plt.axvline(x=output_peak_freq, color='red', linestyle='--', alpha=0.5)
            
            # Add annotations
            if "Corrected_Output_SNR_dB" in snr_results:
                plt.annotate(f"Input THD: {snr_results['Input_THD_dB']:.1f} dB", 
                            xy=(0.05, 0.30), xycoords='axes fraction')
                plt.annotate(f"Output THD: {snr_results['Output_THD_dB']:.1f} dB", 
                            xy=(0.05, 0.25), xycoords='axes fraction')
                plt.annotate(f"Corrected Input SNR: {snr_results['Corrected_Input_SNR_dB']:.1f} dB", 
                            xy=(0.05, 0.20), xycoords='axes fraction')
                plt.annotate(f"Corrected Output SNR: {snr_results['Corrected_Output_SNR_dB']:.1f} dB", 
                            xy=(0.05, 0.15), xycoords='axes fraction')
            else:
                plt.annotate(f"Input THD: {snr_results['Input_THD_dB']:.1f} dB", 
                            xy=(0.05, 0.15), xycoords='axes fraction')
                plt.annotate(f"Output THD: {snr_results['Output_THD_dB']:.1f} dB", 
                            xy=(0.05, 0.10), xycoords='axes fraction')
            
        plt.title("Frequency Domain (dB)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.grid(True)
        plt.legend()
        plt.xscale('log')
        plt.ylim(bottom=-140)  # Show down to -140dB
        
        plt.tight_layout()
        
        # Save the plot
        plot_file = os.path.join(plots_dir, f"snr_plot_trial{trial}.png")
        plt.savefig(plot_file, dpi=150)
        plt.close()
        print(f"Saved verification plot to {plot_file}")
        
    except Exception as e:
        print(f"Error generating SNR plots: {e}")
        import traceback
        traceback.print_exc()

def get_effect_configuration():
    """Get user input for effect configuration with validation"""
    valid_effects = ["bypass", "lowpass", "distortion", "ring_modulator", "noise_gate", "all_effects"]
    
    print("\nAvailable effect configurations:")
    for i, effect in enumerate(valid_effects, 1):
        print(f"{i}. {effect}")
    
    while True:
        effect = input("\nEnter effect configuration: ").lower().strip()
        
        # Check if it's a number
        if effect.isdigit() and 1 <= int(effect) <= len(valid_effects):
            return valid_effects[int(effect) - 1]
        # Check if it's a valid effect name
        elif effect in valid_effects:
            return effect
        else:
            print(f"Invalid effect. Please enter one of: {', '.join(valid_effects)}")

def run_snr_test(noise_floor_data=None):
    """Run a complete SNR test sequence"""
    # Get test configuration
    effect_config = get_effect_configuration()
    signal_shape = input("Enter input signal shape (sine recommended): ") or "sine"
    signal_frequency = input("Enter input signal frequency (1kHz recommended): ") or "1kHz"
    amplitude = input("Enter test signal amplitude in Volts peak-to-peak (1.8V recommended): ") or "1.8"

    try:
        amplitude_vpp = float(amplitude)
    except ValueError:
        print("Invalid amplitude value. Using default 1.8V")
        amplitude_vpp = 1.8

    # Setup oscilloscope with proper scale for the specified amplitude
    setup_oscilloscope(amplitude_range=amplitude_vpp)

    # Store signal information
    signal_info = {
        "shape": signal_shape,
        "frequency": signal_frequency,
        "amplitude": f"{amplitude_vpp}Vpp"
    }

    # Create signal folder name (sanitize for filesystem)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    signal_folder = f"{signal_shape}_{signal_frequency}_{amplitude_vpp}Vpp_{timestamp_str}"
    signal_folder = signal_folder.replace(" ", "_").replace("/", "_")

    # Create directory structure
    effect_dir = os.path.join(output_dir, effect_config)
    os.makedirs(effect_dir, exist_ok=True)

    # Create signal-specific directory under the effect directory
    signal_dir = os.path.join(effect_dir, signal_folder)
    os.makedirs(signal_dir, exist_ok=True)

    # Create subdirectories for trials and plots
    trials_dir = os.path.join(signal_dir, "trial_measurements")
    os.makedirs(trials_dir, exist_ok=True)

    plots_dir = os.path.join(signal_dir, "verification_plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Create summary file in the signal directory
    summary_file = os.path.join(signal_dir, f"snr_summary.csv")
    
    # Prepare columns based on whether we have noise floor data
    if noise_floor_data:
        columns = [
            'Trial', 
            'Input_SNR_dB', 
            'Output_SNR_dB',
            'Corrected_Input_SNR_dB',
            'Corrected_Output_SNR_dB', 
            'SNR_Change_dB', 
            'Corrected_SNR_Change_dB',
            'Input_THD_dB', 
            'Output_THD_dB',
            'Input_Vpp',
            'Output_Vpp',
            'Signal Shape', 
            'Signal Frequency',
            'Timestamp'
        ]
    else:
        columns = [
            'Trial', 
            'Input_SNR_dB', 
            'Output_SNR_dB',
            'SNR_Change_dB', 
            'Input_THD_dB', 
            'Output_THD_dB',
            'Input_Vpp',
            'Output_Vpp',
            'Signal Shape', 
            'Signal Frequency',
            'Timestamp'
        ]
    
    with open(summary_file, 'w', newline='') as f:
        pd.DataFrame(columns=columns).to_csv(f, index=False)

    # Ask for number of trials
    num_trials_input = input("Enter number of trials to run (default 10): ")
    try:
        num_trials = int(num_trials_input) if num_trials_input else 10
    except ValueError:
        print("Invalid value. Using default 10 trials.")
        num_trials = 10

    print(f"\nStarting {num_trials} SNR measurements for {effect_config} with {signal_shape} at {signal_frequency}...")
    print(f"Test signal amplitude: {amplitude_vpp} Vpp")

    results = []
    for trial in range(1, num_trials + 1):
        print(f"\nTrial {trial}/{num_trials}")
        
        # Capture waveforms
        data = capture_waveforms(signal_frequency)
        
        # Save waveform data
        if data["Time"]:
            # Calculate peak-to-peak voltages
            ch1_pp = np.max(data["CH1"]) - np.min(data["CH1"])
            ch2_pp = np.max(data["CH2"]) - np.min(data["CH2"])
            
            print(f"Input peak-to-peak: {ch1_pp:.3f}V")
            print(f"Output peak-to-peak: {ch2_pp:.3f}V")
            
            current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            waveform_file = os.path.join(trials_dir, f"snr_trial{trial}_{current_timestamp}.csv")
            
            # Remove SampleRate before saving CSV
            waveform_data = {k: data[k] for k in ["Time", "CH1", "CH2"]}
            pd.DataFrame(waveform_data).to_csv(waveform_file, index=False)
            print(f"Saved waveform data to {waveform_file}")
            
            # Calculate SNR with improved method
            snr_results = calculate_snr_improved(data, signal_frequency, noise_floor_data)
            
            # Generate verification plot
            generate_snr_plots(data, snr_results, trial, effect_config, signal_dir, signal_info)
            
            if snr_results is not None:
                input_snr = snr_results["Input_SNR_dB"]
                output_snr = snr_results["Output_SNR_dB"]
                snr_change = output_snr - input_snr
                
                result_row = {
                    'Trial': trial, 
                    'Input_SNR_dB': input_snr,
                    'Output_SNR_dB': output_snr,
                    'SNR_Change_dB': snr_change,
                    'Input_THD_dB': snr_results["Input_THD_dB"],
                    'Output_THD_dB': snr_results["Output_THD_dB"],
                    'Input_Vpp': ch1_pp,
                    'Output_Vpp': ch2_pp,
                    'Signal Shape': signal_shape,
                    'Signal Frequency': signal_frequency,
                    'Timestamp': current_timestamp
                }
                
                # Add corrected values if available
                if "Corrected_Input_SNR_dB" in snr_results:
                    corrected_input_snr = snr_results["Corrected_Input_SNR_dB"]
                    corrected_output_snr = snr_results["Corrected_Output_SNR_dB"]
                    corrected_snr_change = corrected_output_snr - corrected_input_snr
                    
                    result_row['Corrected_Input_SNR_dB'] = corrected_input_snr
                    result_row['Corrected_Output_SNR_dB'] = corrected_output_snr
                    result_row['Corrected_SNR_Change_dB'] = corrected_snr_change
                    
                    print(f"Input SNR: {input_snr:.2f} dB (Corrected: {corrected_input_snr:.2f} dB)")
                    print(f"Output SNR: {output_snr:.2f} dB (Corrected: {corrected_output_snr:.2f} dB)")
                    print(f"SNR Change: {snr_change:.2f} dB (Corrected: {corrected_snr_change:.2f} dB)")
                else:
                    print(f"Input SNR: {input_snr:.2f} dB")
                    print(f"Output SNR: {output_snr:.2f} dB")
                    print(f"SNR Change: {snr_change:.2f} dB")
                
                print(f"Input THD: {snr_results['Input_THD_dB']:.2f} dB")
                print(f"Output THD: {snr_results['Output_THD_dB']:.2f} dB")
                
                results.append(result_row)
            else:
                print("Could not calculate SNR")
                results.append({
                    'Trial': trial, 
                    'Input_SNR_dB': None,
                    'Output_SNR_dB': None,
                    'SNR_Change_dB': None,
                    'Input_THD_dB': None,
                    'Output_THD_dB': None,
                    'Input_Vpp': ch1_pp,
                    'Output_Vpp': ch2_pp,
                    'Signal Shape': signal_shape,
                    'Signal Frequency': signal_frequency,
                    'Timestamp': current_timestamp
                })
        else:
            print("No data captured")
            current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results.append({
                'Trial': trial, 
                'Input_SNR_dB': None,
                'Output_SNR_dB': None,
                'SNR_Change_dB': None,
                'Input_THD_dB': None,
                'Output_THD_dB': None,
                'Input_Vpp': None,
                'Output_Vpp': None,
                'Signal Shape': signal_shape,
                'Signal Frequency': signal_frequency,
                'Timestamp': current_timestamp
            })
        
        # Wait before next trial
        time.sleep(2)
    
    # Save summary results
    pd.DataFrame(results).to_csv(summary_file, index=False)
    
    # Calculate statistics if we have data
    valid_results = [r for r in results if r['Input_SNR_dB'] is not None and r['Output_SNR_dB'] is not None]
    if valid_results:
        input_snr_values = [r['Input_SNR_dB'] for r in valid_results]
        output_snr_values = [r['Output_SNR_dB'] for r in valid_results]
        snr_change_values = [r['SNR_Change_dB'] for r in valid_results]
        input_thd_values = [r['Input_THD_dB'] for r in valid_results]
        output_thd_values = [r['Output_THD_dB'] for r in valid_results]
        input_vpp_values = [r['Input_Vpp'] for r in valid_results]
        output_vpp_values = [r['Output_Vpp'] for r in valid_results]
        
        stats = {
            "Mean Input SNR (dB)": np.mean(input_snr_values),
            "Std Dev Input SNR (dB)": np.std(input_snr_values),
            "Mean Output SNR (dB)": np.mean(output_snr_values),
            "Std Dev Output SNR (dB)": np.std(output_snr_values),
            "Mean SNR Change (dB)": np.mean(snr_change_values),
            "Std Dev SNR Change (dB)": np.std(snr_change_values),
            "Mean Input THD (dB)": np.mean(input_thd_values),
            "Mean Output THD (dB)": np.mean(output_thd_values),
            "Mean Input Vpp (V)": np.mean(input_vpp_values),
            "Mean Output Vpp (V)": np.mean(output_vpp_values),
            "Valid Measurements": f"{len(valid_results)}/{num_trials}",
            "Test Date/Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Add corrected stats if available
        if "Corrected_Input_SNR_dB" in valid_results[0]:
            corrected_input_snr_values = [r['Corrected_Input_SNR_dB'] for r in valid_results]
            corrected_output_snr_values = [r['Corrected_Output_SNR_dB'] for r in valid_results]
            corrected_snr_change_values = [r['Corrected_SNR_Change_dB'] for r in valid_results]
            
            stats["Mean Corrected Input SNR (dB)"] = np.mean(corrected_input_snr_values)
            stats["Std Dev Corrected Input SNR (dB)"] = np.std(corrected_input_snr_values)
            stats["Mean Corrected Output SNR (dB)"] = np.mean(corrected_output_snr_values)
            stats["Std Dev Corrected Output SNR (dB)"] = np.std(corrected_output_snr_values)
            stats["Mean Corrected SNR Change (dB)"] = np.mean(corrected_snr_change_values)
            stats["Std Dev Corrected SNR Change (dB)"] = np.std(corrected_snr_change_values)
        
        print("\nSNR Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
        
        # Save statistics to a separate file
        stats_file = os.path.join(signal_dir, f"statistics.txt")
        with open(stats_file, 'w') as f:
            f.write(f"SNR Statistics for {effect_config} with {signal_shape} at {signal_frequency}:\n")
            f.write(f"Test signal amplitude: {amplitude_vpp} Vpp\n")
            f.write(f"Test date/time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for key, value in stats.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.2f}\n")
                else:
                    f.write(f"{key}: {value}\n")
        
        # Generate comparative plots
        plt.figure(figsize=(10, 8))
        
        if "Corrected_Output_SNR_dB" in valid_results[0]:
            plt.subplot(2, 1, 1)
            plt.plot(range(1, len(valid_results)+1), [r['Input_SNR_dB'] for r in valid_results], 'bo-', label='Input SNR')
            plt.plot(range(1, len(valid_results)+1), [r['Output_SNR_dB'] for r in valid_results], 'ro-', label='Output SNR')
            plt.title(f"Raw SNR Comparison - {effect_config} - {signal_shape} at {signal_frequency}")
            plt.ylabel("SNR (dB)")
            plt.grid(True)
            plt.legend()
            
            # Add mean lines
            plt.axhline(y=np.mean(input_snr_values), color='b', linestyle='--', alpha=0.5)
            plt.axhline(y=np.mean(output_snr_values), color='r', linestyle='--', alpha=0.5)
            
            plt.subplot(2, 1, 2)
            plt.plot(range(1, len(valid_results)+1), [r['Corrected_Input_SNR_dB'] for r in valid_results], 'bo-', label='Corrected Input SNR')
            plt.plot(range(1, len(valid_results)+1), [r['Corrected_Output_SNR_dB'] for r in valid_results], 'ro-', label='Corrected Output SNR')
            plt.title(f"Corrected SNR Comparison - {effect_config} - {signal_shape} at {signal_frequency}")
            plt.xlabel("Trial")
            plt.ylabel("SNR (dB)")
            plt.grid(True)
            plt.legend()
            
            # Add mean lines
            plt.axhline(y=np.mean(corrected_input_snr_values), color='b', linestyle='--', alpha=0.5)
            plt.axhline(y=np.mean(corrected_output_snr_values), color='r', linestyle='--', alpha=0.5)
        else:
            plt.plot(range(1, len(valid_results)+1), [r['Input_SNR_dB'] for r in valid_results], 'bo-', label='Input SNR')
            plt.plot(range(1, len(valid_results)+1), [r['Output_SNR_dB'] for r in valid_results], 'ro-', label='Output SNR')
            plt.title(f"SNR Comparison - {effect_config} - {signal_shape} at {signal_frequency}")
            plt.xlabel("Trial")
            plt.ylabel("SNR (dB)")
            plt.grid(True)
            plt.legend()
            
            # Add mean lines
            plt.axhline(y=np.mean(input_snr_values), color='b', linestyle='--', alpha=0.5)
            plt.axhline(y=np.mean(output_snr_values), color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        # Save plot
        summary_plot_file = os.path.join(signal_dir, "snr_comparison_summary.png")
        plt.savefig(summary_plot_file)
        plt.close()
        
        print(f"Saved summary comparison plot to {summary_plot_file}")
    
    return signal_dir

# Main program
if __name__ == "__main__":
    try:
        # Ask if user wants to measure noise floor
        print("Welcome to the High-SNR Audio Measurement Tool")
        print("==============================================")
        noise_floor_data = None
        
        if input("\nWould you like to measure the system noise floor first? (Recommended for high-SNR measurements) (y/n): ").lower().startswith('y'):
            noise_floor_data = measure_noise_floor()
        
        # Run SNR tests
        while True:
            run_snr_test(noise_floor_data)
            if input("\nWould you like to run another test? (y/n): ").lower() != 'y':
                break
    finally:
        # Always close the scope connection
        try:
            scope.close()
            print("Oscilloscope connection closed.")
        except:
            pass
        
        print("\nSNR measurements complete!")