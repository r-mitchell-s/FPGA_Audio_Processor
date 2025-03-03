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

# Basic scope setup
print("Setting up oscilloscope...")
scope.write("*RST")  # Reset the scope
time.sleep(2)  # Wait for reset to complete

# Explicitly enable both channels
scope.write("SELECT:CH1 ON")  # Enable Channel 1 (input signal)
time.sleep(0.5)
scope.write("SELECT:CH2 ON")  # Enable Channel 2 (output signal)
time.sleep(0.5)

# Configure vertical settings - adjust based on your signal levels
scope.write("CH1:SCALE 0.2")  # 200mV/div for input
time.sleep(0.5)
scope.write("CH2:SCALE 0.2")  # 200mV/div for output
time.sleep(0.5)

# Configure horizontal timebase - capture multiple cycles of the test signal
scope.write("HORIZONTAL:SCALE 0.002")  # 2ms/div
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

def capture_waveforms():
    """Capture waveform data from both channels"""
    try:
        # Setup acquisition
        scope.write("ACQUIRE:STATE STOP")
        time.sleep(0.5)
        scope.write("ACQUIRE:MODE SAMPLE")  # Use sample mode for spectral analysis
        time.sleep(0.5)
        scope.write("ACQUIRE:STOPAFTER SEQUENCE")
        time.sleep(0.5)
        scope.write("ACQUIRE:STATE RUN")
        
        # Wait for acquisition to complete
        print("Waiting for trigger...")
        time.sleep(5)  # Fixed wait time
        
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

def calculate_snr(data):
    """Calculate Signal-to-Noise Ratio using FFT method"""
    try:
        # Extract data and sample rate
        input_signal = np.array(data["CH1"])
        output_signal = np.array(data["CH2"])
        sample_rate = data["SampleRate"]
        
        if len(input_signal) < 100 or len(output_signal) < 100:
            print("Insufficient data for SNR calculation")
            return None
        
        # Apply window to reduce spectral leakage
        window = signal.windows.hann(len(input_signal))
        input_windowed = input_signal * window
        output_windowed = output_signal * window
        
        # Compute FFTs
        input_fft = np.fft.rfft(input_windowed)
        output_fft = np.fft.rfft(output_windowed)
        
        # Calculate magnitudes (convert to dB)
        input_magnitude = np.abs(input_fft)
        output_magnitude = np.abs(output_fft)
        
        # Find the fundamental frequency (highest peak)
        input_peak_idx = np.argmax(input_magnitude)
        output_peak_idx = np.argmax(output_magnitude)
        
        # Extract signal power (at fundamental)
        input_signal_power = input_magnitude[input_peak_idx]**2
        output_signal_power = output_magnitude[output_peak_idx]**2
        
        # Calculate noise floor (excluding the fundamental and harmonics)
        # Create a mask to exclude fundamental and harmonics
        input_mask = np.ones_like(input_magnitude, dtype=bool)
        output_mask = np.ones_like(output_magnitude, dtype=bool)
        
        # Mask out fundamental and harmonics (and bins around them to account for leakage)
        harmonic_width = 5  # Width of window around each harmonic to exclude
        for h in range(1, 6):  # Consider up to 5 harmonics
            # Fundamental and harmonics
            h_idx_in = input_peak_idx * h
            h_idx_out = output_peak_idx * h
            
            # Skip if beyond FFT range
            if h_idx_in < len(input_mask):
                # Mask a window around each harmonic
                left_idx = max(0, h_idx_in - harmonic_width)
                right_idx = min(len(input_mask), h_idx_in + harmonic_width + 1)
                input_mask[left_idx:right_idx] = False
            
            if h_idx_out < len(output_mask):
                left_idx = max(0, h_idx_out - harmonic_width)
                right_idx = min(len(output_mask), h_idx_out + harmonic_width + 1)
                output_mask[left_idx:right_idx] = False
        
        # Also exclude DC component
        input_mask[0:5] = False
        output_mask[0:5] = False
        
        # Calculate noise power (average of all non-harmonic components)
        input_noise_power = np.mean(input_magnitude[input_mask]**2) if np.any(input_mask) else 1e-10
        output_noise_power = np.mean(output_magnitude[output_mask]**2) if np.any(output_mask) else 1e-10
        
        # Calculate SNR in dB
        input_snr_db = 10 * np.log10(input_signal_power / input_noise_power)
        output_snr_db = 10 * np.log10(output_signal_power / output_noise_power)
        
        # Calculate THD (Total Harmonic Distortion)
        # Sum power of first 5 harmonics (excluding fundamental)
        input_harmonics_power = 0
        output_harmonics_power = 0
        
        for h in range(2, 6):  # 2nd to 5th harmonics
            h_idx_in = input_peak_idx * h
            h_idx_out = output_peak_idx * h
            
            if h_idx_in < len(input_magnitude):
                # Sum a few bins around the harmonic to account for leakage
                h_width = 2
                left_idx = max(0, h_idx_in - h_width)
                right_idx = min(len(input_magnitude), h_idx_in + h_width + 1)
                input_harmonics_power += np.sum(input_magnitude[left_idx:right_idx]**2)
            
            if h_idx_out < len(output_magnitude):
                h_width = 2
                left_idx = max(0, h_idx_out - h_width)
                right_idx = min(len(output_magnitude), h_idx_out + h_width + 1)
                output_harmonics_power += np.sum(output_magnitude[left_idx:right_idx]**2)
        
        # Calculate THD in dB
        input_thd_db = 10 * np.log10(input_harmonics_power / input_signal_power) if input_signal_power > 0 else -100
        output_thd_db = 10 * np.log10(output_harmonics_power / output_signal_power) if output_signal_power > 0 else -100
        
        # Calculate frequency axis for plotting
        freq_axis = np.fft.rfftfreq(len(input_signal), 1/sample_rate)
        
        # Store results
        result = {
            "Input_SNR_dB": input_snr_db,
            "Output_SNR_dB": output_snr_db,
            "Input_THD_dB": input_thd_db,
            "Output_THD_dB": output_thd_db,
            "Frequencies": freq_axis,
            "Input_FFT": input_magnitude,
            "Output_FFT": output_magnitude,
            "Input_Peak_Idx": input_peak_idx,
            "Output_Peak_Idx": output_peak_idx,
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
        plt.figure(figsize=(12, 8))
        
        # Time domain subplot
        plt.subplot(2, 1, 1)
        plt.plot(data["Time"], data["CH1"], label="Input (CH1)")
        plt.plot(data["Time"], data["CH2"], label="Output (CH2)")
        plt.title(f"Time Domain - {effect_config} - {signal_info['shape']} {signal_info['frequency']} - Trial {trial}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (V)")
        plt.grid(True)
        plt.legend()
        
        # Frequency domain subplot
        plt.subplot(2, 1, 2)
        
        # Plot FFT magnitude in dB scale
        if snr_results:
            freq = snr_results["Frequencies"]
            input_mag_db = 20 * np.log10(snr_results["Input_FFT"] + 1e-10)  # +1e-10 to avoid log(0)
            output_mag_db = 20 * np.log10(snr_results["Output_FFT"] + 1e-10)
            
            # Only show up to 20kHz (audio range)
            cutoff_idx = np.searchsorted(freq, 20000)
            freq = freq[:cutoff_idx]
            input_mag_db = input_mag_db[:cutoff_idx]
            output_mag_db = output_mag_db[:cutoff_idx]
            
            plt.plot(freq, input_mag_db, label=f"Input (SNR: {snr_results['Input_SNR_dB']:.1f} dB)")
            plt.plot(freq, output_mag_db, label=f"Output (SNR: {snr_results['Output_SNR_dB']:.1f} dB)")
            
            # Mark the fundamental and harmonics
            input_peak_freq = freq[snr_results["Input_Peak_Idx"]] if snr_results["Input_Peak_Idx"] < len(freq) else 0
            output_peak_freq = freq[snr_results["Output_Peak_Idx"]] if snr_results["Output_Peak_Idx"] < len(freq) else 0
            
            plt.axvline(x=input_peak_freq, color='green', linestyle='--', alpha=0.5)
            plt.axvline(x=output_peak_freq, color='red', linestyle='--', alpha=0.5)
            
            # Add annotations about THD
            plt.annotate(f"Input THD: {snr_results['Input_THD_dB']:.1f} dB", 
                        xy=(0.05, 0.15), xycoords='axes fraction')
            plt.annotate(f"Output THD: {snr_results['Output_THD_dB']:.1f} dB", 
                        xy=(0.05, 0.08), xycoords='axes fraction')
            
        plt.title("Frequency Domain")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.grid(True)
        plt.legend()
        plt.xscale('log')
        
        plt.tight_layout()
        
        # Save the plot
        plot_file = os.path.join(plots_dir, f"snr_plot_trial{trial}.png")
        plt.savefig(plot_file)
        plt.close()
        print(f"Saved verification plot to {plot_file}")
        
    except Exception as e:
        print(f"Error generating SNR plots: {e}")
        import traceback
        traceback.print_exc()

# Main measurement loop
print("\nReady to start SNR measurements")
effect_config = input("Enter effect configuration (bypass, lowpass, distortion, both): ")
signal_shape = input("Enter input signal shape (sine, square, etc.): ")
signal_frequency = input("Enter input signal frequency (e.g., 1kHz): ")

# Store signal information
signal_info = {
    "shape": signal_shape,
    "frequency": signal_frequency
}

# Create signal folder name (sanitize for filesystem)
signal_folder = f"{signal_shape}_{signal_frequency}"
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
with open(summary_file, 'w', newline='') as f:
    # Include signal information in the summary file
    columns = [
        'Trial', 
        'Input_SNR_dB', 
        'Output_SNR_dB', 
        'SNR_Change_dB', 
        'Input_THD_dB', 
        'Output_THD_dB', 
        'Signal Shape', 
        'Signal Frequency'
    ]
    pd.DataFrame(columns=columns).to_csv(f, index=False)

num_trials = 10  # SNR measurement needs fewer trials than latency
print(f"\nStarting {num_trials} SNR measurements for {effect_config} with {signal_shape} at {signal_frequency}...")

results = []
for trial in range(1, num_trials + 1):
    print(f"\nTrial {trial}/{num_trials}")
    
    # Capture waveforms
    data = capture_waveforms()
    
    # Save waveform data
    if data["Time"]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        waveform_file = os.path.join(trials_dir, f"snr_trial{trial}_{timestamp}.csv")
        
        # Remove SampleRate before saving CSV
        waveform_data = {k: data[k] for k in ["Time", "CH1", "CH2"]}
        pd.DataFrame(waveform_data).to_csv(waveform_file, index=False)
        print(f"Saved waveform data to {waveform_file}")
        
        # Calculate SNR
        snr_results = calculate_snr(data)
        
        # Generate verification plot
        generate_snr_plots(data, snr_results, trial, effect_config, signal_dir, signal_info)
        
        if snr_results is not None:
            input_snr = snr_results["Input_SNR_dB"]
            output_snr = snr_results["Output_SNR_dB"]
            snr_change = output_snr - input_snr
            
            print(f"Input SNR: {input_snr:.2f} dB")
            print(f"Output SNR: {output_snr:.2f} dB")
            print(f"SNR Change: {snr_change:.2f} dB")
            
            results.append({
                'Trial': trial, 
                'Input_SNR_dB': input_snr,
                'Output_SNR_dB': output_snr,
                'SNR_Change_dB': snr_change,
                'Input_THD_dB': snr_results["Input_THD_dB"],
                'Output_THD_dB': snr_results["Output_THD_dB"],
                'Signal Shape': signal_shape,
                'Signal Frequency': signal_frequency
            })
        else:
            print("Could not calculate SNR")
            results.append({
                'Trial': trial, 
                'Input_SNR_dB': None,
                'Output_SNR_dB': None,
                'SNR_Change_dB': None,
                'Input_THD_dB': None,
                'Output_THD_dB': None,
                'Signal Shape': signal_shape,
                'Signal Frequency': signal_frequency
            })
    else:
        print("No data captured")
        results.append({
            'Trial': trial, 
            'Input_SNR_dB': None,
            'Output_SNR_dB': None,
            'SNR_Change_dB': None,
            'Input_THD_dB': None,
            'Output_THD_dB': None,
            'Signal Shape': signal_shape,
            'Signal Frequency': signal_frequency
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
    
    stats = {
        "Mean Input SNR (dB)": np.mean(input_snr_values),
        "Std Dev Input SNR (dB)": np.std(input_snr_values),
        "Mean Output SNR (dB)": np.mean(output_snr_values),
        "Std Dev Output SNR (dB)": np.std(output_snr_values),
        "Mean SNR Change (dB)": np.mean(snr_change_values),
        "Std Dev SNR Change (dB)": np.std(snr_change_values),
        "Valid Measurements": f"{len(valid_results)}/{num_trials}"
    }
    
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
        for key, value in stats.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.2f}\n")
            else:
                f.write(f"{key}: {value}\n")

scope.close()
print("\nSNR measurements complete!")