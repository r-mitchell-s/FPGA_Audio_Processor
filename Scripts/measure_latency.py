# single_pulse_latency_organized.py
import pyvisa
import numpy as np
import time
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt

# Create top-level output directory in ../Documents relative to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.normpath(os.path.join(script_dir, "..", "Documents", "latency_measurements"))
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
scope.timeout = 20000  # 20 seconds

# Basic scope setup
print("Setting up oscilloscope...")
scope.write("*RST")  # Reset the scope
time.sleep(2)  # Wait for reset to complete

# Explicitly enable both channels
scope.write("SELECT:CH1 ON")  # Enable Channel 1
time.sleep(0.5)
scope.write("SELECT:CH2 ON")  # Enable Channel 2
time.sleep(0.5)

# Configure vertical settings
scope.write("CH1:SCALE 0.5")  # Adjust voltage scale for CH1
time.sleep(0.5)
scope.write("CH2:SCALE 0.5")  # Adjust voltage scale for CH2
time.sleep(0.5)

# Configure horizontal timebase - focus on a single pulse
scope.write("HORIZONTAL:SCALE 0.001")  # 1ms/div - to capture just one pulse
time.sleep(0.5)

# Configure trigger
scope.write("TRIGGER:A:TYPE EDGE")
time.sleep(0.5)
scope.write("TRIGGER:A:EDGE:SOURCE CH1")
time.sleep(0.5)
scope.write("TRIGGER:A:EDGE:SLOPE RISE")
time.sleep(0.5)
scope.write("TRIGGER:A:LEVEL 0.5")  # Adjust based on signal
time.sleep(0.5)

def capture_waveforms():
    try:
        # Setup acquisition
        scope.write("ACQUIRE:STATE STOP")
        time.sleep(0.5)
        scope.write("ACQUIRE:STOPAFTER SEQUENCE")
        time.sleep(0.5)
        scope.write("ACQUIRE:STATE RUN")
        
        # Wait for acquisition to complete
        print("Waiting for trigger...")
        time.sleep(3)  # Fixed wait time
        
        # Get data from both channels
        waveform_data = {"Time": [], "CH1": [], "CH2": []}
        
        # Process Channel 1
        print("Getting Channel 1 data...")
        scope.write("DATA:SOURCE CH1")
        time.sleep(0.5)
        scope.write("DATA:ENCDG ASCII")
        time.sleep(0.5)
        record_length = int(scope.query("HORIZONTAL:RECORDLENGTH?").strip())
        scope.write(f"DATA:START 1")
        time.sleep(0.5)
        scope.write(f"DATA:STOP {record_length}")
        time.sleep(0.5)
        
        # Get CH1 scaling
        x_incr = float(scope.query("WFMPRE:XINCR?").strip())
        x_zero = float(scope.query("WFMPRE:XZERO?").strip())
        y_mult1 = float(scope.query("WFMPRE:YMULT?").strip())
        y_zero1 = float(scope.query("WFMPRE:YZERO?").strip())
        y_offset1 = float(scope.query("WFMPRE:YOFF?").strip())
        
        # Get CH1 data
        ch1_raw = scope.query("CURVE?")
        ch1_values = ch1_raw.split(',')
        
        # Process Channel 2
        print("Getting Channel 2 data...")
        scope.write("DATA:SOURCE CH2")
        time.sleep(0.5)
        
        # Get CH2 scaling
        y_mult2 = float(scope.query("WFMPRE:YMULT?").strip())
        y_zero2 = float(scope.query("WFMPRE:YZERO?").strip())
        y_offset2 = float(scope.query("WFMPRE:YOFF?").strip())
        
        # Get CH2 data
        ch2_raw = scope.query("CURVE?")
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
        
        return {
            "Time": time_axis,
            "CH1": ch1_voltages,
            "CH2": ch2_voltages
        }
        
    except Exception as e:
        print(f"Error capturing data: {e}")
        import traceback
        traceback.print_exc()
        return {"Time": [], "CH1": [], "CH2": []}

def calculate_single_pulse_latency(data):
    """Calculate latency focusing on a single pulse"""
    try:
        if not data["Time"] or not data["CH1"] or not data["CH2"]:
            return None
        
        # Find thresholds (50% of peak-to-peak)
        ch1_max, ch1_min = max(data["CH1"]), min(data["CH1"])
        ch2_max, ch2_min = max(data["CH2"]), min(data["CH2"])
        
        ch1_threshold = ch1_min + (ch1_max - ch1_min) * 0.5
        ch2_threshold = ch2_min + (ch2_max - ch2_min) * 0.5
        
        # Since we're focused on a single pulse with the trigger on CH1,
        # we should see only one major rising edge in each channel
        
        # Find the first rising edge in CH1 (should be near the trigger point)
        ch1_idx = None
        for i in range(1, len(data["Time"])):
            if data["CH1"][i-1] <= ch1_threshold and data["CH1"][i] > ch1_threshold:
                ch1_idx = i
                break
        
        # Find the first rising edge in CH2 after the CH1 edge
        ch2_idx = None
        if ch1_idx is not None:
            for i in range(ch1_idx, len(data["Time"])):
                if data["CH2"][i-1] <= ch2_threshold and data["CH2"][i] > ch2_threshold:
                    ch2_idx = i
                    break
        
        if ch1_idx is not None and ch2_idx is not None:
            # Calculate more precise crossing times using linear interpolation
            t1 = data["Time"][ch1_idx-1] + (data["Time"][ch1_idx] - data["Time"][ch1_idx-1]) * \
                 (ch1_threshold - data["CH1"][ch1_idx-1]) / (data["CH1"][ch1_idx] - data["CH1"][ch1_idx-1])
                 
            t2 = data["Time"][ch2_idx-1] + (data["Time"][ch2_idx] - data["Time"][ch2_idx-1]) * \
                 (ch2_threshold - data["CH2"][ch2_idx-1]) / (data["CH2"][ch2_idx] - data["CH2"][ch2_idx-1])
            
            latency = t2 - t1
            
            # Sanity check - FPGA latency should be microseconds to low milliseconds
            if 0 < latency < 0.01:  # Less than 10ms
                return latency, ch1_idx, ch2_idx
        
        return None, None, None
            
    except Exception as e:
        print(f"Error calculating latency: {e}")
        return None, None, None

def generate_verification_plot(data, latency_info, trial, effect_config, signal_dir, signal_info):
    """Generate a plot showing the pulses and latency calculation"""
    try:
        # Save plots in a dedicated plots folder within the signal directory
        plots_dir = os.path.join(signal_dir, "verification_plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        latency, ch1_idx, ch2_idx = latency_info
        
        plt.figure(figsize=(10, 6))
        plt.plot(data["Time"], data["CH1"], label="Input (CH1)")
        plt.plot(data["Time"], data["CH2"], label="Output (CH2)")
        
        if ch1_idx is not None and ch2_idx is not None:
            # Calculate thresholds
            ch1_max, ch1_min = max(data["CH1"]), min(data["CH1"])
            ch2_max, ch2_min = max(data["CH2"]), min(data["CH2"])
            ch1_threshold = ch1_min + (ch1_max - ch1_min) * 0.5
            ch2_threshold = ch2_min + (ch2_max - ch2_min) * 0.5
            
            # Mark the threshold crossings
            plt.axvline(x=data["Time"][ch1_idx], color='green', linestyle='--', 
                        label=f'CH1 crossing: {data["Time"][ch1_idx]*1000:.3f} ms')
            plt.axvline(x=data["Time"][ch2_idx], color='red', linestyle='--', 
                        label=f'CH2 crossing: {data["Time"][ch2_idx]*1000:.3f} ms')
            
            plt.axhline(y=ch1_threshold, color='green', linestyle=':', alpha=0.5)
            plt.axhline(y=ch2_threshold, color='red', linestyle=':', alpha=0.5)
            
            # Draw arrow showing the latency
            arrow_y = max(ch1_max, ch2_max) * 1.1
            plt.annotate('', 
                        xy=(data["Time"][ch2_idx], arrow_y), 
                        xytext=(data["Time"][ch1_idx], arrow_y),
                        arrowprops=dict(arrowstyle='<->', color='black'))
            
            plt.text((data["Time"][ch1_idx] + data["Time"][ch2_idx])/2, arrow_y*1.05, 
                    f'Latency: {latency*1000:.3f} ms', 
                    horizontalalignment='center')
        
        # Set title with signal information
        signal_desc = f"{signal_info['shape']} {signal_info['frequency']}"
        if latency is not None:
            plt.title(f"Latency: {effect_config} - {signal_desc} - Trial {trial} - {latency*1000:.3f} ms")
        else:
            plt.title(f"Latency: {effect_config} - {signal_desc} - Trial {trial} - Could not determine latency")
        
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.legend()
        plt.grid(True)
        
        # Save the plot to the plots directory
        plot_file = os.path.join(plots_dir, f"plot_trial{trial}.png")
        plt.savefig(plot_file)
        plt.close()
        print(f"Saved verification plot to {plot_file}")
        
    except Exception as e:
        print(f"Error generating verification plot: {e}")

# Main measurement loop
print("\nReady to start measurements")
effect_config = input("Enter effect configuration (bypass, lowpass, distortion, both): ")
signal_shape = input("Enter input signal shape (sine, square, pulse, etc.): ")
signal_frequency = input("Enter input signal frequency (e.g., 20Hz, 1kHz): ")

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
summary_file = os.path.join(signal_dir, f"latency_summary.csv")
with open(summary_file, 'w', newline='') as f:
    # Include signal information in the summary file
    columns = ['Trial', 'Latency (s)', 'Latency (ms)', 'Latency (µs)', 'Signal Shape', 'Signal Frequency']
    pd.DataFrame(columns=columns).to_csv(f, index=False)

num_trials = 20
print(f"\nStarting {num_trials} latency measurements for {effect_config} with {signal_shape} at {signal_frequency}...")

results = []
for trial in range(1, num_trials + 1):
    print(f"\nTrial {trial}/{num_trials}")
    
    # Capture waveforms
    data = capture_waveforms()
    
    # Save waveform data
    if data["Time"]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        waveform_file = os.path.join(trials_dir, f"latency_trial{trial}_{timestamp}.csv")
        pd.DataFrame(data).to_csv(waveform_file, index=False)
        print(f"Saved waveform data to {waveform_file}")
        
        # Calculate latency
        latency_info = calculate_single_pulse_latency(data)
        latency = latency_info[0] if latency_info[0] is not None else None
        
        # Generate verification plot
        generate_verification_plot(data, latency_info, trial, effect_config, signal_dir, signal_info)
        
        if latency is not None:
            latency_ms = latency * 1000  # Convert to milliseconds
            latency_us = latency * 1000000  # Convert to microseconds
            print(f"Latency: {latency_ms:.3f} ms ({latency_us:.1f} µs)")
            results.append({
                'Trial': trial, 
                'Latency (s)': latency, 
                'Latency (ms)': latency_ms,
                'Latency (µs)': latency_us,
                'Signal Shape': signal_shape,
                'Signal Frequency': signal_frequency
            })
        else:
            print("Could not calculate latency")
            results.append({
                'Trial': trial, 
                'Latency (s)': None, 
                'Latency (ms)': None,
                'Latency (µs)': None,
                'Signal Shape': signal_shape,
                'Signal Frequency': signal_frequency
            })
    else:
        print("No data captured")
        results.append({
            'Trial': trial, 
            'Latency (s)': None, 
            'Latency (ms)': None,
            'Latency (µs)': None,
            'Signal Shape': signal_shape,
            'Signal Frequency': signal_frequency
        })
    
    # Wait before next trial
    time.sleep(2)

# Save summary results
pd.DataFrame(results).to_csv(summary_file, index=False)

# Calculate statistics if we have data
valid_results = [r['Latency (µs)'] for r in results if r['Latency (µs)'] is not None]
if valid_results:
    mean_latency = np.mean(valid_results)
    std_latency = np.std(valid_results)
    min_latency = np.min(valid_results)
    max_latency = np.max(valid_results)
    
    print(f"\nLatency Statistics for {effect_config} with {signal_shape} at {signal_frequency}:")
    print(f"Mean: {mean_latency:.1f} µs")
    print(f"Std Dev: {std_latency:.1f} µs")
    print(f"Min: {min_latency:.1f} µs")
    print(f"Max: {max_latency:.1f} µs")
    print(f"Valid measurements: {len(valid_results)}/{num_trials}")

    # Save statistics to a separate file
    stats_file = os.path.join(signal_dir, f"statistics.txt")
    with open(stats_file, 'w') as f:
        f.write(f"Latency Statistics for {effect_config} with {signal_shape} at {signal_frequency}:\n")
        f.write(f"Mean: {mean_latency:.1f} µs\n")
        f.write(f"Std Dev: {std_latency:.1f} µs\n")
        f.write(f"Min: {min_latency:.1f} µs\n")
        f.write(f"Max: {max_latency:.1f} µs\n")
        f.write(f"Valid measurements: {len(valid_results)}/{num_trials}\n")

scope.close()
print("\nMeasurements complete!")