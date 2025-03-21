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

def setup_oscilloscope(effect_config=None):
    """Setup the oscilloscope with proper settings for latency measurement"""
    print("Setting up oscilloscope...")
    scope.write("*RST")  # Reset the scope
    time.sleep(2)  # Wait for reset to complete

    # Explicitly enable both channels
    scope.write("SELECT:CH1 ON")  # Enable Channel 1
    time.sleep(0.5)
    scope.write("SELECT:CH2 ON")  # Enable Channel 2
    time.sleep(0.5)

    # Configure vertical settings based on effect
    # For lowpass, we need more sensitive CH2 scale since the signal is attenuated
    if effect_config == "lowpass":
        scope.write("CH1:SCALE 0.5")  # Adjust voltage scale for CH1
        time.sleep(0.5)
        scope.write("CH2:SCALE 0.05")  # More sensitive scale for CH2 with lowpass
        time.sleep(0.5)
        print("Using more sensitive scale for output channel with lowpass filter")
    else:
        scope.write("CH1:SCALE 0.5")  # Adjust voltage scale for CH1
        time.sleep(0.5)
        scope.write("CH2:SCALE 0.1")  # Adjust voltage scale for CH2
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

def calculate_single_pulse_latency(data, effect_config=None):
    """Calculate latency focusing on a single pulse"""
    try:
        if not data["Time"] or not data["CH1"] or not data["CH2"]:
            return None
        
        # Find thresholds (50% of peak-to-peak by default)
        ch1_max, ch1_min = max(data["CH1"]), min(data["CH1"])
        ch2_max, ch2_min = max(data["CH2"]), min(data["CH2"])
        
        # For different effects, use appropriate threshold percentages
        ch1_threshold_percent = 0.5  # 50% for input channel (default)
        ch2_threshold_percent = 0.5  # Default 50% for output channel
        
        # Special handling for different effects
        if effect_config == "lowpass":
            ch2_threshold_percent = 0.3  # Lower threshold (30%) for lowpass to detect smaller peaks
            print(f"Using lower threshold ({ch2_threshold_percent*100}%) for lowpass output detection")
        elif effect_config == "distortion":
            # For distortion, check both rising and falling edges
            print("Distortion effect detected: Checking both rising and falling edges")
        
        ch1_threshold = ch1_min + (ch1_max - ch1_min) * ch1_threshold_percent
        ch2_threshold = ch2_min + (ch2_max - ch2_min) * ch2_threshold_percent
        
        # Find the first rising edge in CH1 (should be near the trigger point)
        ch1_idx = None
        for i in range(1, len(data["Time"])):
            if data["CH1"][i-1] <= ch1_threshold and data["CH1"][i] > ch1_threshold:
                ch1_idx = i
                break
        
        # Find edge in CH2 based on effect
        ch2_idx = None
        if ch1_idx is not None:
            if effect_config == "distortion":
                # Check for both rising and falling edges
                rising_edge_idx = None
                falling_edge_idx = None
                
                for i in range(ch1_idx, min(ch1_idx + 10000, len(data["Time"]))):
                    # Check for rising edge
                    if i > 0 and data["CH2"][i-1] <= ch2_threshold and data["CH2"][i] > ch2_threshold:
                        rising_edge_idx = i
                        break
                
                for i in range(ch1_idx, min(ch1_idx + 10000, len(data["Time"]))):
                    # Check for falling edge
                    if i > 0 and data["CH2"][i-1] >= ch2_threshold and data["CH2"][i] < ch2_threshold:
                        falling_edge_idx = i
                        break
                
                # Use the first edge we find (either rising or falling)
                if rising_edge_idx is not None and falling_edge_idx is not None:
                    # Use whichever comes first
                    if rising_edge_idx < falling_edge_idx:
                        ch2_idx = rising_edge_idx
                        print("Using rising edge for distortion output")
                    else:
                        ch2_idx = falling_edge_idx
                        print("Using falling edge for distortion output")
                elif rising_edge_idx is not None:
                    ch2_idx = rising_edge_idx
                    print("Using rising edge for distortion output")
                elif falling_edge_idx is not None:
                    ch2_idx = falling_edge_idx
                    print("Using falling edge for distortion output")
            else:
                # Standard rising edge detection for other effects
                for i in range(ch1_idx, len(data["Time"])):
                    if data["CH2"][i-1] <= ch2_threshold and data["CH2"][i] > ch2_threshold:
                        ch2_idx = i
                        break
        
        if ch1_idx is not None and ch2_idx is not None:
            # Calculate more precise crossing times using linear interpolation
            t1 = data["Time"][ch1_idx-1] + (data["Time"][ch1_idx] - data["Time"][ch1_idx-1]) * \
                 (ch1_threshold - data["CH1"][ch1_idx-1]) / (data["CH1"][ch1_idx] - data["CH1"][ch1_idx-1])
            
            # Interpolate based on whether it's rising or falling
            if effect_config == "distortion" and data["CH2"][ch2_idx] < data["CH2"][ch2_idx-1]:
                # Falling edge interpolation
                t2 = data["Time"][ch2_idx-1] + (data["Time"][ch2_idx] - data["Time"][ch2_idx-1]) * \
                     (ch2_threshold - data["CH2"][ch2_idx-1]) / (data["CH2"][ch2_idx] - data["CH2"][ch2_idx-1])
            else:
                # Rising edge interpolation
                t2 = data["Time"][ch2_idx-1] + (data["Time"][ch2_idx] - data["Time"][ch2_idx-1]) * \
                     (ch2_threshold - data["CH2"][ch2_idx-1]) / (data["CH2"][ch2_idx] - data["CH2"][ch2_idx-1])
            
            latency = t2 - t1
            
            # Sanity check - FPGA latency should be microseconds to low milliseconds
            # For distortion, we want to specifically capture that ~0.6ms latency you observed
            if effect_config == "distortion":
                if 0 < latency < 0.01:  # Less than 10ms
                    return latency, ch1_idx, ch2_idx, ch1_threshold, ch2_threshold
            else:
                if 0 < latency < 0.01:  # Less than 10ms
                    return latency, ch1_idx, ch2_idx, ch1_threshold, ch2_threshold
        
        return None, None, None, None, None
            
    except Exception as e:
        print(f"Error calculating latency: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None

def generate_verification_plot(data, latency_info, trial, effect_config, signal_dir, signal_info):
    """Generate a plot showing the pulses and latency calculation"""
    try:
        # Save plots in a dedicated plots folder within the signal directory
        plots_dir = os.path.join(signal_dir, "verification_plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Unpack latency info (handling both old and new format)
        if len(latency_info) >= 6:  # New format with validity flag
            latency, ch1_idx, ch2_idx, ch1_threshold, ch2_threshold, validity = latency_info
        else:  # Old format without validity flag
            latency, ch1_idx, ch2_idx, ch1_threshold, ch2_threshold = latency_info if len(latency_info) == 5 else (*latency_info, None, None)
            validity = "valid" if latency is not None else None
        
        plt.figure(figsize=(10, 6))
        plt.plot(data["Time"], data["CH1"], label="Input (CH1)")
        plt.plot(data["Time"], data["CH2"], label="Output (CH2)")
        
        if ch1_idx is not None and ch2_idx is not None:
            # If thresholds are not provided, calculate them
            if ch1_threshold is None or ch2_threshold is None:
                # Calculate thresholds (as in the original function)
                ch1_max, ch1_min = max(data["CH1"]), min(data["CH1"])
                ch2_max, ch2_min = max(data["CH2"]), min(data["CH2"])
                ch1_threshold = ch1_min + (ch1_max - ch1_min) * 0.5
                ch2_threshold = ch2_min + (ch2_max - ch2_min) * 0.5
            
            # Mark the threshold crossings
            plt.axvline(x=data["Time"][ch1_idx], color='green', linestyle='--', 
                        label=f'CH1 crossing: {data["Time"][ch1_idx]*1000:.3f} ms')
            plt.axvline(x=data["Time"][ch2_idx], color='red', linestyle='--', 
                        label=f'CH2 crossing: {data["Time"][ch2_idx]*1000:.3f} ms')
            
            plt.axhline(y=ch1_threshold, color='green', linestyle=':', alpha=0.5,
                        label=f'CH1 threshold: {ch1_threshold:.3f}V')
            plt.axhline(y=ch2_threshold, color='red', linestyle=':', alpha=0.5,
                        label=f'CH2 threshold: {ch2_threshold:.3f}V')
            
            # Draw arrow showing the latency
            arrow_y = max(data["CH1"]) * 1.1
            plt.annotate('', 
                        xy=(data["Time"][ch2_idx], arrow_y), 
                        xytext=(data["Time"][ch1_idx], arrow_y),
                        arrowprops=dict(arrowstyle='<->', color='black'))
            
            # Annotate with latency value and validity
            latency_text = f'Latency: {latency*1000:.3f} ms'
            if validity == "below_threshold":
                latency_text += " (BELOW VALID RANGE)"
                text_color = 'red'
            elif validity == "above_threshold":
                latency_text += " (ABOVE VALID RANGE)"
                text_color = 'red'
            else:
                text_color = 'black'
                
            plt.text((data["Time"][ch1_idx] + data["Time"][ch2_idx])/2, arrow_y*1.05, 
                    latency_text, 
                    horizontalalignment='center',
                    color=text_color,
                    fontweight='bold' if validity != "valid" else 'normal')
        
        # Set title with signal information
        signal_desc = f"{signal_info['shape']} {signal_info['frequency']}"
        if latency is not None:
            valid_marker = ""
            if validity == "below_threshold":
                valid_marker = " [INVALID: TOO LOW]"
            elif validity == "above_threshold":
                valid_marker = " [INVALID: TOO HIGH]"
                
            plt.title(f"Latency: {effect_config} - {signal_desc} - Trial {trial} - {latency*1000:.3f} ms{valid_marker}")
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
        import traceback
        traceback.print_exc()

# Create a function to display available effects and get user choice
def get_effect_configuration():
    print("\nAvailable effect configurations:")
    print("1. bypass")
    print("2. lowpass")
    print("3. distortion")
    print("4. ring_modulator")
    print("5. noise_gate")
    print("6. all_effects (all effects active)")
    print("7. custom combination")
    
    while True:
        choice = input("\nEnter the number or the name of the effect configuration: ")
        
        # Check if input is a number
        if choice.isdigit():
            choice_num = int(choice)
            if choice_num == 1:
                return "bypass"
            elif choice_num == 2:
                return "lowpass"
            elif choice_num == 3:
                return "distortion"
            elif choice_num == 4:
                return "ring_modulator"
            elif choice_num == 5:
                return "noise_gate"
            elif choice_num == 6:
                return "all_effects"
            elif choice_num == 7:
                return input("Enter your custom combination of effects (e.g., 'lowpass+distortion'): ")
            else:
                print("Invalid option. Please enter a valid number.")
        else:
            # Check if input is a valid effect name
            valid_effects = ["bypass", "lowpass", "distortion", "ring_modulator", "noise_gate", "all_effects"]
            if choice in valid_effects or "+" in choice:
                return choice
            else:
                print("Invalid effect configuration. Please enter a valid option.")

# Function to run a batch of latency measurements
def run_batch_measurements():
    """Run batch measurements for multiple effect configurations"""
    # Get common configuration for all measurements
    signal_shape = input("Enter input signal shape (sine, square, pulse, etc.): ")
    signal_frequency = input("Enter input signal frequency (e.g., 20Hz, 1kHz): ")
    num_trials = int(input("Enter the number of valid trials to collect (default is 10): ") or 10)
    
    # Ask about latency range
    print("\nSpecify the valid latency range for measurements:")
    min_latency = float(input("Enter minimum valid latency in microseconds (default 450): ") or 450)
    max_latency = float(input("Enter maximum valid latency in microseconds (default 900): ") or 900)
    
    # Convert to seconds
    min_valid_latency = min_latency / 1e6
    max_valid_latency = max_latency / 1e6
    
    print(f"Valid latency range: {min_valid_latency*1e6:.0f}µs - {max_valid_latency*1e6:.0f}µs")
    
    # Ask if user wants to run a batch of measurements for all configurations
    run_batch = input("\nDo you want to run measurements for all predefined configurations? (y/n): ").lower() == 'y'
    
    if run_batch:
        configurations = ["bypass", "lowpass", "distortion", "ring_modulator", "noise_gate", "all_effects"]
        print(f"\nWill run measurements for: {', '.join(configurations)}")
        
        for effect_config in configurations:
            print(f"\n{'='*50}")
            print(f"NEXT CONFIGURATION: {effect_config}")
            print(f"{'='*50}")
            print(f"Please configure the FPGA switches for the {effect_config} effect.")
            input("Press ENTER when the FPGA is configured and ready for measurement...")
            
            # Initialize scope for this effect
            setup_oscilloscope(effect_config)
            
            print(f"\nStarting measurements for: {effect_config}")
            run_measurement(effect_config, signal_shape, signal_frequency, num_trials, min_valid_latency, max_valid_latency)
            
        print("\nAll batch measurements complete!")
    else:
        # Run a single configuration
        effect_config = get_effect_configuration()
        print(f"\nPlease configure the FPGA switches for the {effect_config} effect.")
        input("Press ENTER when the FPGA is configured and ready for measurement...")
        
        # Initialize scope
        setup_oscilloscope(effect_config)
        
        run_measurement(effect_config, signal_shape, signal_frequency, num_trials, min_valid_latency, max_valid_latency)

# Update the run_measurement function definition to accept the additional parameters
def run_measurement(effect_config, signal_shape, signal_frequency, num_trials, min_valid_latency=450e-6, max_valid_latency=900e-6):
    """Run latency measurements for a specific effect and signal
    
    Args:
        effect_config: The effect configuration to test
        signal_shape: Shape of the input signal
        signal_frequency: Frequency of the input signal
        num_trials: Number of trials to run
        min_valid_latency: Minimum valid latency in seconds (450µs default)
        max_valid_latency: Maximum valid latency in seconds (900µs default)
    """
    # Create timestamp for this measurement session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Store signal information
    signal_info = {
        "shape": signal_shape,
        "frequency": signal_frequency,
        "timestamp": timestamp
    }
    
    # Create signal folder name (sanitize for filesystem)
    signal_folder = f"{signal_shape}_{signal_frequency}_{timestamp}"
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
        # Include signal information and validity column in the summary file
        columns = [
            'Trial', 
            'Latency (s)', 
            'Latency (ms)', 
            'Latency (µs)', 
            'Signal Shape', 
            'Signal Frequency',
            'Timestamp',
            'Validity'  # Add validity column
        ]
        pd.DataFrame(columns=columns).to_csv(f, index=False)
    
    # Setup scope with effect-specific settings
    setup_oscilloscope(effect_config)
    
    print(f"\nStarting {num_trials} latency measurements for {effect_config} with {signal_shape} at {signal_frequency}...")
    print(f"Valid latency range: {min_valid_latency*1e6:.0f}µs - {max_valid_latency*1e6:.0f}µs")
    print(f"Measurement session timestamp: {timestamp}")
    
    # Effect-specific guidance and settings
    if effect_config == "lowpass":
        print("\nNote: For lowpass effect, the output signal may be attenuated.")
        print("Using more sensitive oscilloscope settings and lower threshold detection.")
        print("Please ensure the pulse is clearly visible on both channels.")
        if input("Would you like to adjust oscilloscope settings before continuing? (y/n): ").lower() == 'y':
            ch2_scale = input("Enter CH2 scale value (e.g. 0.05 for 50mV/div, default=0.05): ") or "0.05"
            scope.write(f"CH2:SCALE {ch2_scale}")
            time.sleep(0.5)
            print(f"CH2 scale set to {ch2_scale}V/div")
    
    # Try to auto-find a good trigger level based on actual waveform
    print("Capturing test waveform to optimize settings...")
    test_data = capture_waveforms()
    if test_data["CH1"]:
        ch1_max, ch1_min = max(test_data["CH1"]), min(test_data["CH1"])
        ch2_max, ch2_min = max(test_data["CH2"]), min(test_data["CH2"])
        
        # Set trigger level to 30% of input signal amplitude
        trigger_level = ch1_min + (ch1_max - ch1_min) * 0.3
        scope.write(f"TRIGGER:A:LEVEL {trigger_level:.3f}")
        time.sleep(0.5)
        print(f"Adjusted trigger level to {trigger_level:.3f}V")
        
        # Show signal info
        print(f"Input signal range: {ch1_min:.3f}V to {ch1_max:.3f}V (amplitude: {ch1_max-ch1_min:.3f}V)")
        print(f"Output signal range: {ch2_min:.3f}V to {ch2_max:.3f}V (amplitude: {ch2_max-ch2_min:.3f}V)")
    
    # Confirm with user before starting measurements
    input("Press ENTER to start measurement trials...")
    
    results = []
    valid_trial_count = 0  # Track valid trials that meet the latency bounds
    total_trials = 0  # Total trials attempted 
    
    # Run trials until we have the requested number of valid measurements
    while valid_trial_count < num_trials and total_trials < num_trials * 3:  # Limit max attempts to 3x requested trials
        total_trials += 1
        print(f"\nTrial {total_trials} (Valid trials so far: {valid_trial_count}/{num_trials})")
        
        # Capture waveforms
        data = capture_waveforms()
        
        # Save waveform data
        if data["Time"]:
            trial_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            waveform_file = os.path.join(trials_dir, f"latency_trial{total_trials}_{trial_timestamp}.csv")
            pd.DataFrame(data).to_csv(waveform_file, index=False)
            print(f"Saved waveform data to {waveform_file}")
            
            # Calculate latency with effect-specific settings
            latency_info = calculate_single_pulse_latency(data, effect_config)
            latency = latency_info[0] if latency_info[0] is not None else None
            
            # Generate verification plot
            generate_verification_plot(data, latency_info, total_trials, effect_config, signal_dir, signal_info)
            
            if latency is not None:
                latency_ms = latency * 1000  # Convert to milliseconds
                latency_us = latency * 1000000  # Convert to microseconds
                
                # Check if the latency is within the valid range
                is_valid = min_valid_latency <= latency <= max_valid_latency
                validity_str = "Valid" if is_valid else f"Invalid (outside {min_valid_latency*1e6:.0f}-{max_valid_latency*1e6:.0f}µs range)"
                
                print(f"Latency: {latency_ms:.3f} ms ({latency_us:.1f} µs) - {validity_str}")
                
                # Record all measurements but track which ones are valid
                results.append({
                    'Trial': total_trials, 
                    'Latency (s)': latency, 
                    'Latency (ms)': latency_ms,
                    'Latency (µs)': latency_us,
                    'Signal Shape': signal_shape,
                    'Signal Frequency': signal_frequency,
                    'Timestamp': timestamp,
                    'Validity': validity_str
                })
                
                # Only count it as a valid trial if it's within the bounds
                if is_valid:
                    valid_trial_count += 1
            else:
                print("Could not calculate latency")
                results.append({
                    'Trial': total_trials, 
                    'Latency (s)': None, 
                    'Latency (ms)': None,
                    'Latency (µs)': None,
                    'Signal Shape': signal_shape,
                    'Signal Frequency': signal_frequency,
                    'Timestamp': timestamp,
                    'Validity': 'Failed'
                })
        else:
            print("No data captured")
            results.append({
                'Trial': total_trials, 
                'Latency (s)': None, 
                'Latency (ms)': None,
                'Latency (µs)': None,
                'Signal Shape': signal_shape,
                'Signal Frequency': signal_frequency,
                'Timestamp': timestamp,
                'Validity': 'No Data'
            })
        
        # Wait before next trial
        time.sleep(2)
    
    # Save summary results
    pd.DataFrame(results).to_csv(summary_file, index=False)
    
    # Calculate statistics only using valid measurements
    valid_results = [r['Latency (µs)'] for r in results if r.get('Validity', '').startswith('Valid')]
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
        print(f"Valid measurements: {len(valid_results)}/{total_trials} trials attempted")
        
        # Save statistics to a separate file
        stats_file = os.path.join(signal_dir, f"statistics.txt")
        with open(stats_file, 'w') as f:
            f.write(f"Latency Statistics for {effect_config} with {signal_shape} at {signal_frequency}:\n")
            f.write(f"Measurement timestamp: {timestamp}\n\n")
            f.write(f"Mean: {mean_latency:.1f} µs\n")
            f.write(f"Std Dev: {std_latency:.1f} µs\n")
            f.write(f"Min: {min_latency:.1f} µs\n")
            f.write(f"Max: {max_latency:.1f} µs\n")
            f.write(f"Valid measurements: {len(valid_results)}/{total_trials} trials attempted\n")
            f.write(f"Valid latency range: {min_valid_latency*1e6:.0f} - {max_valid_latency*1e6:.0f} µs\n")
    else:
        print("\nNo valid measurements collected within the specified latency range.")
    
    return signal_dir, effect_config

# Function to generate comparative analysis between effects
def generate_comparative_analysis():
    print("\nGenerating comparative analysis for all configurations...")
    
    # Get list of all measured configurations
    configurations = []
    for item in os.listdir(output_dir):
        if os.path.isdir(os.path.join(output_dir, item)) and not item.startswith("comparisons_"):
            configurations.append(item)
    
    if not configurations:
        print("No configurations found for comparison.")
        return
    
    print(f"Found configurations: {', '.join(configurations)}")
    
    # Ask user to select specific timestamp group to compare 
    # or analyze the most recent measurements across all effects
    print("\nYou can compare:")
    print("1. Most recent measurements across all effects")
    print("2. Specific measurement session (by timestamp)")
    
    compare_choice = input("Enter your choice (1 or 2): ")
    
    if compare_choice == "2":
        # List available timestamps
        all_timestamps = set()
        
        # Collect all timestamps from all configuration folders
        for config in configurations:
            config_dir = os.path.join(output_dir, config)
            for signal_folder in os.listdir(config_dir):
                folder_parts = signal_folder.split('_')
                if len(folder_parts) >= 3:  # Has timestamp format
                    # Extract timestamp (last part of folder name)
                    timestamp = folder_parts[-1]
                    all_timestamps.add(timestamp)
        
        if not all_timestamps:
            print("No timestamped measurement sessions found.")
            return
            
        # Sort timestamps (newest first)
        all_timestamps = sorted(list(all_timestamps), reverse=True)
        
        print("\nAvailable measurement sessions:")
        for i, ts in enumerate(all_timestamps, 1):
            # Convert timestamp to readable format
            try:
                datetime_obj = datetime.strptime(ts, "%Y%m%d_%H%M%S")
                readable_ts = datetime_obj.strftime("%Y-%m-%d %H:%M:%S")
                print(f"{i}. {readable_ts} ({ts})")
            except ValueError:
                print(f"{i}. {ts}")
        
        ts_choice = input("\nEnter the number of the session to analyze: ")
        try:
            selected_timestamp = all_timestamps[int(ts_choice) - 1]
        except (ValueError, IndexError):
            print("Invalid selection. Using most recent session.")
            selected_timestamp = all_timestamps[0]
            
        print(f"Analyzing session with timestamp: {selected_timestamp}")
        
        # Filter folders with the selected timestamp
        signal_folders_by_config = {}
        for config in configurations:
            config_dir = os.path.join(output_dir, config)
            if not os.path.isdir(config_dir):
                continue
                
            for folder in os.listdir(config_dir):
                if selected_timestamp in folder:
                    signal_folders_by_config[config] = folder
    else:
        # Use most recent timestamp for each configuration
        signal_folders_by_config = {}
        for config in configurations:
            config_dir = os.path.join(output_dir, config)
            if not os.path.isdir(config_dir) or not os.listdir(config_dir):
                continue
                
            # Get the most recent folder for this configuration
            folders = [(f, os.path.getmtime(os.path.join(config_dir, f))) 
                      for f in os.listdir(config_dir) if os.path.isdir(os.path.join(config_dir, f))]
            
            if folders:
                # Sort by modification time (newest first)
                folders.sort(key=lambda x: x[1], reverse=True)
                signal_folders_by_config[config] = folders[0][0]
    
    if not signal_folders_by_config:
        print("No suitable measurement folders found for comparison.")
        return
    
    print(f"\nComparing the following configurations:")
    for config, folder in signal_folders_by_config.items():
        print(f"- {config}: {folder}")
    
    # Create a comparison directory
    comparison_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_dir = os.path.join(output_dir, f"comparisons_{comparison_timestamp}")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Collect latency statistics for each configuration
    latencies = []
    labels = []
    
    for config, folder in signal_folders_by_config.items():
        summary_file = os.path.join(output_dir, config, folder, "latency_summary.csv")
        if os.path.exists(summary_file):
            try:
                df = pd.read_csv(summary_file)
                valid_latencies = df['Latency (µs)'].dropna().tolist()
                if valid_latencies:
                    latencies.append(valid_latencies)
                    labels.append(config)
            except Exception as e:
                print(f"Error reading summary for {config}: {e}")
    
    if not latencies:
        print("No valid latency data found for comparison.")
        return
    
    # Generate boxplot comparison
    plt.figure(figsize=(12, 8))
    
    # Latency plot
    plt.boxplot(latencies, labels=labels)
    plt.title(f"Latency Comparison Between Effect Configurations")
    plt.ylabel("Latency (µs)")
    plt.grid(True, alpha=0.3)
    
    # Add mean markers
    means = [np.mean(lat) for lat in latencies]
    plt.scatter(range(1, len(means) + 1), means, marker='o', color='red', s=50, label='Mean')
    
    # Add mean value annotations
    for i, mean in enumerate(means):
        plt.annotate(f'{mean:.1f} µs', 
                     xy=(i+1, mean), 
                     xytext=(0, 10),
                     textcoords='offset points',
                     ha='center')
    
    plt.legend()
    plt.tight_layout()
    
    # Save comparison plot
    comparison_file = os.path.join(comparison_dir, f"latency_comparison.png")
    plt.savefig(comparison_file)
    plt.close()
    
    print(f"Saved comparison plot to {comparison_file}")
    
    # Create summary table with statistics
    stats_table = []
    for i, config in enumerate(labels):
        stats_table.append({
            'Configuration': config,
            'Mean Latency (µs)': np.mean(latencies[i]),
            'Std Dev (µs)': np.std(latencies[i]),
            'Min (µs)': np.min(latencies[i]),
            'Max (µs)': np.max(latencies[i]),
            'Samples': len(latencies[i])
        })
    
    stats_df = pd.DataFrame(stats_table)
    stats_file = os.path.join(comparison_dir, f"latency_statistics.csv")
    stats_df.to_csv(stats_file, index=False)
    
    print(f"Saved latency statistics to {stats_file}")
    
    # Generate latency ranking plot
    plt.figure(figsize=(12, 6))
    
    # Sort configurations by latency (lowest to highest)
    latency_ranking = sorted(
        [(labels[i], np.mean(latencies[i])) for i in range(len(labels))],
        key=lambda x: x[1]
    )
    
    # Create bar chart
    config_names = [x[0] for x in latency_ranking]
    latency_means = [x[1] for x in latency_ranking]
    
    bars = plt.bar(config_names, latency_means)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.5,
            f'{height:.1f} µs',
            ha='center', va='bottom',
            rotation=0
        )
    
    plt.title("Effect Configurations Ranked by Latency (Lower is Better)")
    plt.ylabel("Mean Latency (µs)")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save ranking plot
    ranking_file = os.path.join(comparison_dir, f"latency_ranking.png")
    plt.savefig(ranking_file)
    plt.close()
    
    print(f"Saved latency ranking plot to {ranking_file}")

# Main program flow
print("\nReady to start latency measurements")

try:
    while True:
        print("\n=== FPGA Audio Effects Processor Latency Test Suite ===")
        print("1. Run measurements for a single configuration")
        print("2. Run batch measurements for all configurations")
        print("3. Generate comparative analysis")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            # Get configuration from user
            effect_config = get_effect_configuration()
            
            # Get measurement parameters
            signal_shape = input("Enter input signal shape (sine, square, pulse, etc.): ")
            signal_frequency = input("Enter input signal frequency (e.g., 20Hz, 1kHz): ")
            num_trials = int(input("Enter the number of valid trials to collect (default is 10): ") or 10)
            
            # Ask about latency range
            print("\nSpecify the valid latency range for measurements:")
            min_latency = float(input("Enter minimum valid latency in microseconds (default 450): ") or 450)
            max_latency = float(input("Enter maximum valid latency in microseconds (default 900): ") or 900)
            
            # Convert to seconds
            min_valid_latency = min_latency / 1e6
            max_valid_latency = max_latency / 1e6
            
            print(f"Valid latency range: {min_valid_latency*1e6:.0f}µs - {max_valid_latency*1e6:.0f}µs")
            
            # Setup scope before running
            setup_oscilloscope(effect_config)
            
            # Run the measurement
            run_measurement(effect_config, signal_shape, signal_frequency, num_trials, min_valid_latency, max_valid_latency)
            
        elif choice == '2':
            run_batch_measurements()
            
        elif choice == '3':
            generate_comparative_analysis()
            
        elif choice == '4':
            print("\nExiting latency measurement utility.")
            break
            
        else:
            print("Invalid choice. Please try again.")
finally:
    # Always close the scope connection
    try:
        scope.close()
    except:
        pass
    
    print("\nLatency measurements complete!")