import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import EngFormatter

# Function to load data from CSV files
def load_csv_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Paths to your CSV files
csv_files = {
    "Bypass": "../Documents/latency_measurements/bypass/pulse_20Hz/latency_summary.csv",
    "LowPass": "../Documents/latency_measurements/lowpass/pulse_20Hz/latency_summary.csv",
    "Distortion": "../Documents/latency_measurements/distortion/pulse_20Hz/latency_summary.csv",
    "Both": "../Documents/latency_measurements/both/pulse_20Hz/latency_summary.csv"
}

# Define distinct colors and markers for each effect
effect_colors = {
    "Bypass": "blue",
    "LowPass": "green",
    "Distortion": "red",
    "Both": "purple"
}

effect_markers = {
    "Bypass": "o",
    "LowPass": "s",
    "Distortion": "^",
    "Both": "D"
}

# Create figure
plt.figure(figsize=(12, 8))

# Prepare data
stats_data = {}

# Load and process each file
for effect_name, file_path in csv_files.items():
    if os.path.exists(file_path):
        # Load data
        data = load_csv_data(file_path)
        
        # Extract trial numbers and latency values
        trial_numbers = data['Trial'].values
        latency_values = data['Latency (µs)'].values
        
        # Calculate statistics
        mean = np.mean(latency_values)
        std = np.std(latency_values)
        min_val = np.min(latency_values)
        max_val = np.max(latency_values)
        stats_data[effect_name] = {"mean": mean, "std": std, "min": min_val, "max": max_val}
        
        # print(f"{effect_name}: Mean={mean:.2f}µs, Std={std:.2f}µs, Min={min_val:.2f}µs, Max={max_val:.2f}µs")
        
        # Plot this effect's points with trial number as x-axis
        plt.plot(trial_numbers, latency_values, 
                 color=effect_colors[effect_name],
                 marker=effect_markers[effect_name],
                 linestyle='-', 
                 alpha=0.7,
                 label=f"{effect_name} (μ={mean:.1f}µs)")
                 
        # Add horizontal line for mean
        plt.axhline(y=mean, 
                    color=effect_colors[effect_name], 
                    linestyle='--', 
                    alpha=0.5,
                    linewidth=1)
    else:
        print(f"Warning: File not found - {file_path}")

# Set axis labels and title
plt.xlabel('Trial Number', fontsize=12)
plt.ylabel('Latency (µs)', fontsize=12)
plt.title('FPGA Audio Processor Latency Comparison Across Trials\n20Hz Pulse Input Signal', fontsize=14)

# Set x-axis ticks
plt.xticks(range(1, 21))  # Assuming 20 trials
plt.grid(True, linestyle='--', alpha=0.3)

# Add legend
plt.legend(loc='best', fontsize=10)

# Adjust y-axis limits to focus on the data while keeping all series visible
all_means = [stats["mean"] for stats in stats_data.values()]
all_stds = [stats["std"] for stats in stats_data.values()]
min_mean = min(all_means)
max_mean = max(all_means)
range_mean = max_mean - min_mean
max_std = max(all_stds)

# Set y limits with padding
plt.ylim(min_mean - range_mean * 0.2 - max_std * 3, 
         max_mean + range_mean * 0.1 + max_std * 3)

# # Add a table with detailed statistics
# stats_text = "\n".join([f"{name}: Mean={stats['mean']:.1f}µs, Std={stats['std']:.1f}µs, Min={stats['min']:.1f}µs, Max={stats['max']:.1f}µs" 
#                         for name, stats in stats_data.items()])
# plt.figtext(0.5, 0.01, stats_text, ha="center", fontsize=9, 
#             bbox={"facecolor":"white", "alpha":0.7, "pad":5})

# Save the figure
plt.tight_layout(rect=[0, 0.08, 1, 0.97])  # Adjust for the table at bottom
plt.savefig('fpga_latency_vs_trial.png', dpi=300)
plt.show()

print("Plot saved as 'fpga_latency_vs_trial.png'")