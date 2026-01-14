import re
import matplotlib.pyplot as plt
import os

# Define file paths
log_file = "/root/autodl-tmp/ODE-Adversarial-Prompt-Tuning/output/oxford_pets/AdvPT/vit_b16/adv/resnet_no_mean/log.txt"
output_file = "/root/autodl-tmp/ODE-Adversarial-Prompt-Tuning/loss_curve_advpt_oxford_pets_vit_b16.png"

# Check if log file exists
if not os.path.exists(log_file):
    print(f"Error: Log file not found at {log_file}")
    exit(1)

losses = []
accuracies = []
# Regex to match: loss <current> (<average>) acc <current> (<average>)
# Example: loss 3.7637 (3.9742) acc 3.9062 (4.2188)
loss_pattern = re.compile(r"loss\s+[\d\.]+\s+\(([\d\.]+)\)")
acc_pattern = re.compile(r"acc\s+[\d\.]+\s+\(([\d\.]+)\)")

print(f"Reading log file: {log_file}")
with open(log_file, 'r') as f:
    for line in f:
        if "loss" in line and "acc" in line:
            loss_match = loss_pattern.search(line)
            acc_match = acc_pattern.search(line)
            if loss_match and acc_match:
                try:
                    loss_val = float(loss_match.group(1))
                    acc_val = float(acc_match.group(1))
                    losses.append(loss_val)
                    accuracies.append(acc_val)
                except ValueError:
                    continue

if not losses:
    print("No loss values found in the log file.")
    exit(1)

if not accuracies:
    print("No accuracy values found in the log file.")
    exit(1)

print(f"Found {len(losses)} data points.")

# Plotting
fig, ax1 = plt.figure(figsize=(12, 6)), plt.gca()

# Plot loss on primary y-axis
color = 'tab:blue'
ax1.set_xlabel('Iterations (reported steps)')
ax1.set_ylabel('Loss', color=color)
ax1.plot(losses, label='Average Loss', color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.7)

# Plot accuracy on secondary y-axis
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Accuracy (%)', color=color)
ax2.plot(accuracies, label='Average Accuracy', color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0, 100)  # Set y-axis range from 0 to 100

# Add title and legend
plt.title('Training Loss and Accuracy Curve')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

plt.tight_layout()
plt.savefig(output_file)
print(f"Loss and accuracy curve saved to {output_file}")
