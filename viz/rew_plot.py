import os
import re
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Define base directory
BASE_DIR = "/home/thatblueboy/DOP/logs/Walker2d-v5_PPO_no_best_model/models"


# Define categories based on folder name prefixes
PREFIXES = ["dreamer_2","dreamer_5", "no_dreamer_0", "no_dreamer_2", "no_dreamer_5"]

# Data storage
grouped_rewards = defaultdict(lambda: defaultdict(list))  # {prefix: {step: [values]}}

# Iterate through each subfolder
for subfolder in os.listdir(BASE_DIR):
    subfolder_path = os.path.join(BASE_DIR, subfolder)

    # Match folders strictly with pattern "<prefix>_<number>"
    matching_prefix = next((p for p in PREFIXES if re.match(rf"^{p}_[0-9]+$", subfolder)), None)
    if not matching_prefix or not os.path.isdir(subfolder_path):
        continue
    print(matching_prefix)

    # Find TensorFlow event files
    for root, _, files in os.walk(subfolder_path):
        print(subfolder_path)
        for file in files:
            if file.startswith("events.out.tfevents"):
                event_file = os.path.join(root, file)

                # Extract `ep_rew_mean` values
                for event in tf.compat.v1.train.summary_iterator(event_file):
                    for value in event.summary.value:
                        if value.tag == "rollout/ep_rew_mean":
                            step = event.step  # Training step
                            grouped_rewards[matching_prefix][step].append(value.simple_value)

# Compute averaged rewards per step
averaged_rewards = {
    prefix: {step: np.mean(values) for step, values in sorted(steps.items())}
    for prefix, steps in grouped_rewards.items()
}

# Plot results
plt.figure(figsize=(10, 6))

for prefix, step_data in averaged_rewards.items():
    steps = list(step_data.keys())
    mean_rewards = list(step_data.values())
    plt.plot(steps, mean_rewards, label=prefix, linewidth=2)

plt.xlabel("Training Steps")
plt.ylabel("Average ep_rew_mean")
plt.title("Averaged Episode Reward Mean Across Runs")
plt.legend()
plt.grid(True)
plt.show()

