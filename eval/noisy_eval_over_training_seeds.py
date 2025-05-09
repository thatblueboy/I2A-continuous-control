import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from env.wrapper import DreamWrapper  # Ensure this is correctly imported
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from stable_baselines3 import PPO

def evaluate_agent(model, env, std=0.0, seeds=None):
    
    if seeds is None:
        seeds = [np.random.randint(0, 10000) for _ in range(5)]  # Generate 5 random seeds
    
    results = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        episode_reward = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action, mean=0.0, std=std)
            episode_reward += reward
            if done or truncated:
                break
        results.append(episode_reward)
    return np.mean(results)


def evaluate_models_across_noise(log_dir, env_fn, model_prefixes, std_range=(0, 0.3, 0.1), seeds=[23, 3542, 452, 299, 7, 111111]):
    models_dir = os.path.join(log_dir, "models")
    dreamers_dir = os.path.join(log_dir, "dreamers")
    results_dict = {}
    std_values = np.arange(*std_range)
    print(std_values)

    # Get all model directories
    all_model_names = os.listdir(models_dir)

    for prefix in model_prefixes:
        matching_models = [name for name in all_model_names if name.startswith(prefix + "_")]
        print(f"Found {len(matching_models)} models for prefix '{prefix}'")

        reward_matrix = []  # Will collect rewards from multiple seeds

        for model_name in matching_models:
            model_path = os.path.join(models_dir, model_name, "best_model.zip")
            if not os.path.exists(model_path):
                continue  # Skip if model doesn't exist

            # Load model
            model = PPO.load(model_path, device='cpu')

            # Parse n_future_steps and history_len
            parts = model_name.split("_")
            if model_name.startswith("no_dreamer_"):
                history_len = int(parts[2])
                n_future_steps = 0
            elif model_name.startswith("dreamer_"):
                n_future_steps = int(parts[1])
                history_len = 0
            else:
                continue

            # Create and wrap environment
            env = env_fn()
            wrapped_env = DreamWrapper(env, n_future_steps=n_future_steps, history_len=history_len, eval=True,
                                       policy_hidden_layers=[256, 256],
                                       dynamics_hidden_layers=[512, 256, 128])

            # Load dreamer weights if necessary
            if model_name.startswith("dreamer_"):
                dreamer_weights_path = os.path.join(dreamers_dir, model_name, "best_dreamer_state_dict.pth")
                if os.path.exists(dreamer_weights_path):
                    weights = torch.load(dreamer_weights_path, map_location='cpu')
                    weights = {k: v.float() for k, v in weights.items()}
                    wrapped_env.dreamer.load_state_dict(weights)

            model_rewards = []
            for std in std_values:
                print(std)
                avg_reward = evaluate_agent(model, wrapped_env, std=std, seeds=seeds)
                print("Model:", model_name, "| Std:", std, "| Return:", avg_reward)
                model_rewards.append(avg_reward)

            if model_rewards:
                reward_matrix.append(model_rewards)

        if reward_matrix:
            reward_matrix = np.array(reward_matrix)
            avg_rewards = np.mean(reward_matrix, axis=0).tolist()
            results_dict[prefix] = avg_rewards

    # Save results
    results_path = os.path.join(log_dir, "new_new_eval_results.json")
    with open(results_path, "w") as f:
        json.dump({"std_values": std_values.tolist(), "results": results_dict}, f, indent=4)

    # Plot
    plt.figure(figsize=(10, 6))
    for prefix, rewards in results_dict.items():
        plt.plot(std_values, rewards, label=prefix, linewidth=2)

    plt.xlabel("Standard Deviation of Noise", fontsize=14)
    plt.ylabel("Average Return", fontsize=14)
    plt.title("Model Performance under Observation Noise", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return results_dict

# Example usage

log_dir = "/media/thatblueboy/Seagate/DOP/logs/Ant-v5_PPO/"
model_prefixes = ["no_dreamer_5", "no_dreamer_2", "no_dreamer_0", "dreamer_2", "dreamer_5"]

def env_fn():
    return gym.make("Ant-v5")

seeds = [42, 123, 987, 654, 321, 888, 777, 2024, 111, 555, 999, 2468, 1357, 4321, 1010]

data = evaluate_models_across_noise(log_dir, env_fn, model_prefixes, std_range=(0.0, 0.31, 0.05), seeds=seeds)
