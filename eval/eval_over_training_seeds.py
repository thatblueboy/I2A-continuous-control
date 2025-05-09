import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from env.wrapper import DreamWrapper


def evaluate_agent(model, env, std=0.0, seeds=None):
    
    if seeds is None:
        seeds = [np.random.randint(0, 10000) for _ in range(5)]  # Generate 5 random seeds
    
    results = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        episode_reward = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action, mean=0, std=std)
            episode_reward += reward
            if done or truncated:
                break
        results.append(episode_reward)
    return results


def evaluate_at_zero_noise(log_dir, env_fn, model_prefixes, seeds):
    models_dir = os.path.join(log_dir, "models")
    dreamers_dir = os.path.join(log_dir, "dreamers")

    results = {}

    all_model_names = os.listdir(models_dir)

    for prefix in model_prefixes:
        matching_models = [name for name in all_model_names if name.startswith(prefix + "_")]
        print(f"Evaluating {len(matching_models)} models for prefix '{prefix}'")

        all_rewards = []

        for model_name in matching_models:
            model_path = os.path.join(models_dir, model_name, "best_model.zip")
            if not os.path.exists(model_path):
                continue

            model = PPO.load(model_path, device='cpu')

            # Parse dreamer settings
            parts = model_name.split("_")
            if model_name.startswith("no_dreamer_"):
                history_len = int(parts[2])
                n_future_steps = 0
            elif model_name.startswith("dreamer_"):
                n_future_steps = int(parts[1])
                history_len = 0
            else:
                continue

            env = env_fn()
            wrapped_env = DreamWrapper(env, n_future_steps=n_future_steps, history_len=history_len, eval=True,
                                       policy_hidden_layers=[256, 256],
                                       dynamics_hidden_layers=[512, 256, 128])

            if model_name.startswith("dreamer_"):
                dreamer_weights_path = os.path.join(dreamers_dir, model_name, "best_dreamer_state_dict.pth")
                if os.path.exists(dreamer_weights_path):
                    weights = torch.load(dreamer_weights_path, map_location='cpu')
                    weights = {k: v.float() for k, v in weights.items()}
                    wrapped_env.dreamer.load_state_dict(weights)

            reward = evaluate_agent(model, wrapped_env, std=0.0, seeds=seeds)
            all_rewards = all_rewards + reward

        if all_rewards:
            print("total reward samples", len(all_rewards))
            print(prefix)
            mean_reward = np.mean(all_rewards)
            std_reward = np.std(all_rewards)
            results[prefix] = {"mean": mean_reward, "std": std_reward}
            print(f"{prefix} → Mean: {mean_reward:.2f}, Std: {std_reward:.2f}")

    return results

# Example usage

log_dir = "/media/thatblueboy/Seagate/DOP/logs/Walker2d-v5_PPO/"
model_prefixes = ["no_dreamer_5", "no_dreamer_2", "no_dreamer_0", "dreamer_2", "dreamer_5"]

def env_fn():
    return gym.make("Walker2d-v5")

seeds = [42, 123, 987, 654, 321, 888, 777, 2024, 111, 555, 999, 2468, 1357, 4321, 1010]

data = evaluate_at_zero_noise(log_dir, env_fn, model_prefixes, seeds=seeds)
