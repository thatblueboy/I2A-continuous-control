import numpy as np
import torch
import matplotlib.pyplot as plt
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env.wrapper import DreamWrapper
from stable_baselines3.common.utils import set_random_seed

# Function to evaluate the agent systematically
def evaluate_agent(model, env, num_episodes=10, seeds=None):
    """
    Evaluate the agent across multiple episodes and different seeds.

    Args:
        model: The trained agent.
        env: The wrapped environment.
        num_episodes: Number of episodes to evaluate per seed.
        seeds: List of random seeds. If None, seeds are randomly generated.

    Returns:
        results: A dictionary containing rewards and episode lengths for each seed.
    """
    if seeds is None:
        seeds = [np.random.randint(0, 10000) for _ in range(5)]  # Generate 5 random seeds
    
    results = {"seed": [], "episode_rewards": [], "episode_lengths": []}

    for seed in seeds:
        # set_random_seed(seed)
        # env.action_space.seed(seed)

        seed_rewards = []
        seed_lengths = []

        for episode in range(num_episodes):
            obs, _ = env.reset(seed=seed)
            episode_reward = 0
            episode_length = 0
            print("episode #", episode+1)
            i =0
            while True:
                # print(obs)
                # obs = np.concatenate((obs[:105], np.zeros(210)))
                # obs = np.random.rand(315)
                # action, _ = model.predict(obs, deterministic=True)  # Use deterministic actions
                obs = torch.from_numpy(obs).float()
                # print(obs.dtype)
                
                action = wrapped_env.dreamer.dreamer_p(obs)
                print(action)
                action = np.array(action)
                obs, reward, done, truncated, info = env.step(action)

                # print("step #", i+1)
                i +=1
                episode_reward += reward
                episode_length += 1

                if done or truncated:
                    seed_rewards.append(episode_reward)
                    seed_lengths.append(episode_length)
                    break

        results["seed"].append(seed)
        results["episode_rewards"].append(seed_rewards)
        results["episode_lengths"].append(seed_lengths)

    return results

# Plot the results
def plot_results(results):
    seeds = results["seed"]
    rewards = results["episode_rewards"]

    # Calculate means and standard deviations
    mean_rewards = [np.mean(r) for r in rewards]
    print(mean_rewards)
    std_rewards = [np.std(r) for r in rewards]

    # Plot mean rewards with error bars
    plt.figure(figsize=(8, 6))
    plt.errorbar(seeds, mean_rewards, yerr=std_rewards, fmt="o-", capsize=5, label="Rewards")
    plt.xlabel("Seed")
    plt.ylabel("Episode Reward")
    plt.title("Evaluation Results Across Seeds")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    with torch.no_grad():
        model = PPO.load("/home/thatblueboy/DOP/logs/Ant-v5_PPO_42/models/dreamer_5_steps/model.zip", device='cpu')
        # model = PPO.load("/home/thatblueboy/DOP/logs/Ant-v5_PPO_42/models/no_dreamer/model.zip", device='cpu')

        # env = gym.make("Ant-v5", render_mode='human')
        env = gym.make("Ant-v5")

        wrapped_env = DreamWrapper(env, n_future_steps = 5, n_steps=1024, eval=True)  # Use the trained model to augment future observations
        weights = torch.load("/home/thatblueboy/DOP/logs/Ant-v5_PPO_42/dreamers/dreamer_5_steps/dreamer_state_dict.pth")  # Load weights

        # Convert weights to float32 (single precision)
        for key in weights.keys():
            weights[key] = weights[key].float()
            # print(key)

        random_weights = {}
        for key, tensor in weights.items():
            random_weights[key] = torch.zeros_like(tensor)  # Random weights with the same shape

        wrapped_env.dreamer.load_state_dict(weights)
        results = evaluate_agent(model, env, num_episodes=1, seeds=[23, 3542, 452, 299, 111111])
        # [42, 123, 456, 789, 101112]
        # Save results to a file
        torch.save(results, "evaluation_results.pt")

        plot_results(results)

