import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
# Load the trained model
model = PPO.load("/home/thatblueboy/DOP/logs/Ant-v5_PPO_42/models/no_dreamer/model.zip")

# Create a new environment for testing
env = gym.make("Ant-v5")

# Reset the environment
obs, _ = env.reset()

# Run the trained agent in the environment
for _ in range(1000):  # Run for 1000 steps
    action, _ = model.predict(obs, deterministic=True)  # Use deterministic actions
    obs, reward, done, _, info = env.step(action)
    
    if done:
        print(info)
        obs, _ = env.reset()                            # Reset if the episode ends
