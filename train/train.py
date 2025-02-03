import yaml
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import sys
import os
import argparse

sys.path.append("/home/thatblueboy/DOP")

from env.wrapper import DreamWrapper
from datetime import datetime
from stable_baselines3.common.logger import configure

import numpy as np
import torch

# YAML configuration as a string
DEFAULT_CONFIG_YAML = """
environment:
  name: "Ant-v5"
  wrapper: "DreamWrapper"
#   wrapper: None

  n_future_steps: 5
  n_steps: 1024

ppo_hyperparameters:
  policy: "MlpPolicy"
  n_steps: 512
  batch_size: 32
  gamma: 0.98
  learning_rate: 1.90609e-05
  ent_coef: 4.9646e-07
  clip_range: 0.1
  n_epochs: 10
  gae_lambda: 0.8
  max_grad_norm: 0.6
  vf_coef: 0.677239
  verbose: 1
  device: "cpu"

training:
  total_timesteps: 10000000
  save_wrapper_state_path: "dreamer_state_dict.pth"
"""

# Function to load the YAML configuration
def load_config(config_path=None):
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    else:
        print("No valid config file path provided, using default configuration.")
        config = yaml.safe_load(DEFAULT_CONFIG_YAML)

    # Convert activation function string to actual PyTorch function
    if "policy_kwargs" in config["ppo_hyperparameters"]:
        if "activation_fn" in config["ppo_hyperparameters"]["policy_kwargs"]:
            activation_fn_str = config["ppo_hyperparameters"]["policy_kwargs"]["activation_fn"]
            config["ppo_hyperparameters"]["policy_kwargs"]["activation_fn"] = getattr(torch.nn, activation_fn_str)  # Convert string to function
    return config

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Training PPO on Ant-v5 with DreamWrapper")
    parser.add_argument('--config', type=str, help="Path to the configuration YAML file")
    parser.add_argument('--name', type=str)
    return parser.parse_args()

# Load the YAML config
# CONFIG = yaml.safe_load(CONFIG_YAML)

# Main script
if __name__ == "__main__":
    # Parse arguments
    args = parse_args()

    # Load the configuration (either from file or default)
    CONFIG = load_config(args.config)

    seed = 42

    env_name = CONFIG["environment"]["name"]
    algo_name = "PPO"
    LOGS_ROOT_PATH = os.path.join("/home/thatblueboy/DOP/logs", env_name + "_" +algo_name+"_"+str(seed))
    #     EXPT_NAME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    EXPT_NAME = args.name
    # EXPT_NAME = args["name"]
    MODELS_PATH = os.path.join(LOGS_ROOT_PATH, "models", EXPT_NAME)
    DREAMER_PATH = os.path.join(LOGS_ROOT_PATH, "dreamers", EXPT_NAME)
    # if os.path.exists(MODELS_PATH):
    #     sys.exit(f"Error: Experiment already exists!!")
    # Load config values
    wrapper = CONFIG["environment"]["wrapper"]
    n_future_steps = CONFIG["environment"]["n_future_steps"]
    n_steps = CONFIG["ppo_hyperparameters"]["n_steps"]

    # Environment setup
    env = gym.make(env_name)
    # env.seed(seed)
    print("wrapper is", wrapper)
    if wrapper == "DreamWrapper":
        print("Dreaming!!")
        wrapped_env = DreamWrapper(env, n_future_steps=n_future_steps, n_steps=n_steps,n_steps_dreamer=1024, dreamer_save_path=DREAMER_PATH)
    else:
        print("standard environment")
        wrapped_env = env

    # PPO hyperparameters
    hyperparams = CONFIG["ppo_hyperparameters"]

    # Initialize PPO
    ppo_model = PPO(**hyperparams, env=wrapped_env, seed=seed)
    ppo_model.set_logger(configure(MODELS_PATH, ["tensorboard","stdout"]))

    # Train the model
    print("Training started...")
    total_timesteps = CONFIG["training"]["total_timesteps"]
    ppo_model.learn(total_timesteps=total_timesteps )
    print("Training finished!")

    # Save the model and wrapper state
    # TODO dreamer/wrapper should save dreamer
    if wrapper == "DreamWrapper":
        torch.save(wrapped_env.dreamer.state_dict(), os.path.join(DREAMER_PATH, "dreamer_state_dict.pth"))
    ppo_model.save(os.path.join(MODELS_PATH, "model"))
    print("Model and wrapper state saved!")
