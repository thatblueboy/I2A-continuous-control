import yaml
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback

import sys
import os
import argparse

import random

sys.path.append("/home/thatblueboy/DOP")

from env.wrapper import DreamWrapper
from datetime import datetime
from stable_baselines3.common.logger import configure

import numpy as np
import torch

import logging

# Set up the basic configuration for logging
logging.basicConfig(
    filename='simple_log.log',  # Name of the log file
    level=logging.DEBUG,        # Log level (DEBUG will capture all levels)
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
)

# YAML configuration as a string
DEFAULT_CONFIG_YAML = """
expt_name: test

environment:
  name: "Humanoid-v5"              # Name of the Gymnasium environment
  wrapper: DreamWrapper       # Wrapper for the environment (e.g., DreamWrapper or None)
  history: 0
  n_future_steps: 0         # Number of future steps for the wrapper (if using DreamWrapper)
  n_steps: 1024                # Number of steps in each environment rollout (for training)
  p_hidden: [256, 128, 64, 32] #384, 17
  d_hidden: [2048, 1024, 512] #17+384,  ?, 384

ppo_hyperparameters:
  policy: 'MlpPolicy'
  batch_size: 50
  n_steps: 50
  device: 'cpu'
  gamma: 0.95
  learning_rate: 3.56987e-05
  ent_coef: 0.00238306
  clip_range: 0.3
  n_epochs: 5
  gae_lambda: 0.9
  max_grad_norm: 2
  vf_coef: 0.431892
  verbose: 1
  policy_kwargs:
    log_std_init: -2
    ortho_init: False
    activation_fn: "ReLU"
    net_arch:
      pi: [256, 256]
      vf: [256, 256]

training:
  eval_freq: 1
  seed: 32  
  total_timesteps: 10000    # Total number of timesteps to train the model
  save_wrapper_state_path: "dreamer_state_dict.pth" # Path to save the dreamer model state
"""

class CustomEvalCallback(EvalCallback):

    def __init__(self, *args, dreaming=None, dreamer_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dreaming = dreaming  # Store external parameter
        self.dreamer_path = dreamer_path
         
    def _on_step(self):
        # logging.debug('Step callback - Number of calls: %d', self.n_calls)
        eval = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0

        if eval:
            print(self.eval_env.envs[0])
            if hasattr(self.eval_env.envs[0], "eval"):
                print("changing eval")
                self.eval_env.envs[0].eval = True  # Set eval mode before evaluation
                # logging.debug("Setting eval True!")
            else:
                print("PARAM NOT FOUND")

        result = super()._on_step()  # Run evaluation

        if eval:
            if hasattr(self.eval_env.envs[0], "eval"):
                self.eval_env.envs[0].eval = False  # Reset after evaluation
                # logging.debug("Setting eval False!")

        # logging.debug(f"Best mean reward == last mean reward: {self.best_mean_reward == self.last_mean_reward}")
        # if self.evaluations_results:
            # logging.debug(f"Best mean reward == max(eval): {self.best_mean_reward == float(np.max(self.evaluations_results))}")

        #python check one by one
        if eval and self.dreaming and self.evaluations_results and  self.best_mean_reward == self.last_mean_reward:
                # logging.debug("✅ New best model found! Saving dreamer!")

                self.eval_env.envs[0]
                torch.save(self.eval_env.envs[0].dreamer.state_dict(), os.path.join(self.dreamer_path, "best_dreamer_state_dict.pth"))
                
        return result


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

def train(CONFIG):
    SEED = CONFIG["training"]["seed"]
    
    random.seed(SEED)

    # Set seed for NumPy
    np.random.seed(SEED)

    # Set seed for PyTorch (CPU & CUDA)
    torch.manual_seed(SEED)

    env_name = CONFIG["environment"]["name"]

    algo_name = "PPO"
    LOGS_ROOT_PATH = os.path.join("/home/thatblueboy/DOP/logs", env_name + "_" +algo_name)

    EXPT_NAME = CONFIG["expt_name"]
    
    MODELS_PATH = os.path.join(LOGS_ROOT_PATH, "models", EXPT_NAME)
    DREAMER_PATH = os.path.join(LOGS_ROOT_PATH, "dreamers", EXPT_NAME)
    if EXPT_NAME != "test":
        if os.path.exists(MODELS_PATH):
            sys.exit(f"Error: Experiment already exists!!")
    # Load config values
    wrapper = CONFIG["environment"]["wrapper"]
    n_future_steps = CONFIG["environment"]["n_future_steps"]
    n_steps = CONFIG["ppo_hyperparameters"]["n_steps"]

    # Environment setup
    env = gym.make(env_name)

    print("wrapper is", wrapper)
    # if wrapper == "DreamWrapper":
    #     print("Dreaming!!")
    #     wrapped_env = DreamWrapper(env, 
    #                                history_len=CONFIG["environment"]["history"],
    #                                n_future_steps=n_future_steps,
    #                                 n_steps=n_steps, 
    #                                 n_steps_dreamer=n_steps,
    #                                 policy_hidden_layers=CONFIG["environment"]["p_hidden"],
    #                                 dynamics_hidden_layers=CONFIG["environment"]["d_hidden"],
    #                                 dreamer_save_path=DREAMER_PATH)
    # else:
    #     print("standard environment")
    #     wrapped_env = env

    wrapped_env = DreamWrapper(env, 
                                history_len=CONFIG["environment"]["history"],
                                n_future_steps=n_future_steps,
                                n_steps=n_steps, 
                                n_steps_dreamer=n_steps,
                                dreamer_batch_size=CONFIG["environment"]["batch_size"],
                                policy_hidden_layers=CONFIG["environment"]["p_hidden"],
                                dynamics_hidden_layers=CONFIG["environment"]["d_hidden"],
                                dreamer_save_path=DREAMER_PATH)

    # Initialize PPO
    eval_callback = CustomEvalCallback(
    dreaming=n_future_steps != 0,
    dreamer_path= DREAMER_PATH,
    best_model_save_path=MODELS_PATH,
    log_path=MODELS_PATH,
    eval_freq=CONFIG["training"]["eval_freq"],  # Adjust as needed
    eval_env=wrapped_env,  # Same env for eval
    n_eval_episodes=1,  # Number of eval episodes
    deterministic=True,
    render=False
    )

    # Train with callback
    ppo_model = PPO(**CONFIG["ppo_hyperparameters"],
                    env=wrapped_env, 
                    seed=SEED)
    ppo_model.set_logger(configure(MODELS_PATH, ["tensorboard","stdout"]))

    print("PPO model", ppo_model.policy)

    # Train the model
    print("Training started...")
    total_timesteps = CONFIG["training"]["total_timesteps"]
    ppo_model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    print("Training finished!")

    # Save the model and wrapper state
    # TODO dreamer/wrapper should save dreamer
    if n_future_steps > 0:
        torch.save(wrapped_env.dreamer.state_dict(), os.path.join(DREAMER_PATH, "dreamer_state_dict.pth"))
    ppo_model.save(os.path.join(MODELS_PATH, "model"))
    print("Model and wrapper state saved!")

# Main script
if __name__ == "__main__":
    # Parse arguments
    args = parse_args()

    # Load the configuration (either from file or default)
    CONFIG = load_config(args.config)

    train(CONFIG)  