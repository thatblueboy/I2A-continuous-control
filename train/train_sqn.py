import os
import yaml
import subprocess
from itertools import product

CONFIG_DIR = "/home/thatblueboy/DOP/new_config"  # Update with the actual path to your config files
SEEDS = [42] #[456, 3000, 7]#[100]#[42]  # List of seeds

EXPERIMENTS = [
    {"wrapper": None, "history": 0, "n_future_steps": 0},
    {"wrapper": None, "history": 2, "n_future_steps": 0},
    {"wrapper": None, "history": 5, "n_future_steps": 0},
    {"wrapper": "DreamWrapper", "history": 0, "n_future_steps": 2},
    {"wrapper": "DreamWrapper", "history": 0, "n_future_steps": 5},
]

OUTPUT_DIR = ""  # Update as needed

# Iterate over all config files
for config_file in os.listdir(CONFIG_DIR):
    if not config_file.endswith(".yaml"):
        continue
    
    config_path = os.path.join(CONFIG_DIR, config_file)
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    env_name = config["environment"]["name"]
    
    for exp, seed in product(EXPERIMENTS, SEEDS):
        config["environment"]["wrapper"] = exp["wrapper"]
        config["environment"]["history"] = exp["history"]
        config["environment"]["n_future_steps"] = exp["n_future_steps"]
        config["training"]["seed"] = seed
        
        if exp["wrapper"] == "DreamWrapper":
            exp_name = f"dreamer_{exp['n_future_steps']}_{seed}"
        else:
            exp_name = f"no_dreamer_{exp['history']}_{seed}"
        
        config["expt_name"] = exp_name
        
        temp_config_path = os.path.join(OUTPUT_DIR, f"temp_{exp_name}.yaml")
        with open(temp_config_path, "w") as temp_file:
            yaml.dump(config, temp_file)
        
        print(f"Running experiment: {exp_name}")
        subprocess.run(["python", "train.py", "--config", temp_config_path])
        os.remove(temp_config_path)

print("All experiments completed!")
