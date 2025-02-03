import os
import glob
import argparse
import threading
import concurrent.futures
import subprocess

def run_experiment(config_path, expt_name=None):
    """
    Run a single experiment by calling the training function.
    This example uses subprocess to invoke the training script.
    Adjust the command line call if your training logic is imported instead.
    """
    # If you want to pass the name argument, you could add e.g. '--name', expt_name
    cmd = ["python", "train.py", "--config", config_path]
    if expt_name is not None:
        cmd.extend(["--name", expt_name])
    print(f"Launching experiment for {config_path} with command: {' '.join(cmd)}")
    subprocess.run(cmd)  # Will wait until the experiment finishes

def main():
    parser = argparse.ArgumentParser(description="Run experiments concurrently for all config files in a folder")
    parser.add_argument('--config_folder', type=str, required=True,
                        help="Path to the folder containing YAML config files")
    # Optionally, add a flag for choosing thread vs. process based concurrency.
    parser.add_argument('--use_threads', action='store_true', help="Use threading (default is processes)")
    args = parser.parse_args()

    config_folder = args.config_folder
    # Get all YAML files in the folder (modify the pattern if needed)
    config_files = glob.glob(os.path.join(config_folder, "*.yaml"))
    if not config_files:
        print(f"No YAML config files found in {config_folder}")
        return

    # Use ThreadPoolExecutor or ProcessPoolExecutor based on your use-case:
    if args.use_threads:
        Executor = concurrent.futures.ThreadPoolExecutor
    else:
        Executor = concurrent.futures.ProcessPoolExecutor

    # Launch experiments concurrently.
    with Executor() as executor:
        futures = []
        for config_path in config_files:
            # You can customize the experiment name if needed, e.g., using the config file's basename.
            expt_name = os.path.splitext(os.path.basename(config_path))[0]
            futures.append(executor.submit(run_experiment, config_path, expt_name))
        # Optionally, wait for all experiments to finish.
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An experiment raised an exception: {e}")

if __name__ == "__main__":
    main()
