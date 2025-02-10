import os
import glob
import argparse
import concurrent.futures
import subprocess

def run_experiment(config_path, expt_name=None):
    """
    Run a single experiment by calling the training function.
    This example uses subprocess to invoke the training script.
    Adjust the command line call if your training logic is imported instead.
    """
    cmd = ["python", "train.py", "--config", config_path]
    if expt_name is not None:
        cmd.extend(["--name", expt_name])
    print(f"Launching experiment for {config_path} with command: {' '.join(cmd)}")
    subprocess.run(cmd)  # Will wait until the experiment finishes

def main():
    parser = argparse.ArgumentParser(description="Run experiments concurrently or sequentially for all config files in a folder")
    parser.add_argument('--config_folder', type=str, required=True,
                        help="Path to the folder containing YAML config files")
    parser.add_argument('--use_threads', action='store_true', help="Use threading (default is processes)")
    parser.add_argument('--sequential', action='store_true', help="Run experiments sequentially instead of concurrently")
    args = parser.parse_args()

    config_folder = args.config_folder
    config_files = glob.glob(os.path.join(config_folder, "*.yaml"))
    if not config_files:
        print(f"No YAML config files found in {config_folder}")
        return

    if args.sequential:
        print("Running experiments sequentially...")
        for config_path in config_files:
            expt_name = os.path.splitext(os.path.basename(config_path))[0]
            run_experiment(config_path, expt_name)
    else:
        Executor = concurrent.futures.ThreadPoolExecutor if args.use_threads else concurrent.futures.ProcessPoolExecutor
        with Executor() as executor:
            futures = []
            for config_path in config_files:
                expt_name = os.path.splitext(os.path.basename(config_path))[0]
                futures.append(executor.submit(run_experiment, config_path, expt_name))
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"An experiment raised an exception: {e}")

if __name__ == "__main__":
    main()

