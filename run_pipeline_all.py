import argparse
import subprocess

# Define the models and temperatures
models = [
    "meta-llama/CodeLlama-7b-Instruct-hf",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/CodeLlama-70b-Instruct-hf",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "meta-llama/Meta-Llama-3.1-405B-Instruct",
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    "mistralai/Codestral-22B-v0.1",
    "google/gemma-2-9b-it",
    "google/gemma-2-27b-it",
    "gpt-4o-2024-05-13",
]
temperatures = [0.2, 0.8]


# Function to activate conda and run the command
def run_command(model, temperature, results_dir, data_dir, num_processes):
    num_samples = 1 if temperature == 0.2 else 5
    skip_full = "" if temperature == 0.2 else "--skip_full"
    command = f"""
    eval "$(conda shell.bash hook)"
    conda activate swebench-testing
    python run_pipeline.py --results_dir {results_dir} --dataset_dir {data_dir} --model {model} --temperature {temperature} --num_processes {num_processes} --azure --num_samples {num_samples} {skip_full} --rerun_eval
    """
    subprocess.run(command, shell=True, executable="/bin/bash")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to run entire evaluation pipeline"
    )
    parser.add_argument(
        "--results_dir", type=str, help="Path to results directory", required=True
    )
    parser.add_argument(
        "--dataset_dir", type=str, help="Path to dataset directory", required=True
    )
    parser.add_argument(
        "--num_processes", type=int, help="Number of processes to run", default=64
    )

    args = parser.parse_args()

    # Execute commands sequentially
    for model in models:
        for temperature in temperatures:
            run_command(
                model,
                temperature,
                args.results_dir,
                args.dataset_dir,
                args.num_processes,
            )
