#!/bin/bash

models=(
    "meta-llama/CodeLlama-7b-Instruct-hf"
    "meta-llama/Meta-Llama-3.1-8B-Instruct"
    "meta-llama/CodeLlama-70b-Instruct-hf"
    "meta-llama/Meta-Llama-3.1-70B-Instruct"
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
    "mistralai/Codestral-22B-v0.1"
    "google/gemma-2-9b-it"
    "google/gemma-2-27b-it"
)

temperatures=(0.2 0.8)
dir=DIR_HERE
data_dir=PATH_TO_DATA_DIR

for ((i=0; i<${#models[@]}; i++)); do
    model=${models[$i]}

    echo $model
    for temperature in "${temperatures[@]}"; do
        cat <<EOF > temp_sbatch_script.sh
#!/bin/bash
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
#SBATCH --time=10080
#SBATCH --qos=lowest
#SBATCH --mem=1760g
#SBATCH --account=fair_amaia_cw_codegen
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --partition=learn
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --open-mode=append
#SBATCH --signal=USR1@120
#SBATCH --distribution=block

i=\$SLURM_ARRAY_TASK_ID
ip=\$((\$i+1))

echo $dir
mkdir -p $dir

string="Starting iteration \$i"
echo \$string

eval "\$(CONDA_HOME shell.bash hook)"
conda activate vllm

if [[ "$model" == *"gemma"* ]]; then
    export VLLM_ATTENTION_BACKEND=FLASHINFER
fi

python -m inference.huggingface.run_huggingface \
    --model_name_or_path $model \
    --use_auth_token \
    --trust_remote_code \
    --precision bf16 \
    --temperature $temperature \
    --dataset_name_or_path $data_dir \
    --output_dir $dir
EOF
        chmod +x temp_sbatch_script.sh
        sbatch temp_sbatch_script.sh
        rm temp_sbatch_script.sh
    done
done
