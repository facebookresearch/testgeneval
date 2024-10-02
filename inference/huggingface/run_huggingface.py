import argparse
import fnmatch
import json
import logging
import os
import random
import sys
import traceback
from pathlib import Path

import datasets
import dotenv
import numpy as np
import torch
import transformers
from datasets import Dataset, load_dataset, load_from_disk
from inference.configs.codestral_prompt import CodestralPrompt
from inference.configs.gemma2_prompt import Gemma2Prompt
from inference.configs.instruct_prompt import InstructPrompt
from inference.configs.llama3_prompt import Llama3Prompt
from inference.huggingface.generator import Generator
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
dotenv.load_dotenv()

MODEL_CONFIG = {
    "CodeLlama-7b-Instruct-hf": {
        "tensor_parallel_size": 8,
        "max_context_window": 16_384,
        "prompt_class": InstructPrompt,
        "eager": True,
        "huggingface": False,
    },
    "CodeLlama-70b-Instruct-hf": {
        "tensor_parallel_size": 8,
        "max_context_window": 4_096,
        "prompt_class": InstructPrompt,
        "eager": True,
        "huggingface": False,
    },
    "Meta-Llama-3-8B-Instruct": {
        "tensor_parallel_size": 8,
        "max_context_window": 8_192,
        "prompt_class": Llama3Prompt,
        "eager": True,
        "huggingface": False,
    },
    "Meta-Llama-3-70B-Instruct": {
        "tensor_parallel_size": 8,
        "max_context_window": 8_192,
        "prompt_class": Llama3Prompt,
        "eager": True,
        "huggingface": False,
    },
    "Meta-Llama-3.1-8B-Instruct": {
        "tensor_parallel_size": 8,
        "max_context_window": 128_000,
        "prompt_class": Llama3Prompt,
        "eager": True,
        "huggingface": False,
    },
    "Meta-Llama-3.1-70B-Instruct": {
        "tensor_parallel_size": 8,
        "max_context_window": 128_000,
        "prompt_class": Llama3Prompt,
        "eager": True,
        "huggingface": False,
    },
    "DeepSeek-Coder-V2-Lite-Instruct": {
        "tensor_parallel_size": 8,
        "max_context_window": 128_000,
        "prompt_class": InstructPrompt,
        "eager": True,
        "huggingface": False,
    },
    "gemma-2-9b-it": {
        "tensor_parallel_size": 8,
        "max_context_window": 8_192,
        "prompt_class": Gemma2Prompt,
        "eager": True,
        "huggingface": True,
    },
    "gemma-2-27b-it": {
        "tensor_parallel_size": 8,
        "max_context_window": 8_192,
        "prompt_class": Gemma2Prompt,
        "eager": True,
        "huggingface": True,
    },
    "Codestral-22B-v0.1": {
        "tensor_parallel_size": 8,
        "max_context_window": 32_000,
        "prompt_class": CodestralPrompt,
        "eager": True,
        "huggingface": False,
    },
    "Meta-Llama-3.1-405B-Instruct": {
        "tensor_parallel_size": 16,
        "max_context_window": 128_000,
        "prompt_class": Llama3Prompt,
        "eager": True,
        "huggingface": False,
    },
}

EPSILON = 1000


def extract_prompts_from_raw_files(raw_outputs, raw_output_file, prompt_output):
    """
    Extracts prompts from the raw output file and writes them to a separate prompt file.
    This function should be used when the raw file exists but the prompt file does not.
    """
    prompts = {}

    with open(raw_output_file, "w") as raw_file:
        for fn in raw_outputs:
            new_raw_output = []

            with open(fn, "r") as raw_file_curr:
                raw_lines = raw_file_curr.read().splitlines()
                for line in raw_lines:
                    try:
                        data = json.loads(line)
                        data_id = data["id"]
                        if data_id not in prompts:
                            prompts[data_id] = {
                                "id": data_id,
                                "setting": data["setting"],
                                "prompt": data["prompt"],
                            }

                        del data["prompt"]
                        raw_file.write(json.dumps(data) + "\n")
                    except:
                        pass

        with open(prompt_output, "w") as prompt_file:
            for prompt_data in prompts.values():
                prompt_file.write(json.dumps(prompt_data) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        default="",
        help="Model to evaluate, provide a repo name in Hugging Face hub or a local path",
    )
    parser.add_argument(
        "--use_auth_token",
        action="store_true",
        help="Use the token generated when running `huggingface-cli login` (necessary for private model).",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Use a model with custom code, this requires executing code by the author of the model.",
    )
    parser.add_argument(
        "--no_imports",
        action="store_true",
        help="Use the no imports version of full.",
    )
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        required=True,
        help="HuggingFace dataset name or local path",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        help="Model precision, from: fp32, fp16 or bf16",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Model temperature.",
    )
    parser.add_argument(
        "--num_samples_completion",
        type=int,
        default=5,
        help="Number of samples to generate for completing tests.",
    )
    parser.add_argument(
        "--num_samples_generation",
        type=int,
        default=1,
        help="Number of samples to generate full file.",
    )

    parser.add_argument(
        "--local_model_name", type=str, help="Name of local model.", default=""
    )

    parser.add_argument("--skip_full", help="Skip full setting", action="store_true")

    parser.add_argument(
        "--context_size",
        help="Context window size",
        default=-1,
        type=int,
    )

    parser.add_argument(
        "--skip_completion", help="Skip completion setting", action="store_true"
    )
    args = parser.parse_args()

    precision_map = {
        "fp32": "float32",
        "fp16": "float16",
        "bf16": "bfloat16",
    }

    args.precision = precision_map[args.precision]

    return args


def split_dataset(dataset, existing_ids):
    new_ds_full = []
    new_ds_completion = []

    for datum in dataset:
        for prompt_name, prompt in datum["preds_prompts"].items():
            if datum["id"] in existing_ids and prompt_name in existing_ids[datum["id"]]:
                continue
            if prompt_name == "full":
                new_ds_full.append(
                    {
                        "instance_id": datum["instance_id"],
                        "id": datum["id"],
                        "setting": prompt_name,
                        "prompt": prompt,
                    }
                )
            else:
                new_ds_completion.append(
                    {
                        "instance_id": datum["instance_id"],
                        "id": datum["id"],
                        "setting": prompt_name,
                        "prompt": prompt,
                    }
                )
    return Dataset.from_list(new_ds_full), Dataset.from_list(new_ds_completion)


def process_raw_output(raw_output, prompt_output, full_output, model_name_or_path):
    id_mappings = {}

    with open(raw_output, "r") as raw_file, open(prompt_output, "r") as prompt_file:
        raw_lines = raw_file.read().splitlines()
        prompt_lines = prompt_file.read().splitlines()

        prompts = {}
        for line in prompt_lines:
            prompt_data = json.loads(line)
            prompts[prompt_data["id"]] = prompt_data

        preds = [json.loads(line) for line in raw_lines]

        for pred in preds:
            data_id = pred["id"]
            instance_id = pred["instance_id"]
            if data_id not in id_mappings:
                id_mappings[data_id] = {
                    "instance_id": instance_id,
                    "id": data_id,
                    "model_name_or_path": model_name_or_path,
                    "preds_prompts": {},
                    "preds": {},
                }

            setting = pred["setting"]
            prompt = prompts[data_id]["prompt"]
            pred_text = pred["pred"]

            id_mappings[data_id]["preds_prompts"][setting] = prompt
            if setting not in id_mappings[data_id]["preds"]:
                id_mappings[data_id]["preds"][setting] = [pred_text]
            else:
                id_mappings[data_id]["preds"][setting].append(pred_text)

    with open(full_output, "w") as full_file:
        for data_id, data in id_mappings.items():
            print(json.dumps(data), file=full_file, flush=True)


def truncate_prompts(dataset, tokenizer, max_tokens):
    """Truncate prompts in the dataset to the last max_tokens"""

    def truncate(example):
        prompt = example["prompt"]
        tokens = tokenizer.encode(prompt)
        if len(tokens) > max_tokens:
            truncated_tokens = tokens[-max_tokens:]
            example["prompt"] = tokenizer.decode(truncated_tokens)
        return example

    return dataset.map(truncate, load_from_cache_file=False)


def main():
    args = parse_args()
    model_nickname = Path(args.model_name_or_path).name

    model_config = (
        MODEL_CONFIG.get(model_nickname)
        if len(args.local_model_name) == 0
        else MODEL_CONFIG.get(args.local_model_name)
    )

    if args.context_size != -1:
        model_config["max_context_window"] = args.context_size

    max_tokens = model_config["max_context_window"] - EPSILON
    tensor_parallel_size = model_config["tensor_parallel_size"]
    if tensor_parallel_size > 8:
        import ray

        ray.init(address="auto")

    prompt_info = model_config["prompt_class"]()
    huggingface = model_config["huggingface"]

    transformers.logging.set_verbosity_error()
    datasets.logging.set_verbosity_error()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        use_auth_token=args.use_auth_token,
        truncation_side="left",
        padding_side="right",
    )

    if huggingface:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            use_auth_token=args.use_auth_token if args.use_auth_token else None,
            torch_dtype=args.precision,
            device_map="auto",
            trust_remote_code=args.trust_remote_code,
        )

    else:
        model = LLM(
            model=args.model_name_or_path,
            dtype=args.precision,
            trust_remote_code=args.trust_remote_code,
            gpu_memory_utilization=0.98,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=model_config["max_context_window"],
            enforce_eager=model_config["eager"],
        )

    stop_token_ids = (
        None
        if "Llama-3" not in model_nickname
        else [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    )

    if not tokenizer.eos_token:
        if tokenizer.bos_token:
            tokenizer.eos_token = tokenizer.bos_token
            print("bos_token used as eos_token")
        else:
            raise ValueError("No eos_token or bos_token found")
    tokenizer.pad_token = tokenizer.eos_token

    if Path(args.dataset_name_or_path).exists():
        dataset = load_from_disk(args.dataset_name_or_path)
        dataset_nickname = Path(args.dataset_name_or_path).name
    else:
        dataset = load_dataset(args.dataset_name_or_path)
        dataset_nickname = args.dataset_name_or_path

    output_file = f"{model_nickname}__{dataset_nickname}__{args.temperature}__test"

    if args.context_size != -1:
        output_file += f"__{args.context_size}"

    output_file_raw = output_file + "__raw"
    output_file_prompt = output_file + "__prompt"

    output_file = Path(args.output_dir, output_file + ".jsonl")
    output_file_raw = Path(args.output_dir, output_file_raw + ".jsonl")
    output_file_prompt = Path(args.output_dir, output_file_prompt + ".jsonl")

    # Generate the prompt file if it doesn't exist
    dataset = prompt_info.add_prompts_to_dataset(
        dataset, tokenizer=tokenizer, no_import=args.no_imports
    )["test"]

    existing_ids_full = {}
    lines = []

    lines_full = []
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            try:
                for line in f:
                    data = json.loads(line)
                    data_id = data["id"]
                    instance_id = data["instance_id"]
                    for setting in data["preds"]:
                        for pred in data["preds"][setting][:5]:
                            lines_full.append(
                                {
                                    "instance_id": instance_id,
                                    "id": data_id,
                                    "setting": setting,
                                    "prompt": data["preds_prompts"][setting],
                                    "pred": pred,
                                }
                            )
            except Exception as e:
                print(e)

        with open(output_file_raw, "w") as f:
            for line in lines_full:
                f.write(json.dumps(line) + "\n")

    extract_prompts_from_raw_files([], output_file_raw, output_file_prompt)

    if os.path.exists(output_file_raw):
        with open(output_file_raw, "r") as f:
            try:
                for line in f:
                    data = json.loads(line)
                    lines.append(data)
                    data_id = data["id"]
                    setting = data["setting"]
                    if data_id not in existing_ids_full:
                        existing_ids_full[data_id] = set([setting])
                    else:
                        existing_ids_full[data_id].add(setting)
            except Exception as e:
                print(e)

        os.remove(output_file_raw)

        with open(output_file_raw, "w") as f:
            for line in lines:
                if "prompt" in line:
                    del line["prompt"]
                f.write(json.dumps(line) + "\n")

    print(f"Len of existing ids: {len(existing_ids_full)}")

    if len(dataset) > 0:
        dataset_full, dataset_completion = split_dataset(dataset, existing_ids_full)

        generator = Generator(
            model,
            tokenizer,
            args.temperature,
            output_file_raw,
            output_file_prompt,
            use_huggingface=huggingface,
        )

        if len(dataset_full) > 0 and not args.skip_full:
            dataset_full = truncate_prompts(dataset_full, tokenizer, max_tokens)
            generator.generate(
                dataset_full,
                prompt_info.postprocess_output,
                True,
                args.num_samples_generation,
                8192,
                stop_token_ids,
            )
        if len(dataset_completion) > 0 and not args.skip_completion:
            dataset_completion = truncate_prompts(
                dataset_completion, tokenizer, max_tokens
            )
            generator.generate(
                dataset_completion,
                prompt_info.postprocess_output,
                False,
                args.num_samples_completion,
                512,
                stop_token_ids,
            )

        process_raw_output(
            output_file_raw,
            output_file_prompt,
            output_file,
            model_nickname + f"t={args.temperature}",
        )


if __name__ == "__main__":
    main()
