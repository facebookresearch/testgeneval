#!/usr/bin/env python3

"""This python script is designed to run inference on a dataset using either the OpenAI or Anthropic API, depending on the model specified. 
It sorts instances by length and continually writes the outputs to a specified file, so that the script can be stopped and restarted without losing progress.
"""

import json
import logging
import os
import time
import traceback
from argparse import ArgumentParser
from pathlib import Path

import dotenv
import numpy as np
import openai
import tiktoken
from anthropic import AI_PROMPT, HUMAN_PROMPT, Anthropic
from datasets import DatasetDict, load_dataset, load_from_disk
from frozendict import frozendict
from inference.configs.instruct_prompt import InstructPrompt
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm.auto import tqdm
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
dotenv.load_dotenv()

MODEL_LIMITS = {
    "gpt-3.5-turbo-0125": 16_385,
    "gpt-4-turbo-2024-04-09": 128_000,
    "gpt-4o-2024-05-13": 128_000,
    "gpt-4-0613": 8_192,
    "Meta-Llama-3.1-405B-Instruct": 128_000,
}

# The cost per token for each model input.
MODEL_COST_PER_INPUT = {
    "gpt-3.5-turbo-0125": 0.0000005,
    "gpt-4-turbo-2024-04-09": 0.00001,
    "gpt-4o-2024-05-13": 0.000005,
    "gpt-4-0613": 0.00001,
    "Meta-Llama-3.1-405B-Instruct": 0,
}

# The cost per token for each model output.
MODEL_COST_PER_OUTPUT = {
    "gpt-3.5-turbo-0125": 0.0000015,
    "gpt-4-turbo-2024-04-09": 0.00003,
    "gpt-4o-2024-05-13": 0.000015,
    "gpt-4-0613": 0.00003,
    "Meta-Llama-3.1-405B-Instruct": 0,
}

OUTPUT_LIMITS = {
    "gpt-3.5-turbo-0125": 4_096,
    "gpt-4-turbo-2024-04-09": 8_192,
    "gpt-4o-2024-05-13": 4_096,
    "gpt-4-0613": 8_192,
    "Meta-Llama-3.1-405B-Instruct": 4_096,
}

EPSILON = 1000


def calc_cost(model_name, input_tokens, output_tokens):
    """
    Calculates the cost of a response from the openai API.

    Args:
    response (openai.ChatCompletion): The response from the API.

    Returns:
    float: The cost of the response.
    """
    cost = (
        MODEL_COST_PER_INPUT[model_name] * input_tokens
        + MODEL_COST_PER_OUTPUT[model_name] * output_tokens
    )
    logger.info(
        f"input_tokens={input_tokens}, output_tokens={output_tokens}, cost={cost:.2f}"
    )
    return cost


@retry(wait=wait_random_exponential(min=30, max=600), stop=stop_after_attempt(3))
def call_chat_llama_405B(
    model_name_or_path,
    client,
    inputs,
    temperature,
    top_p,
    max_tokens,
    system_message,
    **model_args,
):
    """
    Calls the OpenAI API to generate completions for the given inputs using the new API interface.

    Args:
        model_name_or_path (str): The name or path of the model to use.
        inputs (str): The inputs to generate completions for.
        temperature (float): The temperature to use.
        top_p (float): The top_p to use.
        **model_args (dict): Additional model arguments.

    Returns:
        tuple: A tuple containing the response and the cost of the completion.
    """
    user_message = inputs
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    model = "meta-llama/Meta-Llama-3.1-405B-Instruct"

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        print(completion.choices[0].message.content)
        return completion.choices[0].message.content, 0

    except Exception as e:
        print(f"API Error: {e}")
        raise


@retry(wait=wait_random_exponential(min=30, max=600), stop=stop_after_attempt(3))
def call_chat(
    model_name_or_path,
    inputs,
    temperature,
    top_p,
    max_tokens,
    system_message,
    **model_args,
):
    """
    Calls the OpenAI API to generate completions for the given inputs using the new API interface.

    Args:
        model_name_or_path (str): The name or path of the model to use.
        inputs (str): The inputs to generate completions for.
        temperature (float): The temperature to use.
        top_p (float): The top_p to use.
        **model_args (dict): Additional model arguments.

    Returns:
        tuple: A tuple containing the response and the cost of the completion.
    """
    user_message = inputs
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    try:
        response = openai.chat.completions.create(
            model=model_name_or_path,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,  # Adjust max_tokens as needed
            top_p=top_p,
            **model_args,
        )

        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cost = calc_cost(model_name_or_path, input_tokens, output_tokens)
        return response, cost

    except Exception as e:
        print(f"API Error: {e}")
        raise


def gpt_tokenize(string: str, encoding) -> int:
    """Returns the number of tokens in a text string."""
    num_tokens = len(encoding.encode(string))
    return num_tokens


def claude_tokenize(string: str, api) -> int:
    """Returns the number of tokens in a text string."""
    num_tokens = api.count_tokens(string)
    return num_tokens


def llama_405B_inference(
    test_dataset,
    model_name_or_path,
    output_file,
    model_args,
    existing_ids,
    max_cost,
    num_samples,
    postprocess_fn,
    system_message,
    system_message_full,
    skip_full,
):
    """
    Runs inference on a dataset using the openai API.

    Args:
    test_dataset (datasets.Dataset): The dataset to run inference on.
    model_name_or_path (str): The name or path of the model to use.
    output_file (str): The path to the output file.
    model_args (dict): A dictionary of model arguments.
    existing_ids (set): A set of ids that have already been processed.
    max_cost (float): The maximum cost to spend on inference.
    num_samples (int): The number of samples to generate for each prompt.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3.1-405B-Instruct",
        trust_remote_code=True,
        use_auth_token=True,
        truncation_side="left",
        padding_side="right",
    )
    model_limit = (
        MODEL_LIMITS[model_name_or_path] - OUTPUT_LIMITS[model_name_or_path] - EPSILON
    )

    def truncate_prompts(example):
        truncated = {}
        max_len = 0
        for key, prompt in example["preds_prompts"].items():
            tokens = tokenizer.encode(prompt)
            curr_len = min(len(tokens), model_limit)
            if curr_len > max_len:
                max_len = curr_len
            if len(tokens) > model_limit:
                # Truncate to the last model_limit tokens and decode back to text
                truncated_tokens = tokens[-model_limit:]
                truncated[key] = tokenizer.decode(
                    truncated_tokens
                )  # Assuming direct slicing works, adjust if necessary
            else:
                truncated[key] = prompt
        example["preds_prompts"] = truncated
        return example

    test_dataset = test_dataset.map(truncate_prompts, load_from_cache_file=False)
    print(model_args)
    temperature = model_args.pop("temperature", 0.2)
    top_p = model_args.pop("top_p", 0.95 if temperature > 0 else 1)
    print(f"Using temperature={temperature}, top_p={top_p}")
    basic_args = {
        "model_name_or_path": model_name_or_path + f"t={temperature}",
    }
    total_cost = 0
    print(f"Filtered to {len(test_dataset)} instances")
    with open(output_file, "a+") as f:
        for datum in tqdm(test_dataset, desc=f"Inference for {model_name_or_path}"):
            print(datum.keys())
            curr_id = datum["id"]
            if curr_id in existing_ids:
                continue
            output_dict = {"id": curr_id, "instance_id": datum["instance_id"]}
            output_dict.update(basic_args)
            output_dict["preds_prompts"] = datum["preds_prompts"]
            output_dict["preds"] = {}
            failed = False
            headers = frozendict(
                {
                    "Content-Type": "application/json",
                }
            )
            client = OpenAI(base_url=f"http://$HOSTNAME:8000/v1", api_key="EMPTY")
            for prompt_name, prompt_text in datum["preds_prompts"].items():
                prompt_predictions = []
                num_samples_curr = 1 if prompt_name == "full" else num_samples
                if skip_full and prompt_name == "full":
                    continue
                for _ in range(num_samples_curr):
                    try:
                        response, cost = call_chat_llama_405B(
                            model_name_or_path,
                            client,
                            prompt_text,
                            temperature,
                            top_p,
                            (
                                OUTPUT_LIMITS[model_name_or_path]
                                if prompt_name == "full"
                                else 512
                            ),
                            (
                                system_message_full
                                if prompt_name == "full"
                                else system_message
                            ),
                        )
                        completion = response
                        prompt_predictions.append(
                            postprocess_fn(completion, prompt_name == "full")
                        )
                        total_cost += cost
                        if max_cost is not None and total_cost >= max_cost:
                            print(f"Reached max cost {max_cost}, exiting")
                            return
                    except Exception as e:
                        print(f"Error: {e}")
                        failed = True

                    if failed:
                        break
                if failed:
                    break
                output_dict["preds"][prompt_name] = prompt_predictions
            if not failed:
                print(json.dumps(output_dict), file=f, flush=True)
                print(f"Total Cost: {total_cost:.2f}")
            else:
                print("Failed, skipping...")


def openai_inference(
    test_dataset,
    model_name_or_path,
    output_file,
    model_args,
    existing_ids,
    max_cost,
    num_samples,
    postprocess_fn,
    system_message,
    system_message_full,
    skip_full,
):
    """
    Runs inference on a dataset using the openai API.

    Args:
    test_dataset (datasets.Dataset): The dataset to run inference on.
    model_name_or_path (str): The name or path of the model to use.
    output_file (str): The path to the output file.
    model_args (dict): A dictionary of model arguments.
    existing_ids (set): A set of ids that have already been processed.
    max_cost (float): The maximum cost to spend on inference.
    num_samples (int): The number of samples to generate for each prompt.
    """
    encoding = tiktoken.encoding_for_model(model_name_or_path)
    model_limit = (
        MODEL_LIMITS[model_name_or_path] - OUTPUT_LIMITS[model_name_or_path] - EPSILON
    )

    # Adjust dataset to truncate prompts to the last model_limit tokens
    def truncate_prompts(example):
        truncated = {}
        for key, prompt in example["preds_prompts"].items():
            tokenized = encoding.encode(prompt)
            if len(tokenized) > model_limit:
                # Truncate to the last model_limit tokens and decode back to text
                truncated[key] = encoding.decode(tokenized[-model_limit:])
            else:
                truncated[key] = prompt
        example["preds_prompts"] = truncated
        return example

    test_dataset = test_dataset.map(truncate_prompts, load_from_cache_file=False)

    openai_key = os.environ.get("OPENAI_API_KEY", None)
    if openai_key is None:
        raise ValueError(
            "Must provide an api key. Expected in OPENAI_API_KEY environment variable."
        )
    openai.api_key = openai_key
    print(f"Using OpenAI key {'*' * max(0, len(openai_key)-5) + openai_key[-5:]}")
    print(model_args)
    temperature = model_args.pop("temperature", 0.2)
    top_p = model_args.pop("top_p", 0.95 if temperature > 0 else 1)
    print(f"Using temperature={temperature}, top_p={top_p}")
    basic_args = {
        "model_name_or_path": model_name_or_path + f"t={temperature}",
    }
    total_cost = 0
    print(f"Filtered to {len(test_dataset)} instances")
    with open(output_file, "a+") as f:
        for datum in tqdm(test_dataset, desc=f"Inference for {model_name_or_path}"):
            print(datum.keys())
            curr_id = datum["id"]
            if curr_id in existing_ids:
                continue
            output_dict = {"id": curr_id, "instance_id": datum["instance_id"]}
            output_dict.update(basic_args)
            output_dict["preds_prompts"] = datum["preds_prompts"]
            output_dict["preds"] = {}
            failed = False
            for prompt_name, prompt_text in datum["preds_prompts"].items():
                prompt_predictions = []
                num_samples_curr = 1 if prompt_name == "full" else num_samples
                if skip_full and prompt_name == "full":
                    continue
                for _ in range(num_samples_curr):
                    try:
                        response, cost = call_chat(
                            model_name_or_path,
                            prompt_text,
                            temperature,
                            top_p,
                            (
                                OUTPUT_LIMITS[model_name_or_path]
                                if prompt_name == "full"
                                else 512
                            ),
                            (
                                system_message_full
                                if prompt_name == "full"
                                else system_message
                            ),
                        )
                        completion = response.choices[0].message.content
                        print(postprocess_fn(completion, prompt_name == "full"))
                        prompt_predictions.append(
                            postprocess_fn(completion, prompt_name == "full")
                        )
                        total_cost += cost
                        if max_cost is not None and total_cost >= max_cost:
                            print(f"Reached max cost {max_cost}, exiting")
                            return
                    except Exception as e:
                        print(f"Error: {e}")
                        failed = True
                output_dict["preds"][prompt_name] = prompt_predictions
            if not failed:
                print(json.dumps(output_dict), file=f, flush=True)
                print(f"Total Cost: {total_cost:.2f}")
            else:
                print("Failed, skipping...")


@retry(wait=wait_random_exponential(min=60, max=600), stop=stop_after_attempt(6))
def call_anthropic(
    inputs,
    anthropic,
    model_name_or_path,
    temperature,
    top_p,
    max_tokens,
    system_message,
    **model_args,
):
    """
    Calls the anthropic API to generate completions for the given inputs.

    Args:
    inputs (str): The inputs to generate completions for.
    anthropic (Anthropic): The anthropic API object.
    model_name_or_path (str): The name or path of the model to use.
    temperature (float): The temperature to use.
    top_p (float): The top_p to use.
    model_args (dict): A dictionary of model arguments.
    """
    try:
        completion = anthropic.completions.create(
            model=model_name_or_path,
            max_tokens_to_sample=max_tokens,
            prompt=inputs,
            temperature=temperature,
            top_p=top_p,
            **model_args,
        )
        response = completion.completion
        input_tokens = anthropic.count_tokens(inputs)
        output_tokens = anthropic.count_tokens(response)
        cost = calc_cost(model_name_or_path, input_tokens, output_tokens)
        return completion, cost
    except Exception as e:
        logger.error(e)
        logger.error(f"Inputs: {inputs}")
        traceback.print_exc()
        time.sleep(20)
        return None


@retry(wait=wait_random_exponential(min=60, max=600), stop=stop_after_attempt(6))
def call_anthropic_v2(
    inputs,
    anthropic,
    model_name_or_path,
    temperature,
    top_p,
    max_tokens,
    system_message,
    **model_args,
):
    """
    Calls the anthropic API to generate completions for the given inputs.

    Args:
    inputs list(str): The inputs to generate completions for.
    anthropic (Anthropic): The anthropic API object.
    model_name_or_path (str): The name or path of the model to use.
    temperature (float): The temperature to use.
    top_p (float): The top_p to use.
    model_args (dict): A dictionary of model arguments.
    """
    user_message = inputs
    try:
        messages = [
            {"role": "user", "content": user_message},
        ]
        response = anthropic.messages.create(
            messages=messages,
            max_tokens=max_tokens,
            model=model_name_or_path,
            temperature=temperature,
            top_p=top_p,
            system=system_message,
        )
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost = calc_cost(response.model, input_tokens, output_tokens)
        return response, cost
    except Exception as e:
        logger.error(e)
        logger.error(f"Inputs: {inputs}")
        traceback.print_exc()
        time.sleep(20)
        return None


def anthropic_inference(
    test_dataset,
    model_name_or_path,
    output_file,
    model_args,
    existing_ids,
    max_cost,
    num_samples,
    postprocess_fn,
    system_message,
    system_message_full,
):
    """
    Runs inference on a dataset using the anthropic API.

    Args:
    test_dataset (datasets.Dataset): The dataset to run inference on.
    model_name_or_path (str): The name or path of the model to use.
    output_file (str): The path to the output file.
    model_args (dict): A dictionary of model arguments.
    existing_ids (set): A set of ids that have already been processed.
    max_cost (float): The maximum cost to spend on inference.
    num_samples (int): The number of samples to generate for each prompt.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", None)
    if api_key is None:
        raise ValueError(
            "Must provide an api key. Expected in ANTHROPIC_API_KEY environment variable."
        )
    print(f"Using Anthropic key {'*' * max(0, len(api_key)-5) + api_key[-5:]}")
    anthropic = Anthropic(api_key=api_key)
    model_limit = (
        MODEL_LIMITS[model_name_or_path] - OUTPUT_LIMITS[model_name_or_path] - EPSILON
    )

    # Adjust dataset to truncate prompts to the last model_limit tokens
    def truncate_prompts(example):
        truncated = {}
        for key, prompt in example["preds_prompts"].items():
            tokenized = anthropic.count_tokens(prompt)
            if tokenized > model_limit:
                # Truncate to the last model_limit tokens and decode back to text
                truncated[key] = prompt[
                    -model_limit:
                ]  # Assuming direct slicing works, adjust if necessary
            else:
                truncated[key] = prompt
        example["preds_prompts"] = truncated
        return example

    test_dataset = test_dataset.map(truncate_prompts, load_from_cache_file=False)

    temperature = model_args.pop("temperature", 0.2)
    top_p = model_args.pop("top_p", 0.95 if temperature > 0 else 1)
    print(f"Using temperature={temperature}, top_p={top_p}")
    basic_args = {
        "model_name_or_path": model_name_or_path + f"t={temperature}",
    }
    total_cost = 0
    print(f"Filtered to {len(test_dataset)} instances")
    if "claude-3" in model_name_or_path.lower():
        call_api = call_anthropic_v2
    else:
        call_api = call_anthropic
    with open(output_file, "a+") as f:
        for datum in tqdm(test_dataset, desc=f"Inference for {model_name_or_path}"):
            curr_id = datum["id"]
            if curr_id in existing_ids:
                continue
            output_dict = {"id": curr_id, "instance_id": datum["instance_id"]}
            output_dict.update(basic_args)
            output_dict["preds_prompts"] = datum["preds_prompts"]
            output_dict["preds"] = {}
            for prompt_name, prompt_text in datum["preds_prompts"].items():
                prompt_predictions = []
                num_samples_curr = 1 if prompt_name == "full" else num_samples
                if skip_full and prompt_name == "full":
                    continue
                for _ in range(num_samples_curr):
                    try:
                        completion, cost = call_api(
                            prompt_text,
                            anthropic,
                            model_name_or_path,
                            temperature,
                            top_p,
                            (
                                OUTPUT_LIMITS[model_name_or_path]
                                if prompt_name == "full"
                                else 512
                            ),
                            (
                                system_message_full
                                if prompt_name == "full"
                                else system_message**model_args
                            ),
                        )
                    except Exception as e:
                        logger.error(e)
                        traceback.print_exc()
                        continue
                    prompt_predictions.append(
                        postprocess_fn(completion.completion, prompt_name == "full")
                    )
                    total_cost += cost
                    if max_cost is not None and total_cost >= max_cost:
                        print(f"Reached max cost {max_cost}, exiting")
                        return
                output_dict["preds"][prompt_name] = prompt_predictions
            print(json.dumps(output_dict), file=f, flush=True)
            print(f"Total Cost: {total_cost:.2f}")


def parse_model_args(model_args):
    """
    Parses a string of model arguments and returns a dictionary of keyword arguments.

    Args:
        model_args (str): A string of comma-separated key-value pairs representing model arguments.

    Returns:
        dict: A dictionary of keyword arguments parsed from the input string.
    """
    kwargs = dict()
    if model_args is not None:
        for arg in model_args.split(","):
            key, value = arg.split("=")
            # infer value type
            if value in {"True", "False"}:
                kwargs[key] = value == "True"
            elif value.isnumeric():
                kwargs[key] = int(value)
            elif value.replace(".", "", 1).isnumeric():
                kwargs[key] = float(value)
            elif value in {"None"}:
                kwargs[key] = None
            elif value in {"[]"}:
                kwargs[key] = []
            elif value in {"{}"}:
                kwargs[key] = {}
            elif value.startswith("'") and value.endswith("'"):
                kwargs[key] = value[1:-1]
            elif value.startswith('"') and value.endswith('"'):
                kwargs[key] = value[1:-1]
            else:
                kwargs[key] = value
    return kwargs


def main(
    dataset_name_or_path,
    split,
    model_name_or_path,
    shard_id,
    num_shards,
    output_dir,
    no_imports,
    model_args,
    max_cost,
    num_samples,
    skip_full,
):
    if shard_id is None and num_shards is not None:
        logger.warning(
            f"Received num_shards={num_shards} but shard_id is None, ignoring"
        )
    if shard_id is not None and num_shards is None:
        logger.warning(f"Received shard_id={shard_id} but num_shards is None, ignoring")
    model_args = parse_model_args(model_args)

    prompt_info = InstructPrompt()

    model_nickname = model_name_or_path
    if "checkpoint" in Path(model_name_or_path).name:
        model_nickname = Path(model_name_or_path).parent.name
    else:
        model_nickname = Path(model_name_or_path).name

    temperature = model_args["temperature"] if "temperature" in model_args else 0.2
    output_file = f"{model_nickname}__{dataset_name_or_path.split('/')[-1]}__{temperature}__{split}"
    if shard_id is not None and num_shards is not None:
        output_file += f"__shard-{shard_id}__num_shards-{num_shards}"
    output_file = Path(output_dir, output_file + ".jsonl")
    logger.info(f"Will write to {output_file}")
    existing_ids = set()
    if os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                data = json.loads(line)
                curr_id = data["id"]
                existing_ids.add(curr_id)
    logger.info(f"Read {len(existing_ids)} already completed ids from {output_file}")
    if Path(dataset_name_or_path).exists():
        dataset = load_from_disk(dataset_name_or_path)
    else:
        dataset = load_dataset(dataset_name_or_path)

    dataset = prompt_info.add_prompts_to_dataset(dataset, no_import=no_imports)

    if not split in dataset:
        raise ValueError(f"Invalid split {split} for dataset {dataset_name_or_path}")
    dataset = dataset[split]

    print(dataset[0].keys())
    if len(existing_ids) > 0:
        dataset = dataset.filter(
            lambda x: x["id"] not in existing_ids,
            desc="Filtering out existing ids",
            load_from_cache_file=False,
        )
    if shard_id is not None and num_shards is not None:
        dataset = dataset.shard(num_shards, shard_id, contiguous=True)
    inference_args = {
        "test_dataset": dataset,
        "model_name_or_path": model_name_or_path,
        "output_file": output_file,
        "model_args": model_args,
        "existing_ids": existing_ids,
        "max_cost": max_cost,
        "num_samples": num_samples,
        "postprocess_fn": prompt_info.postprocess_output,
        "system_message": prompt_info.system_message,
        "system_message_full": prompt_info.system_message_full,
        "skip_full": skip_full,
    }
    if model_name_or_path.startswith("claude"):
        anthropic_inference(**inference_args)
    elif model_name_or_path == "Meta-Llama-3.1-405B-Instruct":
        llama_405B_inference(**inference_args)
    elif model_name_or_path.startswith("gpt"):
        openai_inference(**inference_args)
    else:
        raise ValueError(f"Invalid model name or path {model_name_or_path}")
    logger.info(f"Done!")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        required=True,
        help="HuggingFace dataset name or local path",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Name of API model. Update MODEL* constants in this file to add new models.",
        choices=sorted(list(MODEL_LIMITS.keys())),
        default="gpt-3.5-turbo-1106",
    )
    parser.add_argument(
        "--shard_id",
        type=int,
        default=None,
        help="Shard id to process. If None, process all shards.",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=None,
        help="Number of shards. If None, process all shards.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output file.",
    )
    parser.add_argument(
        "--no_imports",
        action="store_true",
        help="Use the no imports version of full.",
    )
    parser.add_argument(
        "--model_args",
        type=str,
        default=None,
        help="List of model arguments separated by commas. (e.g. 'top_p=0.95,temperature=0.70')",
    )
    parser.add_argument(
        "--max_cost",
        type=float,
        default=None,
        help="Maximum cost to spend on inference.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples to generate for each prompt.",
    )
    parser.add_argument(
        "--skip_full",
        help="Whether to skip full setting.",
        action="store_true",
    )
    args = parser.parse_args()
    print(args.model_args)
    main(**vars(args))
