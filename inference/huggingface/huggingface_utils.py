import json
import math
import warnings
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import IterableDataset
from tqdm import tqdm


class TokenizedDataset(IterableDataset):
    """Tokenize and preprocess the dataset
    Multiple copies of the same prompt are sent sequentially. See compute_code for more details.
    The prompt can either be:
    - one prompt: normal code completion
    - two prompts: for infilling mode (prefix, suffix) or instruction-tuning mode (instruction, context)
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        n_tasks,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.n_tasks = n_tasks

    def __iter__(self):
        prompts = []
        instance_ids = []
        ids = []
        settings = []
        for sample in range(self.n_tasks):
            dataset_sample = self.dataset[sample]
            prompt_contents = dataset_sample["prompt"]
            assert isinstance(prompt_contents, str)
            prompts.append(prompt_contents)
            instance_ids.append(dataset_sample["instance_id"])
            ids.append(dataset_sample["id"])
            settings.append(dataset_sample["setting"])

        return_token_type_ids = None  # default

        outputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=False,
            return_tensors="pt",
            return_token_type_ids=return_token_type_ids,
        )

        for sample in range(self.n_tasks):
            yield {
                "instance_id": instance_ids[sample],
                "id": ids[sample],
                "prompt": prompts[sample],
                "setting": settings[sample],
                "ids": outputs.input_ids[sample],
                "input_len": outputs.attention_mask[sample].sum(),
            }


def complete_code(
    model,
    tokenizer,
    sampling_params,
    dataloader,
    num_samples,
    n_tasks,
    postprocess_fn,
    is_full,
    output_file,
    prompt_file,
    use_huggingface,
):
    total = math.ceil(n_tasks)
    if use_huggingface:
        model.eval()
    with open(output_file, "a+") as pred_f, open(prompt_file, "a+") as prompt_f:
        if use_huggingface:
            with torch.no_grad():
                for step, batch in tqdm(enumerate(dataloader), total=total):
                    inputs = batch["ids"][:, : batch["input_len"]].tolist()
                    input_tensor = torch.Tensor(inputs).long().cuda()
                    generated_texts = []
                    for i in range(num_samples):
                        outputs = model.generate(
                            input_ids=input_tensor,
                            max_new_tokens=sampling_params.max_tokens,
                            temperature=sampling_params.temperature,
                            do_sample=True,
                        )
                        outputs = outputs[0][
                            batch["input_len"] :
                        ]  # Skip the input tokens
                        generated_texts += [
                            tokenizer.decode(outputs, skip_special_tokens=True)
                        ]

                    generated_instance_ids = (
                        np.array(batch["instance_id"]).repeat(num_samples).tolist()
                    )
                    generated_ids = np.array(batch["id"]).repeat(num_samples).tolist()
                    generated_settings = (
                        np.array(batch["setting"]).repeat(num_samples).tolist()
                    )
                    prompts = np.array(batch["prompt"]).repeat(num_samples).tolist()

                    # Write prompts to the separate prompt file
                    for prompt, setting, curr_id in zip(
                        prompts, generated_settings, generated_ids
                    ):
                        prompt_data = {
                            "id": curr_id,
                            "setting": setting,
                            "prompt": prompt,
                        }
                        prompt_f.write(json.dumps(prompt_data) + "\n")

                    # Write predictions to the separate prediction file
                    for instance_id, curr_id, setting, text in zip(
                        generated_instance_ids,
                        generated_ids,
                        generated_settings,
                        generated_texts,
                    ):
                        text_processed = postprocess_fn(text, is_full)
                        pred_f.write(
                            json.dumps(
                                {
                                    "instance_id": instance_id,
                                    "id": curr_id,
                                    "setting": setting,
                                    "pred": text_processed,
                                }
                            )
                            + "\n"
                        )

        else:
            for step, batch in tqdm(enumerate(dataloader), total=total):
                inputs = batch["ids"][:, : batch["input_len"]].tolist()

                outputs = model.generate(
                    prompt_token_ids=inputs,
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )
                generated_texts = [o.text for o in outputs[0].outputs]

                generated_instance_ids = (
                    np.array(batch["instance_id"]).repeat(num_samples).tolist()
                )
                generated_ids = np.array(batch["id"]).repeat(num_samples).tolist()
                generated_settings = (
                    np.array(batch["setting"]).repeat(num_samples).tolist()
                )
                prompts = np.array(batch["prompt"]).repeat(num_samples).tolist()

                # Write prompts to the separate prompt file
                for prompt, setting, curr_id in zip(
                    prompts, generated_settings, generated_ids
                ):
                    prompt_data = {"id": curr_id, "setting": setting, "prompt": prompt}
                    prompt_f.write(json.dumps(prompt_data) + "\n")

                # Write predictions to the separate prediction file
                for instance_id, curr_id, setting, text in zip(
                    generated_instance_ids,
                    generated_ids,
                    generated_settings,
                    generated_texts,
                ):
                    text_processed = postprocess_fn(text, is_full)
                    pred_f.write(
                        json.dumps(
                            {
                                "instance_id": instance_id,
                                "id": curr_id,
                                "setting": setting,
                                "pred": text_processed,
                            }
                        )
                        + "\n"
                    )
