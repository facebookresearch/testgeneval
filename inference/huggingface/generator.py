import sys
from math import ceil

import numpy as np
from inference.huggingface.huggingface_utils import TokenizedDataset, complete_code
from torch.utils.data import DataLoader
from vllm import SamplingParams


class Generator:
    def __init__(
        self,
        model,
        tokenizer,
        temperature,
        output_file,
        prompt_file,
        shuffle=True,
        use_huggingface=False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.shuffle = shuffle
        self.output_file = output_file
        self.prompt_file = prompt_file  # New prompt file parameter
        self.temperature = temperature
        self.use_huggingface = use_huggingface

    def generate(
        self,
        dataset,
        postprocess_fn,
        is_full,
        num_samples,
        max_len,
        stop_token_ids=None,
    ):
        dataset_rows = range(dataset.num_rows)

        # shuffle the dataset
        if self.shuffle:
            dataset_rows = np.random.permutation(dataset_rows)
            dataset = dataset.select(dataset_rows)

        n_tasks = dataset.num_rows

        ds_tokenized = TokenizedDataset(
            dataset,
            self.tokenizer,
            n_tasks,
        )

        sampling_params = SamplingParams(
            n=num_samples,
            temperature=self.temperature,
            top_p=0.95,
            top_k=-1,
            max_tokens=max_len,
        )

        if stop_token_ids is not None:
            sampling_params.stop_token_ids = stop_token_ids

        ds_loader = DataLoader(ds_tokenized, batch_size=1)

        # Pass both the output_file and prompt_file to complete_code
        complete_code(
            self.model,
            self.tokenizer,
            sampling_params,
            ds_loader,
            num_samples,
            n_tasks,
            postprocess_fn,
            is_full,
            self.output_file,
            self.prompt_file,
            self.use_huggingface,
        )
