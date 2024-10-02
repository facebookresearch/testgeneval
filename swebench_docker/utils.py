import json
import os

from datasets import load_dataset, load_from_disk
from swebench_docker.constants import KEY_ID


def get_eval_refs(data_path_or_name):
    decode_keys = False
    if os.path.isfile(data_path_or_name):
        if data_path_or_name.endswith(".jsonl"):
            data = [json.loads(l) for l in open(data_path_or_name).readlines()]
        elif data_path_or_name.endswith(".json"):
            data = json.load(open(data_path_or_name, "r"))
    elif os.path.isdir(data_path_or_name):
        data = load_from_disk(data_path_or_name)
        decode_keys = True
    else:
        data = load_dataset(data_path_or_name)
        decode_keys = True
    if isinstance(data, dict):
        all_data = list()
        all_data.extend(data["test"])
        data = all_data

    return {d[KEY_ID]: d for d in data}
