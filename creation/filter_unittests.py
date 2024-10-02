import argparse
import ast
import copy
import json

from ast_utils import get_local_import_statements
from datasets import Dataset, DatasetDict, load_from_disk

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--baseline_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="")
    args = parser.parse_args()

    dataset = load_from_disk(args.dataset_path)

    test_data = dataset["test"]

    with open(args.baseline_path, "r") as f:
        baseline_data = json.load(f)

    final_arr = []

    for i in range(len(test_data)):
        test_datum = test_data[i]
        if test_datum["id"] not in baseline_data:
            continue

        baseline_datum = baseline_data[test_datum["id"]]

        bad_data = False
        for setting in ["first", "last_minus_one", "last"]:
            try:
                if baseline_datum[setting]["coverage"][0] == -1:
                    bad_data = True
            except Exception as e:
                bad_data = True

        if bad_data:
            continue

        # Backwards compatibility with old baseline data
        if "golds" in test_datum:
            del test_datum["golds"]

        test_datum["baseline_covs"] = {
            "first": baseline_datum["first"]["coverage"][0],
            "last_minus_one": baseline_datum["last_minus_one"]["coverage"][0],
            "last": baseline_datum["last"]["coverage"][0],
        }

        final_arr.append(test_datum)

    fin_set = {"test": Dataset.from_list(final_arr)}

    for split in dataset:
        if split != "test":
            fin_set[split] = dataset[split]

    final_dataset = DatasetDict(fin_set)

    final_dataset.save_to_disk(args.output_path)
