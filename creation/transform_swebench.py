import copy
import glob
import os
import shutil
import signal
import subprocess
import traceback
from argparse import ArgumentParser
from contextlib import contextmanager

import git
from ast_utils import get_local_import_statements
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from get_prompt_contexts import get_context_gold_json
from thefuzz import fuzz, process
from tqdm import tqdm


# Define a timeout exception class
class TimeoutException(Exception):
    pass


# Define a context manager to handle the timeout
@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def extract_files_from_patch(patch_content):
    files = set()
    for line in patch_content.splitlines():
        if line.startswith("diff --git a/"):
            parts = line.split(" ")
            if len(parts) >= 3:
                file_path = parts[2]
                files.add(file_path[2:])
    return list(files)


def heuristic_name_check_python(test_file_paths, code_file_path, test_files, code_file):
    candidates = []
    for i in range(len(test_file_paths)):
        test_file = test_files[i]
        if test_file == code_file + "_test":
            candidates.append(test_file_paths[i])
        if test_file == "test_" + code_file:
            candidates.append(test_file_paths[i])

    # no heuristic, match, return None
    if len(candidates) == 0:
        return None, -1

    # want to pick test file path with closest match to code file path
    test_file, score = process.extractOne(
        code_file_path, test_file_paths, scorer=fuzz.partial_ratio
    )
    return test_file, score


def get_closest_test(test_file_paths, code_file_path):
    """Align python code and test files based on fuzzy string match."""

    test_files = [
        test_file_path.split("/")[-1][:-3].lower() for test_file_path in test_file_paths
    ]
    code_file = code_file_path.split("/")[-1][:-3].lower()

    closest_test, score = heuristic_name_check_python(
        test_file_paths, code_file_path, test_files, code_file
    )

    if closest_test is not None:
        return closest_test, score

    test_file, test_score = process.extractOne(
        code_file, test_files, scorer=fuzz.partial_ratio
    )

    if test_score > 85:
        test_filepath, score = process.extractOne(
            test_file + ".py", test_file_paths, scorer=fuzz.partial_ratio
        )

        return test_filepath, test_score

    return None, -1


def align_dataset(dataset, is_train=False):
    new_data = []

    for data in dataset:
        curr_data = {
            "repo": data["repo"],
            "base_commit": data["base_commit"],
            "version": data["version"],
            "instance_id": data["instance_id"],
            "patch": data["patch"],
            "test_patch": data["test_patch"],
        }

        code_files = extract_files_from_patch(data["patch"])

        test_files = extract_files_from_patch(data["test_patch"])

        if not is_train:
            fp_map = {}
            if len(code_files) == 1 and len(test_files) == 1:
                fp_map[code_files[0]] = test_files[0]

            else:
                for code_file in code_files:
                    test_file, score = get_closest_test(test_files, code_file)
                    if test_file is not None:
                        fp_map[code_file] = test_file

            if len(fp_map) > 0:
                curr_data["file_pairs"] = fp_map
                new_data.append(curr_data)
        else:
            new_data.append(curr_data)

    return new_data


def apply_patch(repo_path, patch_content):
    patch_file_path = os.path.join(repo_path, "temp_patch.patch")
    with open(patch_file_path, "w") as patch_file:
        patch_file.write(patch_content)

    try:
        subprocess.check_call(["git", "apply", "temp_patch.patch"], cwd=repo_path)
    finally:
        os.remove(patch_file_path)


def process_repo(repo_data, repo_dir, id_curr, errors, is_train):
    repo_dir = os.path.join(repo_dir, repo_data["repo"].replace("/", "_"))
    repo_url = f"https://github.com/{repo_data['repo']}.git"
    base_commit = repo_data["base_commit"]
    patch = repo_data["patch"]
    test_patch = repo_data["test_patch"]

    total_repo_data = []

    print(f"Processing {repo_data['repo']}")
    print(os.path.exists(repo_dir))
    # Check if the repository already exists
    if os.path.exists(repo_dir):
        repo = git.Repo(repo_dir)
        print(f"Resetting {repo_data['repo']} to {base_commit}")
        repo.git.reset("--hard")
        repo.git.clean("-xdf")
        repo.git.checkout(base_commit)
    else:
        print(f"Cloning {repo_data['repo']} to {repo_dir}")
        print(repo_url)
        repo = git.Repo.clone_from(repo_url, repo_dir)
        repo.git.checkout(base_commit)

    # Apply the patches
    apply_patch(repo_dir, patch)

    if not is_train:
        apply_patch(repo_dir, test_patch)

    cwd = os.getcwd()
    os.chdir(repo_dir)

    if is_train:
        fp_map = {}
        code_files = extract_files_from_patch(patch)
        test_files = glob.glob("**/test_*", recursive=True)

        if len(code_files) == 1 and len(test_files) == 1:
            fp_map[code_files[0]] = test_files[0]

        else:
            for code_file in code_files:
                test_file, score = get_closest_test(test_files, code_file)
                if test_file is not None:
                    fp_map[code_file] = test_file

        if len(fp_map) > 0:
            repo_data["file_pairs"] = fp_map

    if "file_pairs" in repo_data:
        for code_file, test_file in repo_data["file_pairs"].items():
            new_data = copy.copy(repo_data)
            context = get_context_gold_json(code_file, test_file)

            if context is None:
                errors["no_context"] += 1
                continue

            if not os.path.exists(code_file) or not os.path.exists(test_file):
                print("Skipping file not found")
                errors["file_not_found"] += 1
                continue

            code_src = open(code_file, "r").read()

            new_data["preds_context"] = {
                "none": "",
                "preamble": context["preamble"],
                "first": context["first"],
                "last_minus_one": context["last_minus_one"],
                "last": context["last"],
            }
            # new_data["golds"] = {"none": [""], "first": [context["first"]], "last": [context["last"]], "extra": [context["last"]]}
            new_data["code_src"] = code_src
            new_data["test_src"] = open(test_file, "r").read()
            new_data["code_file"] = code_file
            new_data["test_file"] = test_file

            code_file_suf = code_file.split("/")[-1].split(".py")[0]
            new_data["local_imports"] = get_local_import_statements(
                code_src, code_file, code_file_suf, new_data["test_src"], test_file
            )

            del new_data["file_pairs"]

            if "__init__" in code_file:
                errors["init_file"] += 1
                continue

            if len(new_data["local_imports"]) == 0:
                errors["no_local_imports"] += 1
                continue

            new_data["id"] = f"{new_data['instance_id']}-{id_curr}"
            id_curr += 1
            total_repo_data.append(new_data)

    os.chdir(cwd)
    return total_repo_data, id_curr


def aligned_dataset_to_prompts(
    aligned_dataset, repo_dir, id_curr, errors, timeout=600, is_train=False
):
    prompts = []
    for data in tqdm(aligned_dataset):
        cwd = os.getcwd()
        try:
            with time_limit(timeout):
                new_data, new_id_curr = process_repo(
                    data, repo_dir, errors, id_curr, is_train
                )
                prompts.extend(new_data)
                id_curr = new_id_curr
                os.chdir(cwd)
        except TimeoutException as e:
            print(e)
            print(f"Timeout occurred for instance {data['instance_id']}")
            os.chdir(cwd)
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            print(f"Skipping instance {data['instance_id']}")
            os.chdir(cwd)

    return Dataset.from_list(prompts), id_curr


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_name_or_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--repo_dir", type=str, default="repos")
    parser.add_argument("--select_one", action="store_true")
    parser.add_argument(
        "--splits",
        type=str,
        help="Comma separated list of splits to select from dataset",
        default="dev,test",
    )

    args = parser.parse_args()

    dataset = (
        load_from_disk(args.dataset_name_or_path)
        if os.path.exists(args.dataset_name_or_path)
        else load_dataset(args.dataset_name_or_path)
    )

    data_dict = {}
    splits = args.splits.split(",")
    id_curr = 0
    for split in splits:
        errors = {
            "no_local_imports": 0,
            "no_context": 0,
            "file_not_found": 0,
            "init_file": 0,
        }
        test_data = dataset[split]

        print(f"Processing {split} split")

        if args.select_one:
            test_data = test_data.select(range(1))

        aligned_dataset = align_dataset(test_data, split == "train")
        new_data, new_id_curr = aligned_dataset_to_prompts(
            aligned_dataset, args.repo_dir, id_curr, errors, 600, split == "train"
        )
        data_dict[split] = new_data
        id_curr = new_id_curr
        print(errors)

    final_dataset = DatasetDict(data_dict)

    final_dataset.save_to_disk(args.output_path)
