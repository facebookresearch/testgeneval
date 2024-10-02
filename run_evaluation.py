#!/usr/bin/env python3
# Adapted from: https://github.com/aorwall/SWE-bench-docker/blob/main/run_evaluation.py
"""Run evaluation"""
import argparse
import asyncio
import hashlib
import logging
import os

from swebench_docker.constants import (
    KEY_ID,
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTIONS,
    MAP_REPO_TO_TEST_FRAMEWORK,
)
from swebench_docker.run_docker import run_docker_evaluation
from swebench_docker.swebench_utils import get_instances, get_test_directives
from swebench_docker.utils import get_eval_refs

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("run_evaluation")


def validate_predictions(predictions_path, tasks_ids):
    # Check that predictions file exists
    if not any([predictions_path.endswith(x) for x in [".jsonl"]]):
        raise ValueError("Predictions path must be .jsonl file")
    predictions = get_instances(predictions_path)
    not_in_tasks = []
    # Check that predictions are correctly formatted
    for pred in predictions:
        if any(
            [
                x not in pred
                for x in [KEY_ID, KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTIONS]
            ]
        ):
            raise ValueError(
                f"Every prediction must have {KEY_INSTANCE_ID}, {KEY_MODEL}, {KEY_ID}, and {KEY_PREDICTIONS} fields"
            )
        if pred[KEY_ID] not in tasks_ids:
            not_in_tasks.append(pred[KEY_ID])
    # Check that instance IDs specified by predictions exist
    if len(not_in_tasks) > 0:
        logger.warning(
            "Predictions for the following instance_ids were not "
            + "found in the tasks file and will not be considered: "
            + ", ".join(not_in_tasks)
        )


async def main(
    predictions_path: str,
    swe_bench_tasks: str,
    namespace: str,
    log_dir: str,
    skip_existing: bool = False,
    timeout: int = 900,
    num_processes: int = -1,
    skip_mutation: bool = False,
):
    """
    Runs evaluation on predictions for each model/repo/version combination.

    Args:
        predictions_path (str): Path to the predictions file.
        swe_bench_tasks (str): Path to the SWE-bench tasks file OR HF dataset name.
        namespace (str): Docker repository namespace.
        log_dir (str): Path to the directory where logs will be saved.
        skip_existing (bool): Whether to skip evaluations for predictions that already have logs.
        timeout (int): Timeout for each evaluation.
        num_processes (int): Number of processes to run in parallel (-1 = unlimited)

    Raises:
        ValueError: If log_dir is not a directory, testbed is not a directory, or swe_bench_tasks does not exist.
    """
    # Validate arguments
    if not os.path.exists(log_dir) or not os.path.isdir(log_dir):
        raise ValueError("--log_dir must exist and point at a directory")

    # Make sure that the log directory is world-writable so that we can write to it from
    # within the container, even if we're not the root user.
    os.chmod(log_dir, 0o777)

    tasks = list(get_eval_refs(swe_bench_tasks).values())

    # Verify arguments are formatted correctly
    if not isinstance(tasks, list):
        raise ValueError(f"{swe_bench_tasks} must contain an array of tasks")
    tasks_map = {t[KEY_ID]: t for t in tasks}
    predictions_path = os.path.abspath(predictions_path)
    validate_predictions(predictions_path, [t[KEY_ID] for t in tasks])

    predictions = get_instances(predictions_path)

    if len(predictions) == 0:
        logger.info("No predictions to evaluate")
        return

    # Remove predictions that have already been evaluated
    if skip_existing:
        # Skip logs that already exist
        predictions_filtered = []
        for p in predictions:
            all_exist = True
            if KEY_PREDICTIONS not in p:
                continue
            for setting in p[KEY_PREDICTIONS]:
                log_file_name = f"{p[KEY_ID]}.{p[KEY_MODEL]}.{setting}.eval.log"
                log_file = os.path.join(log_dir, log_file_name)
                if not os.path.exists(log_file):
                    all_exist = False
                    break
            if not all_exist:
                predictions_filtered.append(p)
        if len(predictions_filtered) == 0:
            logger.info(f"All predictions already exist, skipping")
            return
        else:
            logger.info(
                f"# of predictions to evaluate: {len(predictions_filtered)} "
                + f"({len(predictions) - len(predictions_filtered)} already evaluated)"
            )
            predictions = predictions_filtered
    else:
        logger.info(f"# of predictions to evaluate: {len(predictions)}")

    task_instances = []

    # Set the relevant data on task_instances
    for prediction in predictions:
        task = tasks_map[prediction[KEY_ID]]

        test_type = MAP_REPO_TO_TEST_FRAMEWORK[task["repo"]]
        test_directives = get_test_directives(task)
        test_cmd = f"{test_type} {' '.join(test_directives)}"

        task_instances.append(
            {
                "repo": task["repo"],
                "version": task["version"],
                "base_commit": task["base_commit"],
                KEY_ID: prediction[KEY_ID],
                KEY_INSTANCE_ID: prediction[KEY_INSTANCE_ID],
                KEY_MODEL: prediction[KEY_MODEL],
                KEY_PREDICTIONS: prediction[KEY_PREDICTIONS],
                "preds_context": task["preds_context"],
                "test_patch": task["test_patch"],
                "test_file": task["test_file"],
                "code_file": task["code_file"],
                "patch": task["patch"],
                "test_directives": test_directives,
                "test_cmd": test_cmd,
            }
        )

    task_instances = sorted(task_instances, key=lambda x: x[KEY_ID])

    sem = asyncio.Semaphore(num_processes if num_processes > 0 else len(task_instances))
    tasks = []
    for task_instance in task_instances:
        if task_instance[KEY_PREDICTIONS]:

            async def run_docker_throttled(*args, **kwargs):
                async with sem:
                    return await run_docker_evaluation(
                        *args,
                        **kwargs,
                        only_baseline=False,
                        skip_mutation=skip_mutation,
                    )

            for setting in task_instance[KEY_PREDICTIONS]:
                task = asyncio.create_task(
                    run_docker_throttled(
                        task_instance, namespace, log_dir, setting, timeout
                    )
                )
                tasks.append(task)
        else:
            logger.info(f"[{task_instance[KEY_ID]}] No prediction found.")

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions_path", type=str, help="Path to predictions file", required=True
    )
    parser.add_argument(
        "--log_dir", type=str, help="Path to log directory", required=True
    )
    parser.add_argument(
        "--swe_bench_tasks",
        type=str,
        help="Path to dataset file or HF datasets name",
        required=True,
    )
    parser.add_argument(
        "--namespace",
        type=str,
        help="Docker repository namespace",
        required=False,
        default="aorwall",
    )
    parser.add_argument(
        "--skip_existing", action="store_true", help="(Optional) Skip existing logs"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="(Optional) Timeout in seconds (default: 60)",
        default=60,
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        help="(Optional) Number of processes to run in parallel (-1 for unlimited)",
        default=-1,
    )
    parser.add_argument(
        "--skip_mutation", action="store_true", help="(Optional) Skip mutation"
    )
    args = parser.parse_args()
    asyncio.run(main(**vars(args)))
