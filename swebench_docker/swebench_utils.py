# Adapted from: https://github.com/aorwall/SWE-bench-docker/blob/main/swebench_docker/utils.py

import glob
import json
import os
import re
from enum import Enum
from typing import Callable, Dict, Optional, Tuple

from swebench_docker.constants import (
    INSTALL_FAIL,
    KEY_ID,
    NON_TEST_EXTS,
    RESET_FAILED,
    TESTS_CONFIG,
    TESTS_ERROR,
    TESTS_FAILED,
    TESTS_PASSED,
    TESTS_TIMEOUT,
    UNFILTERED_TESTS_FAILED,
    UNFILTERED_TESTS_PASSED,
    VALID_K,
)


# Test Status Enum
class TestStatus(Enum):
    FAILED = "FAILED"
    PASSED = "PASSED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"


def classify_error(test_str: str) -> str:
    if TESTS_PASSED in test_str:
        return "Success"
    elif "Error" in test_str:
        error_type = test_str.rsplit("Error")[0].split()[-1].split(".")[-1] + "Error"
        if "Test" in error_type:
            return "Other"
        return error_type
    elif "Val..." in test_str:
        return "ValueError"
    elif "Test script run timed out" in test_str:
        return "TimeoutError"
    elif "ass..." in test_str:
        return "AssertionError"
    elif "Ass" in test_str:
        return "AssertionError"
    elif "test" not in test_str:
        return "PostprocessingError"
    else:
        return "Other"


def get_logs_eval(log_fp: str) -> Dict[str, dict]:
    """
    Retrieve evaluation results for a task instance from its corresponding log file

    Args:
        log_fp (str): path to log file
    Returns:
        bool: whether the patch applied successfully
        dict: status map
    """
    repo = get_repo_from_lp(log_fp)

    results: Dict[str, dict] = {}
    with open(log_fp) as f:
        content = f.read()

        test_data = content.split(TESTS_CONFIG)[1:]

        for config in test_data:
            config_line = config.split("\n")[0].split()

            setting = config_line[0]

            test_passed = TESTS_PASSED in config

            unfiltered_tests_passed = UNFILTERED_TESTS_PASSED in config
            unfiltered_tests_compiled = (
                UNFILTERED_TESTS_FAILED in config or unfiltered_tests_passed
            )

            test_compiled = TESTS_FAILED in config or test_passed

            if setting not in results:
                results[setting] = {
                    "tests_passed": [],
                    "tests_compiled": [],
                    "coverage": [],
                    "test_time": [],
                    "test_error": [],
                }
                if setting == "full":
                    results[setting]["unfiltered_tests_passed"] = []
                    results[setting]["unfiltered_tests_compiled"] = []
                    results[setting]["mutation_score"] = []
                    results[setting]["mutation_uncertainty"] = []
                    results[setting]["mutation_num"] = []

            if "CoverageLOG" in config:
                coverage = (
                    float(config.split("CoverageLOG: ")[1].split("%")[0])
                    if test_passed
                    else -1
                )
            else:
                coverage = -1

            if "TestsTime: " in config:
                test_time = float(config.split(f"TestsTime: ")[1].split("\n")[0])
            else:
                test_time = -1

            if setting == "full":
                if "MutationLOG" in config:
                    mutation_score = (
                        float(config.split("MutationLOG: ")[1].split("%")[0])
                        if test_passed
                        else -1
                    )
                else:
                    mutation_score = -1

                if "MutationUncertainty" in config:
                    mutation_uncertainty = (
                        float(config.split("MutationUncertainty: ")[1].split("\n")[0])
                        if test_passed
                        else -1
                    )
                else:
                    mutation_uncertainty = -1

                if "MutationNum" in config:
                    mutation_num = (
                        float(config.split("MutationNum: ")[1].split("\n")[0])
                        if test_passed
                        else -1
                    )
                else:
                    mutation_num = -1

                results[setting]["mutation_score"].append(mutation_score)
                results[setting]["mutation_uncertainty"].append(mutation_uncertainty)
                results[setting]["mutation_num"].append(mutation_num)
                results[setting]["unfiltered_tests_passed"].append(
                    unfiltered_tests_passed
                )
                results[setting]["unfiltered_tests_compiled"].append(
                    unfiltered_tests_compiled
                )

            results[setting]["tests_passed"].append(test_passed)
            results[setting]["tests_compiled"].append(test_compiled)
            results[setting]["coverage"].append(coverage)
            results[setting]["test_time"].append(test_time)
            results[setting]["test_error"].append(classify_error(config))

    return results


def get_eval_reports_for_logs(
    eval_logs: list,
    swe_bench_instances: dict,
    callback: Optional[Callable[[str], bool]] = None,
    verbose: bool = False,
    raw_only: bool = False,
    is_baseline: bool = False,
) -> Dict[str, dict]:
    """
    Wrapper for getting eval report for a list of evaluation log paths.

    Args:
        eval_logs (list): list of paths to evaluation logs
        swe_bench_instances (str): path to eval task instances (swe-bench.json)
        callback (callable): callback function for evaluation logs
        verbose (bool): whether to print verbose output
    Returns:
        reports_patch_success (dict): dict of eval reports for patch apply successes
        reports_patch_failure (dict): dict of eval reports for patch apply failures
    """
    report_tests = {}

    from tqdm import tqdm

    for eval_log in tqdm(eval_logs):
        # Remove task instances that do not satisfy callback
        if callback is not None and not callback(eval_log):
            continue
        try:
            # Get eval logs
            eval_sm = get_logs_eval(eval_log)
            instance_id = eval_log.split("/")[-1].split(".")[0]

            if raw_only:
                eval_sm["baseline_covs"] = swe_bench_instances[instance_id][
                    "baseline_covs"
                ]

            report = (
                get_eval_report(eval_sm, swe_bench_instances, instance_id, is_baseline)
                if not raw_only
                else eval_sm
            )
            report_tests[get_file_name_from_lp(eval_log)] = report
        except Exception as e:
            print(e)
            raise e
            print(f"Skipping instance {get_file_name_from_lp(eval_log)}")

    report_final = {}

    # Merge settings
    for eval_log in eval_logs:
        instance_id = eval_log.split("/")[-1].split(".")[0]
        if instance_id not in report_final:
            report_final[instance_id] = report_tests[get_file_name_from_lp(eval_log)]
        else:
            report_final[instance_id].update(
                report_tests[get_file_name_from_lp(eval_log)]
            )

    return report_final


def add_execution_metric(eval_sm, final_results, setting, baseline_info, metric_name):
    metric_ds = eval_sm[setting][metric_name]
    metric_non_negative_1 = [
        metric_ds[i] for i in range(len(metric_ds)) if metric_ds[i] >= 0
    ]

    if "full" in setting:
        final_results[f"{setting}_av_{metric_name}"] = (
            sum(metric_non_negative_1) / len(metric_non_negative_1)
            if len(metric_non_negative_1) > 0
            else 0
        )
        final_results[f"{setting}_av_pass_{metric_name}"] = (
            sum(metric_non_negative_1) / len(metric_non_negative_1)
            if len(metric_non_negative_1) > 0
            else -1
        )
    else:
        EXECUTION_MAPPING = {"last": "last_minus_one", "extra": "last"}

        if setting != "first":
            metric_baseline_ds = baseline_info[EXECUTION_MAPPING[setting]]
            metric_non_negative_baseline = [
                metric_baseline_ds
                for i in range(len(metric_ds))
                if metric_baseline_ds >= 0 and metric_ds[i] >= 0
            ]
            metric_non_negative_pred = [
                metric_ds[i]
                for i in range(len(metric_ds))
                if metric_baseline_ds >= 0 and metric_ds[i] >= 0
            ]
        else:
            metric_non_negative_baseline = [0 for i in range(len(metric_ds))]
            metric_non_negative_pred = [
                metric_ds[i] for i in range(len(metric_ds)) if metric_ds[i] >= 0
            ]

        if len(metric_non_negative_pred) == 0:
            final_results[f"{setting}_av_{metric_name}_imp_baseline"] = 0
            final_results[f"{setting}_av_pass_{metric_name}_imp_baseline"] = -1
        else:
            metric_non_negative_baseline_av = sum(metric_non_negative_baseline) / len(
                metric_non_negative_baseline
            )
            metric_non_negative_pred_av = sum(metric_non_negative_pred) / len(
                metric_non_negative_pred
            )
            if metric_non_negative_pred_av - metric_non_negative_baseline_av < 0:
                final_results[f"{setting}_av_{metric_name}_imp_baseline"] = 0
                final_results[f"{setting}_av_pass_{metric_name}_imp_baseline"] = -1
                print(
                    f"{setting}_av_{metric_name}_imp_baseline: {metric_non_negative_pred_av - metric_non_negative_baseline_av}"
                )
            else:
                final_results[f"{setting}_av_{metric_name}_imp_baseline"] = (
                    metric_non_negative_pred_av - metric_non_negative_baseline_av
                )
                final_results[f"{setting}_av_pass_{metric_name}_imp_baseline"] = (
                    metric_non_negative_pred_av - metric_non_negative_baseline_av
                )


def get_eval_report(
    eval_sm: dict,
    swe_bench_instances: dict,
    instance_id: str,
    is_baseline: bool = False,
) -> dict:
    """
    Create a report based on failure/pass change from gold results to eval results.

    Args:
        eval_sm (dict): evaluation status map
    Returns:
        report (dict): report of metrics
    """
    # Calculate resolution metrics

    final_results: Dict[str, float] = {}

    for setting in eval_sm:
        tests_passed = eval_sm[setting]["tests_passed"]
        unfiltered_tests_passed = (
            eval_sm[setting]["unfiltered_tests_passed"]
            if "unfiltered_tests_passed" in eval_sm[setting]
            else []
        )

        if not is_baseline:
            baseline_cov_info = swe_bench_instances[instance_id]["baseline_covs"]

            add_execution_metric(
                eval_sm, final_results, setting, baseline_cov_info, "coverage"
            )
            if setting == "full":
                add_execution_metric(
                    eval_sm, final_results, setting, {}, "mutation_score"
                )
        else:
            final_results[f"{setting}_av_coverage"] = eval_sm[setting]["coverage"][0]

        for k in VALID_K:
            if len(tests_passed) >= k:
                final_results[f"{setting}_pass_at_{k}"] = any(tests_passed[:k])
            if len(tests_passed) >= k:
                final_results[f"{setting}_avg_pass_at_{k}"] = sum(
                    tests_passed[:k]
                ) / len(tests_passed[:k])
            if len(unfiltered_tests_passed) >= k:
                final_results[f"{setting}_unfiltered_pass_at_{k}"] = any(
                    unfiltered_tests_passed[:k]
                )

        for metric in eval_sm[setting]:
            if metric not in [
                "tests_passed",
                "tests_compiled",
                "unfiltered_tests_passed",
                "unfiltered_tests_compiled",
                "coverage",
                "mutation_score",
                "test_error",
            ]:
                met_non_negative = [
                    eval_sm[setting][metric][i]
                    for i in range(len(eval_sm[setting][metric]))
                    if eval_sm[setting][metric][i] >= 0
                ]
                if len(met_non_negative) == 0:
                    final_results[f"{setting}_av_{metric}"] = -1
                else:
                    final_results[f"{setting}_av_{metric}"] = sum(
                        eval_sm[setting][metric]
                    ) / len(eval_sm[setting][metric])
    return final_results


get_file_name_from_lp = lambda x: x.rsplit("/", 1)[-1]


get_id_from_lp = lambda x: get_file_name_from_lp(x).split(".")[0]


get_repo_from_lp = lambda x: get_id_from_lp(x).rsplit("-", 1)[0].replace("__", "/")


test_passed = lambda case, sm: case in sm and sm[case] == TestStatus.PASSED.value

test_failed = lambda case, sm: case not in sm or any(
    [sm[case] == status for status in [TestStatus.FAILED.value, TestStatus.ERROR.value]]
)


def get_eval_reports_for_dir(
    eval_dir: str,
    swe_bench_instances: dict,
    model_name,
    callback: Optional[Callable[[str], bool]] = None,
    verbose=False,
    raw_only=False,
    is_baseline=False,
) -> dict:
    """
    Wrapper for getting eval report for a directory of evaluation logs.

    Args:
        eval_dir (str): path to directory of evaluation logs
        (See get_eval_reports_for_logs for other args)
    """
    if not os.path.exists(eval_dir):
        raise ValueError(f"Path {eval_dir} does not exist")
    logs_list = [x for x in glob.glob(os.path.join(eval_dir, f"*{model_name}*.log"))]
    return get_eval_reports_for_logs(
        logs_list, swe_bench_instances, callback, verbose, raw_only, is_baseline
    )


### MARK - Model Evaluation Summary


def get_model_eval_summary(
    predicts_path: str,
    eval_dir: str,
    swe_bench_instances: dict,
    model_name: str,
    repo: Optional[str] = None,
    is_baseline: bool = False,
) -> dict:
    """
    Generate a summary of model evaluation results.

    Args:
        predicts_path (str): path to predictions file
        eval_dir (str): path to directory of evaluation logs
        swe_bench_instances (str): path to eval references (swe-bench-eval-refs.json)
        repo (str): if given, repo name to limit evaluation to
    """
    # Load Predictions
    preds = []
    if len(predicts_path) > 0:
        with open(predicts_path) as f:
            for line in f.readlines():
                preds.append(json.loads(line))

        # Filter by repo if provided
        criteria_eval_sm = None
        if repo is not None:
            criteria_pred = lambda pred: repo in pred[KEY_ID]
            criteria_eval_sm = lambda eval_log: repo in eval_log
            preds = [x for x in preds if criteria_pred(x)]

        # Get reports
        report_net = get_eval_reports_for_dir(
            eval_dir,
            swe_bench_instances,
            is_baseline=is_baseline,
            callback=criteria_eval_sm,
            verbose=False,
            model_name=model_name,
        )
    else:
        report_net = get_eval_reports_for_dir(
            eval_dir,
            swe_bench_instances,
            is_baseline=is_baseline,
            verbose=False,
            model_name=model_name,
        )

    # Print reports for different granularities of patch success/failure
    summary = {
        "repo": repo if repo is not None else "all",
        "total_predictions": len(preds),
    }

    format_dec = lambda x: round(x * 100, 2)

    total_metrics: Dict[str, list] = {}
    for fn in report_net:
        for key in report_net[fn]:
            if key not in total_metrics:
                total_metrics[key] = []
            total_metrics[key].append(report_net[fn][key])
    for met in total_metrics:
        cleansed_metrics = [e for e in total_metrics[met] if e != -1]
        if len(cleansed_metrics) == 0:
            summary[met] = -1
        else:
            summary[met] = sum(cleansed_metrics) / len(cleansed_metrics)

    return summary


def get_model_report(
    model: str,
    predictions_path: str,
    log_dir: str,
    verbose: bool = False,
) -> dict:
    """
    Generate a report of model evaluation results from predictions, task instances,
    and evaluation logs.

    Args:
        model (str): model name
        predictions_path (str): path to predictions file
        log_dir (str): path to directory of evaluation logs
        verbose (bool): show tqdm to track progress
    Returns:
        report_map (dict): map of repo to report
    """
    from tqdm import tqdm

    # Get predictions
    predictions = []
    if predictions_path.endswith("jsonl"):
        with open(predictions_path) as f:
            for line in f.readlines():
                predictions.append(json.loads(line))
    elif predictions_path.endswith("json"):
        predictions = json.load(open(predictions_path))
    else:
        raise ValueError("Predictions file must be in json or jsonl format")
    report_map: Dict[str, list] = {}

    # Iterate through predictions
    report_map = {
        "no_generation": [],
        "generated": [],
        "with_logs": [],
        "install_fail": [],
        "reset_failed": [],
        "test_errored": [],
        "test_timeout": [],
        "mutation_timeout": [],
    }
    for p in tqdm(predictions, desc="Processing predictions new", disable=not verbose):
        report_map["generated"].append(p[KEY_ID])

        # Get log file
        log_path = os.path.join(log_dir, f"{p[KEY_ID]}.{model}.eval.log")
        if not os.path.exists(log_path):
            continue
        report_map["with_logs"].append(p[KEY_ID])
        log_content = open(log_path).read()

        # Check if there is a reset failure
        if RESET_FAILED in log_content:
            report_map["reset_failed"].append(p[KEY_ID])
            continue

        # Get evaluation logs
        eval_sm = get_logs_eval(log_path)

        # Check if any tests errored or timed out
        for status in [
            ("test_errored", TESTS_ERROR),
            ("test_timeout", TESTS_TIMEOUT),
            ("mutation_timeout", "MutationTimeout"),
        ]:
            if status[1] in log_content:
                report_map[status[0]].append(p[KEY_ID])
                continue

    return report_map


def get_instances(instance_path: str) -> list:
    """
    Get task instances from given path

    Args:
        instance_path (str): Path to task instances
    Returns:
        task_instances (list): List of task instances
    """
    if any([instance_path.endswith(x) for x in [".jsonl", ".jsonl.all"]]):
        task_instances = list()
        with open(instance_path) as f:
            for line in f.readlines():
                task_instances.append(json.loads(line))
        return task_instances

    with open(instance_path) as f:
        task_instances = json.load(f)
    return task_instances


def get_test_directives(instance: dict, keep_as_files: bool = False) -> list:
    """
    Get test directives from the test_patch of a task instance

    Args:
        instance (dict): task instance
        keep_as_files (bool): if true, we will return a list of files not a list of test directives
    Returns:
        directives (list): List of test directives
    """
    # For seq2seq code repos, testing command is fixed
    if instance["repo"] == "swe-bench/humaneval":
        return ["test.py"]

    # Get test directives from test patch and remove non-test files
    diff_pat = r"diff --git a/.* b/(.*)"
    test_patch = instance["test_patch"]
    directives = re.findall(diff_pat, test_patch)
    directives = [
        d for d in directives if not any(d.endswith(ext) for ext in NON_TEST_EXTS)
    ]

    if keep_as_files:
        return directives

    # For Django tests, remove extension + "tests/" prefix and convert slashes to dots (module referencing)
    if instance["repo"] == "django/django":
        directives_transformed = []
        for d in directives:
            d = d[: -len(".py")] if d.endswith(".py") else d
            d = d[len("tests/") :] if d.startswith("tests/") else d
            d = d.replace("/", ".")
            directives_transformed.append(d)
        directives = directives_transformed

    return directives
