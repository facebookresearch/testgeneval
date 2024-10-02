# Adapted from: https://github.com/aorwall/SWE-bench-docker/blob/main/swebench_docker/evaluate_instance.py

import base64
import json
import logging
import os
import re
import subprocess
import sys
from typing import Optional, Tuple

from swebench_docker.constants import (
    KEY_BASELINES,
    KEY_MODEL,
    KEY_PREDICTIONS,
    KEY_TEST_FILE_PATH,
    MAP_REPO_TO_TEST_FRAMEWORK,
    SETTING_PROMPT_MAP,
    TESTS_CONFIG,
    TESTS_FAILED,
    UNFILTERED_TESTS_FAILED,
    UNFILTERED_TESTS_PASSED,
    PatchType,
)
from swebench_docker.context_manager import TaskEnvContextManager
from swebench_docker.swebench_utils import get_test_directives

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger("evaluate_instance")


def indent_text(text, indent_level):
    return "\n".join(
        " " * indent_level + line if line.strip() else line for line in text.split("\n")
    )


def extract_preamble_classes_and_functions(code, tcm):
    class_pattern = re.compile(
        r"(^(\s*@[\w\.\(\)\', ]+\s*)*^\s*class ([\w]+)\([^)]+\):)", re.MULTILINE
    )
    # Capture methods with or without decorators
    test_method_pattern = re.compile(
        r"(^(\s*@.*\s*)*^\s*def\s+test\w+\(.*\):)", re.MULTILINE
    )

    # Capture functions with or without decorators
    test_function_pattern = re.compile(
        r"(^(\s*@.*\s*)*^\s*def\s+test\w+\(.*\):)", re.MULTILINE
    )

    preamble = ""
    classes = []
    test_functions = []

    current_position = 0

    def extract_class_body(code: str, start_index: int) -> Tuple[str, int]:
        """
        Extracts the body of a class from the given code starting from the specified index.
        Returns the class body and the end index of the class body.
        """
        if not code or start_index < 0 or start_index >= len(code):
            raise ValueError("Invalid code or start index")

        # Split the code into lines
        lines = code[start_index:].split("\n")
        class_body_lines = []

        # Find the starting indentation level of the class definition
        class_start_line = lines[0]
        start_indent = len(class_start_line) - len(class_start_line.lstrip())

        inside_multiline_comment = False
        end_index = start_index
        for i, line in enumerate(lines[1:], start=1):
            stripped_line = line.strip()
            current_indent = len(line) - len(line.lstrip())

            # Handle multiline comments or docstrings
            if stripped_line.startswith('"""') or stripped_line.startswith("'''"):
                if inside_multiline_comment:
                    inside_multiline_comment = False
                else:
                    inside_multiline_comment = True

            if not inside_multiline_comment:
                # Stop when we reach a line with less indentation than the class definition
                if current_indent <= start_indent and stripped_line:
                    break

            # Add lines that are part of the class body
            class_body_lines.append(line)
            # Update the end index to the current line end
            end_index = start_index + len("\n".join(lines[: i + 1])) + 1

        return code[start_index:end_index], end_index

    while current_position < len(code):
        class_match = class_pattern.search(code, current_position)
        method_match = test_function_pattern.search(code, current_position)

        if class_match and (
            not method_match or class_match.start() < method_match.start()
        ):
            class_name = class_match.group(0)
            class_body, end_idx = extract_class_body(code, class_match.end())
            current_position = end_idx

            methods = []
            class_prefix = class_name
            set_prefix = False
            for method_match in test_method_pattern.finditer(class_body):
                method_name = method_match.group()
                method_start = method_match.start()
                if not set_prefix:
                    class_prefix = class_name + class_body[:method_start]
                    set_prefix = True
                next_method = test_method_pattern.search(
                    class_body, method_start + len(method_name)
                )
                method_body = (
                    class_body[method_start : next_method.start()]
                    if next_method
                    else class_body[method_start:]
                )
                methods.append((method_name, method_body))

            if methods:
                classes.append((class_prefix, methods, class_match.start()))
            else:
                preamble += class_name + class_body

        elif method_match:
            function_name = method_match.group(0)
            start_idx = method_match.start()
            next_function = test_function_pattern.search(
                code, start_idx + len(function_name)
            )
            function_body = (
                code[start_idx : next_function.start()]
                if next_function
                else code[start_idx:]
            )
            test_functions.append((function_body, start_idx))
            current_position = method_match.end()

        else:
            break

    if classes and test_functions:
        preamble = code[: min(classes[0][2], test_functions[0][1])]
    else:
        preamble = (
            code[: classes[0][2]]
            if classes
            else code[: test_functions[0][1]] if test_functions else code
        )

    return preamble.strip(), classes, test_functions


def postprocess_tests(
    task_instance, preamble, class_name, methods, successful_tests, tcm
):
    repo = task_instance["repo"]
    django_repo = repo == "django/django"

    def needs_django_harness(preamble):
        no_django_test = "TestCase" not in preamble
        no_unittest = "unittest" not in preamble
        no_simple_test_case = "SimpleTestCase" not in preamble
        return no_django_test and no_unittest and no_simple_test_case

    if django_repo and needs_django_harness(preamble):
        preamble = "from django.test import SimpleTestCase\n" + preamble
        preamble += "\n\nclass TestsHarness(SimpleTestCase):\n"
        added_class = True
    else:
        added_class = False

    for method_name, test_case in methods:
        if django_repo and added_class:
            if "(self):" not in test_case:
                test_case = test_case.replace("():", "(self):", 1)

        class_content = f"{class_name}\n{test_case}\n"

        with open(task_instance[KEY_TEST_FILE_PATH], "w") as f:
            test_content = preamble + "\n\n" + class_content
            f.write(test_content)

        _, success = tcm.run_tests_task(
            task_instance, log_data=False, skip_mutation=True
        )
        if success:
            successful_tests.append((class_name, method_name, test_case))


def postprocess_functions(
    task_instance, preamble, test_functions, successful_tests, tcm
):
    repo = task_instance["repo"]
    django_repo = repo == "django/django"

    def needs_django_harness(preamble):
        no_django_test = "TestCase" not in preamble
        no_unittest = "unittest" not in preamble
        no_simple_test_case = "SimpleTestCase" not in preamble
        return no_django_test and no_unittest and no_simple_test_case

    added_class = False
    if django_repo and needs_django_harness(preamble):
        preamble = "from django.test import SimpleTestCase\n" + preamble
        class_wrapper_start = "\n\nclass TestsHarness(SimpleTestCase):\n"
        preamble += class_wrapper_start
        added_class = True

    class_content = ""
    for test_function, start in test_functions:
        with open(task_instance[KEY_TEST_FILE_PATH], "w") as f:
            if django_repo and added_class:
                if "(self):" not in test_function:
                    test_function = test_function.replace("():", "(self):", 1)
                test_content = preamble + "\n\n" + indent_text(test_function, 4)
            else:
                test_content = preamble + "\n\n" + test_function
            f.write(test_content)

        _, success = tcm.run_tests_task(
            task_instance, log_data=False, skip_mutation=True
        )
        if success:
            if django_repo and added_class:
                class_content += indent_text(test_function, 4) + "\n"
            else:
                successful_tests.append((None, test_function))

    if django_repo and class_content:
        successful_tests.append((None, class_wrapper_start + class_content))


def full_processing(prompt_list, tcm, task_instance, skip_mutation):
    for prompt in prompt_list:
        preamble, classes, test_functions = extract_preamble_classes_and_functions(
            prompt, tcm
        )
        successful_tests = []

        if classes:
            for class_name, methods, start in classes:
                postprocess_tests(
                    task_instance, preamble, class_name, methods, successful_tests, tcm
                )

        if test_functions:
            postprocess_functions(
                task_instance, preamble, test_functions, successful_tests, tcm
            )

        tcm.log.write(f"{TESTS_CONFIG}full pred\n")
        if len(successful_tests) > 0:
            success_tests = []
            class_definitions = {}
            for item in successful_tests:
                if item[0]:  # It's a class method
                    class_def, method_name, method_content = item
                    if class_def not in class_definitions:
                        class_definitions[class_def] = [method_content]
                    else:
                        class_definitions[class_def].append(method_content)
                else:  # It's a standalone function
                    success_tests.append(item[1])

            for class_def, methods in class_definitions.items():
                class_content = f"{class_def}\n" + "\n".join(methods)
                success_tests.append(class_content)

            success_tests_str = "\n\n".join(success_tests)

            with open(task_instance[KEY_TEST_FILE_PATH], "w") as f:
                f.write(preamble + "\n" + success_tests_str)

            _, success = tcm.run_tests_task(task_instance, skip_mutation=skip_mutation)

            total_tests = len(test_functions) + sum(
                len(methods) for _, methods, _ in classes
            )
            if success and len(successful_tests) == total_tests:
                tcm.log.write(UNFILTERED_TESTS_PASSED)
            else:
                tcm.log.write(UNFILTERED_TESTS_FAILED)
        else:
            tcm.log.write("TestsTime: 0.0")
            tcm.log.write(TESTS_FAILED)
            tcm.log.write(UNFILTERED_TESTS_FAILED)


def completion_processing(
    prompt_list, tcm, setting, task_instance, only_baseline, skip_mutation
):
    i = 0
    for prompt_ind in range(len(prompt_list)):
        prompt = prompt_list[prompt_ind]
        skip_prompt = False
        tcm.log.write(
            f"{TESTS_CONFIG}{setting} {'baseline' if only_baseline else 'pred'}\n"
        )
        if only_baseline:
            with open(task_instance[KEY_TEST_FILE_PATH], "w") as f:
                f.write(prompt)
        else:
            file_content = task_instance["preds_context"][SETTING_PROMPT_MAP[setting]]

            full_prompt = file_content + "\n" + prompt

            if (
                "assert" not in prompt
                and ".raises" not in prompt
                and "Error" not in prompt
            ) or "def" not in prompt:
                skip_prompt = True
            else:
                with open(task_instance[KEY_TEST_FILE_PATH], "w") as f:
                    f.write(full_prompt)

        if not skip_prompt:
            tcm.run_tests_task(task_instance, skip_mutation=True)
        else:
            tcm.log.write(TESTS_FAILED)


def main(
    task_instance: dict,
    testbed_name: str,
    setting: str,
    repo_dir: str,
    log_dir: str,
    timeout: Optional[int],
    image_type: str = "conda",
    only_baseline: bool = False,
    skip_mutation: bool = False,
):
    logger.info(
        "Instance ID: "
        + task_instance["instance_id"]
        + "\nID: "
        + task_instance["id"]
        + "\nTestbed: "
        + testbed_name
        + "\nLog dir: "
        + log_dir
        + "\nSetting: "
        + setting
    )
    logger.info(f"Only Baseline: {only_baseline}")

    if only_baseline:
        task_instance[KEY_MODEL] = "baseline"
        test_type = MAP_REPO_TO_TEST_FRAMEWORK[task_instance["repo"]]
        test_directives = get_test_directives(task_instance)
        test_cmd = f"{test_type} {' '.join(test_directives)}"
        task_instance["test_directives"] = test_directives
        task_instance["test_cmd"] = test_cmd

    with TaskEnvContextManager(
        task_instance,
        setting,
        testbed_name,
        repo_dir,
        log_dir,
        timeout=timeout,
        mutation_timeout=3600,
        image_type=image_type,
    ) as tcm:
        test_patch = task_instance["test_patch"]
        if not tcm.apply_patch(
            task_instance["patch"], patch_type=PatchType.PATCH_GOLD.value
        ) or (
            test_patch
            and not tcm.apply_patch(test_patch, patch_type=PatchType.PATCH_TEST.value)
        ):
            logger.warning("Evaluation failed")
            sys.exit(1)

        # Make baselines a list so the loop below works
        prompt_list = (
            [task_instance[KEY_BASELINES][setting]]
            if only_baseline
            else task_instance[KEY_PREDICTIONS][setting]
        )
        if setting == "full":
            full_processing(prompt_list, tcm, task_instance, skip_mutation)
        else:
            completion_processing(
                prompt_list, tcm, setting, task_instance, only_baseline, skip_mutation
            )

        logger.info("Evaluation succeeded")


if __name__ == "__main__":
    TASK_INSTANCE_JSON = "/home/swe-bench/task_instance.json"
    if os.path.exists(TASK_INSTANCE_JSON):
        with open(TASK_INSTANCE_JSON, "r") as f:
            task_instance = json.load(f)
    else:
        instance_encoded = os.getenv("INSTANCE")
        if instance_encoded is None:
            raise ValueError("INSTANCE environment variable is not set")
        task_instance = json.loads(base64.b64decode(instance_encoded).decode("utf-8"))
    log_dir = os.getenv("LOG_DIR")
    if log_dir is None:
        raise ValueError("LOG_DIR environment variable is not set")

    testbed_name = os.getenv("TESTBED_NAME")
    if testbed_name is None:
        raise ValueError("TESTBED_NAME environment variable is not set")

    repo_dir = os.getenv("REPO_DIR") if os.getenv("REPO_DIR") else os.getenv("TESTBED")
    if repo_dir is None:
        raise ValueError("REPO_DIR environment variable is not set")

    timeout = os.getenv("TIMEOUT")
    int_timeout: Optional[int] = None
    if timeout is not None:
        try:
            int_timeout = int(timeout)
        except ValueError:
            raise ValueError("TIMEOUT environment variable must be an integer or None")

    setting = os.getenv("SETTING")
    if setting is None:
        raise ValueError("SETTING environment variable is not set")

    main(
        task_instance=task_instance,
        testbed_name=testbed_name,
        repo_dir=repo_dir,
        log_dir=log_dir,
        timeout=int_timeout,
        setting=setting,
        image_type=os.getenv("IMAGE_TYPE", "conda"),
        only_baseline=os.getenv("ONLY_BASELINE") == "True",
        skip_mutation=os.getenv("SKIP_MUTATION") == "True",
    )
