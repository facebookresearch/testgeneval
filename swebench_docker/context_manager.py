# Adapted from: https://github.com/aorwall/SWE-bench-docker/blob/main/swebench_docker/context_manager.py

import configparser
import json
import logging
import os
import shutil
import subprocess
import time
from logging import DEBUG, ERROR, INFO, Logger
from traceback import format_exc
from typing import Dict, Optional

from swebench_docker.constants import (
    APPLY_PATCH_FAIL,
    APPLY_PATCH_PASS,
    INSTALL_FAIL,
    KEY_ID,
    KEY_INSTANCE_ID,
    KEY_MODEL,
    MAP_VERSION_TO_INSTALL,
    MUTATION_TEMPLATE,
    TESTS_ERROR,
    TESTS_FAILED,
    TESTS_PASSED,
    TESTS_TIMEOUT,
    PatchType,
)
from swebench_docker.swebench_utils import get_test_directives

logger_taskenv = logging.getLogger("taskenv")


class LogWrapper:
    def __init__(
        self,
        log_file: str,
        logger: Optional[Logger] = None,
        prefix: Optional[str] = None,
    ):
        self.log_file = log_file
        self.logger = logger
        self.prefix = prefix

    def write(self, message: str, mode: str = "a", level: int = INFO):
        with open(self.log_file, mode) as f:
            log = (
                f"{self.prefix} {message} \n"
                if self.prefix is not None
                else f"{message} \n"
            )
            f.write(log)
        if self.logger is not None:
            self.logger.log(level, message)


class ExecWrapper:
    def __init__(
        self,
        subprocess_args: Optional[Dict] = None,
        logger: Optional[LogWrapper] = None,
    ):
        self.logger = logger
        if subprocess_args is None:
            self.subprocess_args = {}
        else:
            self.subprocess_args = subprocess_args

    def __call__(self, cmd, raise_error=True, **kwargs):
        try:
            if isinstance(cmd, list):
                self.logger.write(f"Command: {' '.join(cmd)}", level=DEBUG)
            else:
                self.logger.write(f"Command: {cmd}", level=DEBUG)
            combined_args = {**self.subprocess_args, **kwargs}
            self.logger.write(
                f"Subprocess args: {json.dumps(combined_args)}", level=DEBUG
            )
            output = subprocess.run(cmd, **combined_args)
            self.logger.write(f"Std. Output:\n{output.stdout}", level=DEBUG)
            if output.stderr:
                self.logger.write(f"Std. Error:\n{output.stderr}", level=DEBUG)
            self.logger.write(f"Return Code: {output.returncode}", level=DEBUG)
            return output
        except subprocess.CalledProcessError as e:
            if raise_error and self.logger is not None:
                self.logger.write(f"Error: {e}", level=ERROR)
                self.logger.write(f"Error stdout: {e.stdout}", level=ERROR)
                if e.stderr:
                    self.logger.write(f"Error stderr: {e.stderr}", level=ERROR)
                self.logger.write(f"Error traceback: {format_exc()}", level=ERROR)
                raise e


class TaskEnvContextManager:

    def __init__(
        self,
        task_instance: dict,
        setting: str,
        testbed_name: str,
        repo_dir: str,
        log_dir: str,
        timeout: Optional[int] = None,
        mutation_timeout: Optional[int] = None,
        is_eval: bool = True,
        image_type: str = "conda",
    ):
        self.instance_id = task_instance[KEY_INSTANCE_ID]
        self.id = task_instance[KEY_ID]
        self.instance = task_instance
        self.testbed_name = testbed_name
        self.repo_dir = repo_dir
        self.cwd = os.getcwd()
        self.is_eval = is_eval
        self.image_type = image_type
        self.mutation_timeout = mutation_timeout

        model = task_instance[KEY_MODEL]
        if image_type == "conda":
            self.cmd_conda_run = f"conda run -n {testbed_name} "
        else:
            self.cmd_conda_run = ""

        self.timeout = timeout

        log_file_name = f"{self.id}.{model}.{setting}.eval.log"

        self.log_file = os.path.join(log_dir, log_file_name)
        self.log = LogWrapper(
            self.log_file,
            logger=logger_taskenv,
            prefix=f"[{testbed_name}] [{self.instance_id}]",
        )

        self.exec = ExecWrapper(
            subprocess_args={
                "cwd": self.repo_dir,
                "check": True,
                "shell": False,
                # "capture_output": False,
                "universal_newlines": True,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
            },
            logger=self.log,
        )

    def add_coverage_tox(self, config_file):
        # Create a ConfigParser object
        config = configparser.ConfigParser()

        # Read the existing tox.ini file
        config.read(config_file)

        # Check if the 'testenv' section exists
        if "testenv" in config:
            # Get the existing commands or set default if not found
            commands = config.get("testenv", "commands")

            self.log.write("OLD COMMANDS: " + commands)
            # Modify the command to include coverage
            if "coverage run" not in commands:
                # Assuming the command is something like `python -m pytest {posargs}`
                # We need to replace it with `coverage run -m pytest {posargs}`
                modified_commands = []
                for command in commands.split("\n"):
                    if "--cov" not in command and "pytest" in command:
                        # Replace python with coverage run

                        command = command.replace("pytest", "pytest --cov sphinx")

                    modified_commands.append(command)
                modified_commands.append("coverage json -o coverage.json")

                commands = "\n".join(modified_commands)

            self.log.write("NEW COMMANDS: " + commands)
            # Set the modified commands back to the config
            config.set("testenv", "commands", commands)

            # Write the changes back to the tox.ini file
            with open(config_file, "w") as configfile:
                config.write(configfile)
            self.log.write(f"Coverage added to {config_file}\n")
            self.log.write(commands)

    def __enter__(self):
        """
        Enter task environment, set up log file
        """
        os.chdir(self.repo_dir)
        enter_msg = (
            f"Task Metadata:"
            f"\n\t- Instance ID: {self.instance[KEY_INSTANCE_ID]}"
            f"\n\t- Testbed: {self.testbed_name}"
        )
        if self.is_eval:
            enter_msg += f"\n\t- Evaluation Model: {self.instance[KEY_MODEL]}"

        output = self.exec("python --version".split())
        enter_msg += f"\n\t- Python version: {output.stdout}"

        self.log.write(enter_msg, mode="w")

        self.exec(
            f"git config --global --add safe.directory {self.repo_dir}".split(" ")
        )
        self.exec(
            f"git -c advice.detachedHead=false checkout {self.instance['base_commit']}".split(
                " "
            )
        )

        specifications = MAP_VERSION_TO_INSTALL[self.instance["repo"]][
            self.instance["version"]
        ]
        if "pre_test" in specifications:
            for cmd_pre_install in specifications["pre_test"]:
                self.log.write(f"Running pre-test command: {cmd_pre_install}")
                cmd_pre_install = f"{self.cmd_conda_run} {cmd_pre_install}"

                out_pre_install = self.exec(
                    cmd_pre_install, timeout=self.timeout, shell=True
                )
                with open(self.log_file, "a") as f:
                    f.write(f"Pre-installation Command: {cmd_pre_install}\n")
                    f.write(f"Std. Output: {out_pre_install.stdout}\n")
                    if out_pre_install.stderr:
                        f.write(f"Std. Error: {out_pre_install.stderr}\n")
                if out_pre_install.returncode != 0:
                    self.log.write(f"Pre-install setup failed", level=ERROR)
                    with open(self.log_file, "a") as f:
                        f.write(f"\n{INSTALL_FAIL}\n")
                    return False

        return self

    def apply_patch(
        self, patch: str, patch_type: PatchType, revert: bool = False
    ) -> bool:
        """
        Apply patch to task environment

        Args:
            patch (str): Plaintext of patch to apply
            patch_type (str): Type of patch (e.g. "eval", "test")
        Returns:
            bool: True if patch applied successfully, False otherwise
        """
        init_diff_patch_path = os.path.join(
            os.path.dirname(self.repo_dir.rstrip("/")),
            f"temp_{self.instance_id}_{patch_type}_init.patch",
        )
        self.exec(f"git diff > {init_diff_patch_path}", shell=True)

        # If patch is `None`, indicate in log and skip
        if patch is None:
            self.log.write(f"Patch is `None` ({patch_type})")
            with open(self.log_file, "a") as f:
                f.write(f"{APPLY_PATCH_FAIL}; Prediction patch is `None`")
            return False

        # Write patch to temporary patch file in parent directory
        patch_path = os.path.join(
            os.path.dirname(self.repo_dir.rstrip("/")),
            f"temp_{self.instance_id}_{patch_type}.patch",
        )

        with open(patch_path, "w") as f:
            f.write(patch)

        # Restore test files before applying if patch_type is 'test'
        if patch_type == PatchType.PATCH_TEST.value:
            for test in get_test_directives(self.instance, keep_as_files=True):
                if os.path.exists(test):
                    self.exec(f"git restore {test}".split(" "))

        # Apply patch to testbed directory
        apply_cmd = (
            f"git apply -v -R {patch_path}" if revert else f"git apply -v {patch_path}"
        )
        out_patch = self.exec(apply_cmd.split(" "), raise_error=False, check=False)

        # If git command fails, try patch command
        if out_patch.returncode != 0:
            # Patch may has been partially applied so we should revert it.
            # NOTE: we do not revert the test patch because it may unintentionally revert previously applied patches
            if patch_type != PatchType.PATCH_TEST.value:
                self.exec("git restore .".split(" "))
                # revert to the state of the repo before the patch was applied
                output = self.exec(
                    f"git apply {init_diff_patch_path}".split(),
                    raise_error=False,
                    check=False,
                )
                self.log.write(
                    f"Output (git apply - revert to initial state): {output.stdout}"
                )
            apply_cmd = (
                f"patch -R --batch --fuzz=5 -p1 -i {patch_path}"
                if revert
                else f"patch --batch --fuzz=5 -p1 -i {patch_path}"
            )
            out_patch = self.exec(apply_cmd.split(" "), raise_error=False, check=False)

        # TODO os.remove(patch_path)

        log_cmd = "Revert" if revert else "Apply"
        if out_patch.returncode != 0:
            # Patch apply failed
            self.log.write(f"{log_cmd} patch failed ({patch_type})", level=ERROR)
            with open(self.log_file, "a") as f:
                f.write(f"{APPLY_PATCH_FAIL}; ({patch_type})\nOutput:\n")
                f.write(out_patch.stdout)
                if out_patch.stderr:
                    f.write(out_patch.stderr)
                if (
                    patch_type != PatchType.PATCH_TEST.value
                    and "patching" in out_patch.stdout
                ):
                    # Patch has been partially applied so we should revert it.
                    self.exec("git restore .".split(" "))
                    # revert to the state of the repo before the patch was applied
                    output = self.exec(
                        f"git apply {init_diff_patch_path}".split(),
                        raise_error=False,
                        check=False,
                    )
                    self.log.write(
                        f"Output (git apply - revert to initial state): {output.stdout}"
                    )
            return False

        # Patch apply succeeded
        self.log.write(f"{log_cmd} patch successful ({patch_type})")
        with open(self.log_file, "a") as f:
            f.write(f"{APPLY_PATCH_PASS} ({patch_type})\n")
        return True

    def run_mutation_testing(self, instance, specifications, test_time, test_cmd):
        with open("mutation.toml", "w") as mutant_file:
            formatted_content = MUTATION_TEMPLATE.format(
                source_fp=instance["code_file"],
                timeout=max(10, 1.5 * test_time),
                test_cmd=test_cmd,
            )
            mutant_file.write(formatted_content)

        if "image" in specifications and specifications["image"] == "python":
            self.exec(
                "cosmic-ray init mutation.toml mutation.sqlite".split(),
                shell=False,
                check=False,
                timeout=self.timeout,
            )
            try:
                self.exec(
                    "cosmic-ray exec mutation.toml mutation.sqlite".split(),
                    shell=False,
                    check=False,
                    timeout=self.mutation_timeout,
                )
            except subprocess.TimeoutExpired:
                self.log.write("MutationTimeout")

            output = str(
                self.exec(
                    "cr-rate mutation.sqlite  --estimate --confidence 95.0".split(),
                    shell=False,
                    check=False,
                ).stdout
            )
            num_output = str(
                self.exec(
                    "cr-report mutation.sqlite".split(), shell=False, check=False
                ).stdout
            )
            if "total jobs: " in num_output:
                num_mutants = num_output.split("total jobs: ")[1].split("\n")[0]
                self.log.write(f"\nMutationNum: {num_mutants}")

        else:
            self.exec(
                f"{self.cmd_conda_run} cosmic-ray init mutation.toml mutation.sqlite".split(),
                shell=False,
                check=False,
                timeout=self.timeout,
            )
            try:
                self.exec(
                    f"{self.cmd_conda_run} cosmic-ray exec mutation.toml mutation.sqlite".split(),
                    shell=False,
                    check=False,
                    timeout=self.mutation_timeout,
                )
            except subprocess.TimeoutExpired:
                self.log.write("MutationTimeout")

            output = str(
                self.exec(
                    f"{self.cmd_conda_run} cr-rate mutation.sqlite  --estimate --confidence 95.0".split(),
                    shell=False,
                    check=False,
                ).stdout
            )
            num_output = str(
                self.exec(
                    f"{self.cmd_conda_run} cr-report mutation.sqlite".split(),
                    shell=False,
                    check=False,
                ).stdout
            )
            if "total jobs: " in num_output:
                num_mutants = num_output.split("total jobs: ")[1].split("\n")[0]
                self.log.write(f"\nMutationNum: {num_mutants}")

        if len(output.strip().split(" ")) == 3:
            low, val, high = output.split(" ")
            low = float(low)
            val = float(val)
            high = float(high)

            confidence_range = high - val
            mutation_score = 100 - val

            self.log.write(f"\nMutationLOG: {mutation_score}%")
            self.log.write(f"\nMutationUncertainty: {confidence_range}")
        else:
            self.log.write(f"\nMutationFAIL")

    def run_testing_diagnostic(self, instance: dict, log_data=True):
        specifications = MAP_VERSION_TO_INSTALL[self.instance["repo"]][
            self.instance["version"]
        ]

        try:
            if "tox" in instance["test_cmd"]:
                test_cmd = self.add_coverage_tox("tox.ini")

            if "image" in specifications and specifications["image"] == "python":
                test_cmd = f"{instance['test_cmd']}"
            else:
                test_cmd = f"{self.cmd_conda_run} {instance['test_cmd']}"

            start = time.time()
            out_test = self.exec(
                test_cmd.split(), shell=False, timeout=self.timeout, check=False
            )
            end = time.time()

            test_time = end - start
            self.log.write(f"TestsTime: {test_time}")

            self.log.write(f"\n{TESTS_PASSED}\n")
            self.log.write(f"\nCoverageLOG: 100%\n")

            with open("mutation.toml", "w") as mutant_file:
                formatted_content = MUTATION_TEMPLATE.format(
                    source_fp=instance["code_file"], timeout=10, test_cmd="test"
                )
                mutant_file.write(formatted_content)

            if "image" in specifications and specifications["image"] == "python":
                self.exec(
                    "cosmic-ray init mutation.toml mutation.sqlite".split(),
                    shell=False,
                    check=False,
                )
                output = str(
                    self.exec(
                        "cr-report mutation.sqlite".split(), shell=False, check=False
                    ).stdout
                )
            else:
                self.exec(
                    f"{self.cmd_conda_run} cosmic-ray init mutation.toml mutation.sqlite".split(),
                    shell=False,
                    check=False,
                )
                output = str(
                    self.exec(
                        f"{self.cmd_conda_run} cr-report mutation.sqlite".split(),
                        shell=False,
                        check=False,
                    ).stdout
                )

            num_mutants = output.split("total jobs: ")[1].split("\n")[0]
            self.log.write(f"\nMutationLOG: {num_mutants}%")
        except subprocess.TimeoutExpired:
            # Test command run timed out
            self.log.write("Test script run timed out", level=ERROR)
            if log_data:
                self.log.write(f"{TESTS_TIMEOUT} after {self.timeout} seconds\n")

    def run_tests_task(self, instance: dict, log_data=True, skip_mutation=False):
        """
        Run tests for task instance

        Args:
            instance (dict): Task instance
        Returns:
            bool: True if test script ran successfully, False otherwise
        """
        try:
            if os.path.exists(".coveragerc"):
                os.remove(".coveragerc")

            # Run test command for task instance
            specifications = MAP_VERSION_TO_INSTALL[self.instance["repo"]][
                self.instance["version"]
            ]

            if "tox" in instance["test_cmd"]:
                test_cmd = self.add_coverage_tox("tox.ini")

            if "image" in specifications and specifications["image"] == "python":
                test_cmd = f"{instance['test_cmd']}"
            else:
                test_cmd = f"{self.cmd_conda_run} {instance['test_cmd']}"

            if log_data:
                self.log.write(f"Test Script: {test_cmd};\n")

            start = time.time()
            out_test = self.exec(
                test_cmd.split(), shell=False, timeout=self.timeout, check=False
            )
            end = time.time()

            test_time = end - start

            self.log.write(f"TestsTime: {test_time}")

            if log_data:
                # Write pass/fail status to log file
                if out_test.returncode != 0:
                    self.log.write(f"\n{TESTS_FAILED}\n")
                else:
                    self.log.write(f"\n{TESTS_PASSED}\n")
                    self.log.write(f"Current Working Directory: {os.getcwd()}")

                    if (
                        "image" in specifications
                        and specifications["image"] == "python"
                    ):
                        coverage_data_cmd = f"coverage json -o coverage.json"
                    else:
                        coverage_data_cmd = (
                            f"{self.cmd_conda_run} coverage json -o coverage.json"
                        )

                    self.exec(coverage_data_cmd.split(), shell=False, check=False)
                    cov_success = False
                    with open("coverage.json", "r") as cov_file:
                        coverage_data = json.load(cov_file)
                        if instance["code_file"] in coverage_data["files"].keys():
                            file_data = coverage_data["files"][instance["code_file"]]
                            cov_success = True
                            self.log.write(
                                f"\nCoverageLOG: {file_data['summary']['percent_covered']}%\n"
                            )
                        else:
                            self.log.write(
                                f"\nCoverageFAIL:{instance['code_file']} not found in coverage data\n"
                            )
                    if cov_success and not skip_mutation:
                        self.log.write("Running mutation testing")
                        self.run_mutation_testing(
                            instance, specifications, test_time, test_cmd
                        )

            self.log.write(f"Test script run successful")
            return True, out_test.returncode == 0
        except subprocess.TimeoutExpired:
            # Test command run timed out
            self.log.write("Test script run timed out", level=ERROR)
            if log_data:
                self.log.write(f"{TESTS_TIMEOUT} after {self.timeout} seconds\n")
            return False, False
        except Exception as e:
            # Test command run failed
            self.log.write(f"Test script run failed", level=ERROR)
            if log_data:
                self.log.write(f"{TESTS_ERROR}: {e}")
            self.log.write(format_exc(), level=ERROR)
            exit()
            # return False
        finally:
            if os.path.exists("mutation.sqlite"):
                os.remove("mutation.sqlite")

            if os.path.exists("mutation.toml"):
                os.remove("mutation.toml")

            if os.path.exists("coverage.json"):
                os.remove("coverage.json")
            if os.path.exists(".coverage"):
                self.log.write("Removing coverage")
                os.remove(".coverage")
            if os.path.exists(".pytest_cache"):
                self.log.write("Removing cache")
                shutil.rmtree(".pytest_cache")

    def __exit__(self, exc_type, exc_value, exc_traceback):
        os.chdir(self.cwd)
        try:
            os.chmod(self.log_file, 0o666)
        except Exception as e:
            self.log.write(f"Error changing file permissions: {e}", level=ERROR)
