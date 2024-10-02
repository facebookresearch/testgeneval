# Adapted from: https://github.com/aorwall/SWE-bench-docker/blob/main/swebench_docker/run_docker.py

import asyncio
import base64
import json
import logging
import os
import subprocess
import tempfile
import time

import dotenv
from swebench_docker.constants import MAP_VERSION_TO_INSTALL

logger = logging.getLogger(__name__)
dotenv.load_dotenv()

# Needs to be a fully qualified path for log dir


async def run_docker_evaluation(
    task_instance: dict,
    namespace: str,
    log_dir: str,
    setting: str,
    ind: int,
    timeout: int = 60,
    verbose: bool = False,
    base64_instance: bool = True,
    only_baseline: bool = False,
    skip_mutation: bool = False,
):
    repo_name = task_instance["repo"].replace("/", "_")

    specifications = MAP_VERSION_TO_INSTALL[task_instance["repo"]][
        task_instance["version"]
    ]
    image_prefix = "swe-bench"

    # TODO: Change this when deciding
    if "packages" in specifications and specifications["packages"] == "environment.yml":
        container_log_dir = "/home/swe-bench/logs"
    else:
        container_log_dir = "/opt/logs"

    if specifications.get("instance_image", False):
        docker_image = f"{namespace}/{image_prefix}-{repo_name}-instance:{task_instance['instance_id']}"
    else:
        docker_image = (
            f"{namespace}/{image_prefix}-{repo_name}-testbed:{task_instance['version']}"
        )

    swebench_docker_fork_dir = os.environ.get("SWEBENCH_DOCKER_FORK_DIR")

    if swebench_docker_fork_dir:
        # Create a temporary file to store the task_instance JSON
        tmpfile_path = tempfile.mktemp(suffix=".json")
        with open(tmpfile_path, "w+") as f:
            json.dump(task_instance, f)

        docker_command = [
            "docker",
            "run",
            "--rm",
            "--network",
            "host",
            "-v",
            f"{log_dir}:{container_log_dir}",
            # Map the swebench_docker fork dir to the container
            # for some reason, swebench_docker has different locations for the different containers :(
            # so we need to map all of them to make it work
            "-v",
            f"{swebench_docker_fork_dir}/swebench_docker:/opt/swebench_docker:ro",
            "-v",
            f"{swebench_docker_fork_dir}/swebench_docker:/home/swe-bench/swebench_docker:ro",
            "-v",
            f"{swebench_docker_fork_dir}/swebench_docker:/home/swe-bench/swebench:ro",
            # =======
            # Map file instead pass the instance as env var to avoid "Argument list too long" error
            "-v",
            f"{tmpfile_path}:/home/swe-bench/task_instance.json:ro",
            "-e",
            f"LOG_DIR={container_log_dir}",
            "-e",
            f"SETTING={setting}",
            "-e",
            f"IND={ind}",
            "-e",
            f"TIMEOUT={timeout}",
            "-e",
            f"ONLY_BASELINE={only_baseline}",
            "-e",
            f"SKIP_MUTATION={skip_mutation}",
            docker_image,
        ]
    else:
        # Base64 encode the instance JSON to be sure it can be passed as an environment variable
        instance_b64 = base64.b64encode(
            json.dumps(task_instance).encode("utf-8")
        ).decode("utf-8")
        docker_command = [
            "docker",
            "run",
            "--rm",
            "--network",
            "host",
            "--memory_swappiness" "5",
            "-v",
            f"{log_dir}:{container_log_dir}",
            "-e",
            f"INSTANCE={instance_b64}",
            "-e",
            f"LOG_DIR={container_log_dir}",
            "-e",
            f"SETTING={setting}",
            "-e",
            f"IND={ind}",
            "-e",
            f"TIMEOUT={timeout}",
            "-e",
            f"ONLY_BASELINE={only_baseline}",
            "-e",
            f"SKIP_MUTATION={skip_mutation}",
            docker_image,
        ]

    cmd_string = " ".join(docker_command)

    if verbose:
        logger.info(cmd_string)

    start_time = time.time()

    try:
        process = await asyncio.create_subprocess_shell(
            cmd_string, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        stdout, stderr = await process.communicate()
        # Decode stdout and stderr from bytes to str
        str_stdout = stdout.decode() if stdout else ""
        str_stderr = stderr.decode() if stderr else ""

        elapsed_time = time.time() - start_time

        if process.returncode != 0:
            logger.warning(
                f"[{task_instance['id']}][{docker_image}]  Error running container:"
            )
            logger.warning(f"Command: {cmd_string}")
            logger.warning(f"Stdout - {str_stdout}")
            logger.warning(f"Stderr - {str_stderr}")

        elif "Evaluation succeeded" not in str_stdout:
            logger.warning(
                f"[{task_instance['id']}][{docker_image}]  Container ran successfully in {elapsed_time} seconds, but evaluation failed."
            )
            logger.warning(f"Command: {cmd_string}")
            logger.warning(f"stdout - {str_stdout}")
        else:
            logger.info(
                f"[{task_instance['id']}][{docker_image}]  Container ran successfully in {elapsed_time} seconds."
            )
    except Exception as e:
        logger.warning(
            f"[{task_instance['id']}][{docker_image}]  Error running container: {e}"
        )
    finally:
        if swebench_docker_fork_dir:
            # Ensure the temporary file is deleted after the Docker process completes
            os.unlink(tmpfile_path)
