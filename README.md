# TestGenEval: A Large Scale Test Generation Benchmark


TestGenEval consists of 1,210 code test file pairs from 11 large, well-maintained repositories (3,523-78,287 stars). We use these file pairs to construct two testing tasks: 1) unit test completion for the first, last and additional tests and 2) full file unit test generation. Our benchmark is easy to run and extend, as we have docker containers for each version of each repository with coverage and mutation testing dependencies installed. For both task we use execution based metrics, including pass@1, pass@5 along with code coverage improvement, and mutation score improvement compared to the gold (human written) tests. Code and test files in \benchmark are long in length (on average 782 LOC per code file and 677 LOC per test file) and high coverage (median coverage of 60.4\%).

We measure the following metrics for the test completion task:
- pass@k (k = 1, 5)
- coverage improvement (how much generated test improves existing coverage)
- coverage improvement@pass (coverage improvement averaged only over passing tests)
- average pass@5

We measure the following metrics for the test generation task:
- pass@1
- all pass@1 (all tests generated in suite pass)
- coverage (coverage of generated tests)
- coverage@pass (coverage of generated tests for passing examples)
- mutation score (mutation score of generated tests)
- mutation score@pass (mutation score of generated tests for passing examples)

## Datasets

### TestGenEvalLite
Docker images for testbeds used in the `TestGenEvalLite` dataset has been built and tested.

### TestGenEval
Docker images for testbeds used in the `TestGenEval` dataset has been built and tested.

## Setup and Installation

To setup the repository run
```
git@github.com:facebookresearch/testgeneval
cd testgeneval
conda env create -f testgeneval.yaml
conda activate testgeneval
```

Modify the `.env_template` file with the appropriate values and rename it to `.env` (specifically make sure to set SWEBENCH_DOCKER_FORK_DIR to the current directory where the repository was cloned)

To build the docker images (adapted from [SWEBench Docker](https://github.com/aorwall/SWE-bench-docker/tree/main/docker)) run one of these commands:

**TestGenEvalLite** - TestGenEvalLite for faster evaluation
```
make -f Makefile.testgenevallite
```

**TestGenEval** - full TestGenEval (takes hours to a full day to build)
```
make -f Makefile.testgeneval
```

## Running TestGenEval

Running TestGenEval is relatively simple.

There is a python script that will run both prediction and inference.

```
python run_pipeline.py \
--results_dir results
--dataset_dir dataset/tesgenevallite
--model meta-llama/Meta-Llama-3.1-8B-Instruct
```

## Adding a new model to TestGenEval

Adding a new model is quite simple. Under `inference/configs` create a new file with the system prompts and the function to add prompts to the dataset.

`add_prompts_to_dataset` should output a prompt for all four settings: `full`, `first`, `last`, `extra`. The `preds_context` attribute of each datapoint contains the preamble of the file, the first test, the file without the last test and the file with the last test (full file)

Once you update this file our standard evaluation flow will work.

## TestGenEval creation

All creation scripts are housed in the `creation` subdirectory.

`transform_swebench.py` is the main script that takes the SWEBench dataset and converts it for test generation.

`filter_unittests.py` takes the baseline results and filters out datapoints with no coverage of gold tests (gold tests must cover the code under test).

## Licensing

The majority of code in this repository is licensed under CC-by-NC, however the third party code/files may be subject to different licenses.
