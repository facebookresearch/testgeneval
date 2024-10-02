from datasets import Dataset, DatasetDict
from inference.configs.config_utils import get_first_method_partial_python


class InstructPrompt:
    def __init__(self):
        self.SYSTEM_MESSAGE = "You are an expert Python software testing assistant. Your job is to complete the next test given a code file an some existing test context."
        self.SYSTEM_MESSAGE_FULL = "You are an expert Python automated testing assistant. Your job is to generate a test file given a code file."

        self.PROMPT_FULL = """Below is a code file:
```python
{code_src}
```

The code file is called: {code_filename}

Your job is to output a corresponding unit test file that obtains high coverage and invokes the code under test.

Here are some examples of how to import {code_filename}, (you should use these as reference)

```python
{imports}
```

Each unit test must be a function starting with test_. Include all your test imports and setup before your first test. Do not 
run the tests in the file, just output a series of tests. Do not include a main method to run the tests.

Only output the unit test Python file in this format:

```python
Unit test Python code (file level)
```
"""

        self.PROMPT_FULL_NO_IMPORT = """Below is a code file:
```python
{code_src}
```

The code file is called: {code_filename}

Your job is to output a corresponding unit test file that obtains high coverage and invokes the code under test.

Each unit test must be a function starting with test_. Include all your test imports and setup before your first test. Do not 
run the tests in the file, just output a series of tests. Do not include a main method to run the tests.

Only output the unit test Python file in this format:

```python
Unit test Python code (file level)
```
"""

        self.PROMPT_COMPLETION = """Below is a code file:
```python
{code_src}
```

And the current unit test file
```python
{test_src}
```

Your job write the Python code the next test in the file. Ideally your next test should improve coverage of the 
existing unit test file for the code file.

Only output the next unit test, preserve indentation and formatting. Do not output anything else. Format like this:

```python
Next unit test Python code
```
"""

    @property
    def system_message(self):
        return self.SYSTEM_MESSAGE

    @property
    def system_message_full(self):
        return self.SYSTEM_MESSAGE_FULL

    def postprocess_output(self, text, is_full):
        text = text.replace("```python", "```")
        if "```" not in text:
            return "compilation error"
        text_cleaned = text.split("```")[1].split("```")[0]
        return (
            text_cleaned if is_full else get_first_method_partial_python(text_cleaned)
        )

    def add_prompts_to_dataset(self, dataset, no_import=False, tokenizer=None):
        test_data = dataset["test"]

        new_arr = []
        for new_data in test_data:
            code_src = new_data["code_src"]
            full_context = self.PROMPT_FULL.format(
                code_src=code_src,
                code_filename=new_data["code_file"],
                imports="\n".join(new_data["local_imports"]),
            )
            full_context_no_import = self.PROMPT_FULL_NO_IMPORT.format(
                code_src=code_src, code_filename=new_data["code_file"]
            )
            first_context = self.PROMPT_COMPLETION.format(
                code_src=code_src, test_src=new_data["preds_context"]["preamble"]
            )
            last_context = self.PROMPT_COMPLETION.format(
                code_src=code_src, test_src=new_data["preds_context"]["last_minus_one"]
            )
            extra_context = self.PROMPT_COMPLETION.format(
                code_src=code_src, test_src=new_data["preds_context"]["last"]
            )

            new_data["preds_prompts"] = {
                "full": full_context_no_import if no_import else full_context,
                "first": first_context,
                "last": last_context,
                "extra": extra_context,
            }
            new_arr.append(new_data)

        final_dataset = DatasetDict({"test": Dataset.from_list(new_arr)})

        return final_dataset
