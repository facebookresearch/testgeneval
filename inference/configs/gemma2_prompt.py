from datasets import Dataset, DatasetDict
from inference.configs.config_utils import get_first_method_partial_python
from inference.configs.instruct_prompt import InstructPrompt


class Gemma2Prompt(InstructPrompt):
    def postprocess_output(self, text, is_full):
        text = text.replace("```python", "```")
        text = text.split("<|end_of_turn|>")[-1]
        if "```" not in text:
            return "compilation error"
        text_cleaned = text.split("```")[1].split("```")[0]
        return (
            text_cleaned if is_full else get_first_method_partial_python(text_cleaned)
        )

    def add_prompts_to_dataset(self, dataset, no_import=False, tokenizer=None):
        assert tokenizer != None
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
            for setting in new_data["preds_prompts"]:
                chat = []
                chat.append(
                    {"role": "user", "content": new_data["preds_prompts"][setting]}
                )
                new_data["preds_prompts"][setting] = tokenizer.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True
                )
            new_arr.append(new_data)

        final_dataset = DatasetDict({"test": Dataset.from_list(new_arr)})

        return final_dataset
