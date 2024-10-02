import json
import os
import sys
import time
import traceback

from ast_utils import get_all_method_defs_from_graph, get_python_program_graph


def get_preamble_context(method_defs, test_file_content):
    context = ""
    curr_method = None
    for method in method_defs:
        print("method", method)
        if "test" in method:
            print("found first test method", method)
            curr_method = method
            break

    if curr_method is not None:
        first_test_method = method_defs[method]
        first_test_method_start_lineno = int(first_test_method["lineno"].split("-")[0])
        test_file_content = test_file_content.split("\n")
        context = "\n".join(test_file_content[: first_test_method_start_lineno - 1])
    # print('get_preamble_context\n',context)
    return context


def get_first_test_context(method_defs, test_file_content):
    context = ""
    curr_method = None
    for method in method_defs:
        print("method", method)
        if "test" in method:
            print("found first test method", method)
            curr_method = method
            break

    if curr_method is not None:
        first_test_method = method_defs[method]
        first_test_method_end_lineno = int(first_test_method["lineno"].split("-")[1])
        test_file_content = test_file_content.split("\n")
        context = "\n".join(test_file_content[:first_test_method_end_lineno])
    # print('get_first_test_context\n',context)
    return context


def get_extra_test_context(method_defs, test_file_content):
    context = test_file_content
    # print('get_extra_test_context\n',context)
    return context


def get_last_test_context(method_defs, test_file_content):
    if len(method_defs.keys()) > 1:
        last_but_one_method = method_defs[list(method_defs.keys())[-2]]
    else:
        print(
            "get_last_test_context == get_preamble_context since len(method_defs.keys())=",
            len(method_defs.keys()),
        )
        return get_preamble_context(method_defs, test_file_content)
    last_but_one_method_end_lineno = int(last_but_one_method["lineno"].split("-")[1])
    test_file_content = test_file_content.split("\n")
    context = "\n".join(test_file_content[:last_but_one_method_end_lineno])
    # print('get_last_test_context\n',context)
    return context


def get_incremental_test_files(
    method_defs, test_file_content, eof_snippet, test_filename, output_path
):
    incremental_methods = []

    test_file_content = test_file_content.split("\n")
    for method in method_defs:
        if "test" in method:
            method_end_lineno = int(method_defs[method]["lineno"].split("-")[1])
            context = "\n".join(test_file_content[:method_end_lineno])
            context += "\n"
            incremental_methods.append(method)

    return incremental_methods


def get_eof_snippet(method_defs, test_file_content):
    last_method = method_defs[list(method_defs.keys())[-1]]
    last_method_end_lineno = int(last_method["lineno"].split("-")[1])
    test_file_content = test_file_content.split("\n")
    eof_snippet = "\n".join(test_file_content[last_method_end_lineno:])
    print(f"eof_snippet:\n {eof_snippet}")
    return eof_snippet


def get_first_test_method(method_defs, test_file_content, test_filename, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    test_file_content = test_file_content.split("\n")
    for method in method_defs:
        if "test" in method:
            method_body = method_defs[method]["body"]
            return method
    return None


def get_last_test_method(method_defs, test_file_content, test_filename, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    test_file_content = test_file_content.split("\n")
    last_method = method_defs[list(method_defs.keys())[-1]]
    method_body = last_method["body"]
    return method_body


def get_context_gold_json(code_filepath, test_filepath):
    code_filename = code_filepath.split("/")[-1][:-3]
    test_filename = test_filepath.split("/")[-1][:-3]
    try:
        test_file_graph, test_file_content = get_python_program_graph(test_filepath)
        test_method_defs = get_all_method_defs_from_graph(
            test_file_graph, test_file_content
        )
        """
        for method in method_defs:
            print(method)
            print(method_defs[method])
            print('\n')
        """
    except Exception as e:
        print("Exception getting code graph in", test_filename, "--", e)
        return None

    preamble_context = get_preamble_context(test_method_defs, test_file_content)

    first_test_context = get_first_test_context(test_method_defs, test_file_content)

    last_test_context = get_last_test_context(test_method_defs, test_file_content)

    extra_test_context = get_extra_test_context(test_method_defs, test_file_content)

    if len(first_test_context) == 0 or len(extra_test_context) == 0:
        print("Skipping because cannot find first test or file is blank")
        return None

    return {
        "preamble": preamble_context,
        "first": first_test_context,
        "last_minus_one": last_test_context,
        "last": extra_test_context,
    }
