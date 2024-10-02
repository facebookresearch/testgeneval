import ast
from collections import defaultdict


def get_functions_and_classes(content, fp):
    tree = ast.parse(content, filename=fp)

    functions = []
    classes = []

    class FunctionClassVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            functions.append(node.name)
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node):
            functions.append(node.name)
            self.generic_visit(node)

        def visit_ClassDef(self, node):
            classes.append(node.name)
            self.generic_visit(node)

    visitor = FunctionClassVisitor()
    visitor.visit(tree)

    return functions, classes


def get_local_import_statements(
    code_content, code_fp, code_file_suf, test_content, test_fp
):
    tree = ast.parse(test_content, filename=test_fp)
    import_statements = []
    lines = test_content.splitlines()
    code_fns, code_classes = get_functions_and_classes(code_content, code_fp)

    def get_multiline_import(start_lineno):
        statement_lines = []
        in_parentheses = False
        for lineno in range(start_lineno - 1, len(lines)):
            line = lines[lineno].strip()
            if "(" in line:
                in_parentheses = True
            if line.endswith("\\") or in_parentheses:
                statement_lines.append(line)
                if line.endswith(")"):
                    in_parentheses = False
                    break
            else:
                statement_lines.append(line)
                break
        return " ".join(statement_lines).replace("\\", "")

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            import_statement = get_multiline_import(node.lineno)
            import_statement_lower = import_statement.lower()
            file_suf_lower = code_file_suf.lower()
            found_import = False
            if file_suf_lower in import_statement_lower:
                import_statements.append(import_statement)
                found_import = True
            if not found_import:
                for code_fn in code_fns:
                    if code_fn.lower() in import_statement_lower:
                        import_statements.append(import_statement)
                        found_import = True
                        break
            if not found_import:
                for code_class in code_classes:
                    if code_class.lower() in import_statement_lower:
                        import_statements.append(import_statement)
                        found_import = True
                        break
    return import_statements


def get_python_program_graph(filename):
    """Get the python program graph."""
    with open(filename, "r", errors="ignore") as f_in:
        file_content = f_in.read()

    tree = ast.parse(file_content)
    return tree, file_content


def add_parent_info(node, parent=None):
    """Recursively add parent information to each node."""
    node.parent = parent
    for child in ast.iter_child_nodes(node):
        add_parent_info(child, node)


def get_class_name(node):
    """Get the class name if the function is within a class."""
    while node:
        if isinstance(node, ast.ClassDef):
            return node.name + "."
        node = getattr(node, "parent", None)
    return ""


def get_all_method_defs_from_graph(tree, file_content):
    """Extract all method definitions along with the span and the method bodies from the tree."""
    method_defs = defaultdict(dict)
    file_content = file_content.split("\n")

    class FunctionVisitor(ast.NodeVisitor):
        def __init__(self):
            self.func_defs = []

        def visit_FunctionDef(self, node):
            self.func_defs.append(node)
            self.generic_visit(node)

    add_parent_info(tree)  # Add parent information to each node

    visitor = FunctionVisitor()
    visitor.visit(tree)

    for func_def_node in visitor.func_defs:
        start_lineno = func_def_node.lineno
        end_lineno = func_def_node.end_lineno
        func_def_value = func_def_node.name
        func_start_lineno = start_lineno
        func_end_lineno = end_lineno
        func_body = file_content[func_start_lineno - 1 : func_end_lineno]
        class_value = get_class_name(func_def_node)
        method_defs[class_value + func_def_value]["lineno"] = (
            str(func_start_lineno) + "-" + str(func_end_lineno)
        )
        method_defs[class_value + func_def_value]["body"] = "\n".join(func_body)
    return method_defs


def resolve_object_type(method_calls, object_class_map):
    """Helper: Resolve the type of objects."""
    resolved_method_calls = []
    for method_call in method_calls:
        if (
            "." in method_call and method_call.split(".")[0] in object_class_map.keys()
        ):  # obj.func()
            resolved_method_calls.append(
                object_class_map[method_call.split(".")[0]]
                + "."
                + method_call.split(".")[1]
            )
        elif method_call in object_class_map.keys():  # obj()
            resolved_method_calls.append(object_class_map[method_call])
        else:
            resolved_method_calls.append(method_call)  # func()
    return resolved_method_calls


def get_changed_method_names(diff_lines, method_defs):
    """Get the names of methods that changed using the diff line numbers."""
    changed_methods = []
    for method in method_defs.keys():
        span = method_defs[method]["lineno"]
        start = int(span.split("-")[0])
        end = int(span.split("-")[1])
        for line in diff_lines:
            if line >= start and line <= end:
                changed_methods.append(method)
                break
    return changed_methods


def get_code_test_method_mapping_ast(
    test_changed_method_calls_before,
    test_changed_method_calls_after,
    code_changed_method_names,
):
    """Align the code and test methods based on the function calls in the test method.
    Code method must be present in both the before and after test file."""
    mapped = []
    if (
        test_changed_method_calls_before
        and test_changed_method_calls_after
        and code_changed_method_names
    ):
        for code_method in code_changed_method_names:
            if (
                code_method in test_changed_method_calls_before
                and code_method in test_changed_method_calls_after
            ):
                mapped.append(code_method)
    return mapped
