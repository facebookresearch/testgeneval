import re


def check_if_new_file_started(content):
    """
    Check if the content added to a file resembles the start of a new file.

    Args:
    content (str): The content that was added to the file.

    Returns:
    bool: True if the content resembles the start of a new file, False otherwise.
    """
    # Patterns that might indicate the start of a new file
    patterns = [
        r"^\s*import\s+\w+",  # import statements at the start
        r"^\s*from\s+\w+\s+import\s+\w+",  # from ... import ... statements
        r"^\s*#!",  # Shebang line (common in Unix-based scripts)
        r'^\s*"""',  # Docstring at the start of the file
        r"^\s*def\s+\w+\s*\(",  # Function definition at the start
        r"^\s*class\s+\w+\s*:",  # Class definition at the start
    ]

    # Check if any pattern matches the start of the content
    for pattern in patterns:
        if re.match(pattern, content.strip(), re.MULTILINE):
            return True

    return False


def get_first_method_partial_python(text_cleaned):
    lines = text_cleaned.split("\n")
    # Initialize variables to track the first top-level method
    first_method = []
    in_method = False
    indent_level = None

    # Iterate over each line to find the first top-level method
    for line in lines:
        stripped_line = line.strip()

        # Check if the line contains a method definition
        if re.match(r"def\s+\w+\s*\(", stripped_line):
            current_indent = len(line) - len(line.lstrip())

            # If we are not currently in a method, or this is a top-level method
            if not in_method or (
                indent_level is not None and current_indent <= indent_level
            ):
                if not first_method:
                    # Start capturing the first method
                    first_method = [line]
                    indent_level = current_indent
                    in_method = True
                elif current_indent <= indent_level:
                    # If another method at the same or higher level starts, stop capturing
                    break
        elif in_method:
            # Continue capturing lines that are part of the method body
            if line.strip() == "" or len(line) - len(line.lstrip()) > indent_level:
                first_method.append(line)
            else:
                # Stop capturing if the indentation returns to the level of the method definition or less
                break

    # Join the lines of the first method to form the complete method body
    return "\n".join(first_method) if first_method else "this does not compile"
