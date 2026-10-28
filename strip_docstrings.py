"""
Remove the module-level docstring (the description block at the top) from
every git-tracked .py file in the repo.

Safe by default: it only PRINTS what it would change.

    python strip_docstrings.py            # dry run, changes nothing
    python strip_docstrings.py --apply    # rewrite the files
    python strip_docstrings.py --apply --short   # keep only a one-line docstring

Only the docstring at the very top of each file is touched. Function and
class docstrings, comments, shebang and encoding lines are left alone.
Files that read __doc__ (e.g. to print usage) are skipped. Every rewritten
file is re-parsed to prove it is still valid Python; if not, it is not saved.
"""

import ast
import subprocess
import sys

APPLY = "--apply" in sys.argv
SHORT = "--short" in sys.argv


def tracked_py():
    out = subprocess.run(["git", "ls-files", "*.py"], capture_output=True, text=True, check=True)
    return [f for f in out.stdout.splitlines() if f.strip()]


def process(path):
    raw = open(path, "rb").read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return "skipped (not UTF-8)"
    bom = text.startswith("﻿")
    if bom:
        text = text[1:]
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        return f"skipped (syntax error: {e.msg})"
    if not tree.body:
        return "no docstring"
    first = tree.body[0]
    if not (isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant)
            and isinstance(first.value.value, str)):
        return "no docstring"
    if "__doc__" in text.replace("ast.get_docstring", ""):
        return "skipped (file uses __doc__)"

    lines = text.splitlines(keepends=True)
    start, end = first.lineno - 1, first.end_lineno
    nl = "\r\n" if "\r\n" in text else "\n"
    replacement = []
    if SHORT:
        doc = first.value.value.strip().splitlines()
        one = doc[0].strip() if doc else ""
        if one:
            replacement = ['"""' + one.replace('"""', "'''") + '"""' + nl]
    new = lines[:start] + replacement + lines[end:]
    i = start + len(replacement)
    if replacement:                      # keep one blank line after a kept one-liner
        while i < len(new) and new[i].strip() == "":
            del new[i]
        new.insert(i, nl)
    else:                                # drop blank lines left at the top
        while i < len(new) and new[i].strip() == "":
            del new[i]
    result = "".join(new)
    try:
        ast.parse(result)
    except SyntaxError as e:
        return f"NOT CHANGED (result would not parse: {e.msg})"
    removed = end - start
    if APPLY:
        out = ("﻿" if bom else "") + result
        open(path, "wb").write(out.encode("utf-8"))
        return f"rewritten (removed {removed} docstring lines)"
    return f"would remove {removed} docstring lines"


def main():
    files = tracked_py()
    if not files:
        print("No tracked .py files found. Run this from inside the repo.")
        return
    print(("APPLYING" if APPLY else "DRY RUN (nothing is changed)") +
          (" - keeping a one-line docstring" if SHORT else "") + "\n")
    for f in files:
        print(f"{f:60s} {process(f)}")
    if not APPLY:
        print("\nLooks right? Run again with --apply")
    else:
        print("\nDone. Review with:  git --no-pager diff --stat")


if __name__ == "__main__":
    main()
