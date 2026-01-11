from __future__ import annotations

import ast
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/describe_tests.py <test_file.py>")

    path = Path(sys.argv[1])
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))

    lines = src.splitlines()

    def get_line(n: int) -> str:
        if 1 <= n <= len(lines):
            return lines[n - 1]
        return ""

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            print(f"\nCLASS {node.name}  (line {node.lineno})")
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name.startswith("test_"):
                    doc = ast.get_docstring(item) or ""
                    print(f"\n  TEST {item.name}  (line {item.lineno})")
                    if doc:
                        print(f"    doc: {doc}")
                    # print assert statements (source lines)
                    for sub in ast.walk(item):
                        if isinstance(sub, ast.Assert):
                            print(f"    assert@{sub.lineno}: {get_line(sub.lineno).strip()}")
        elif isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            doc = ast.get_docstring(node) or ""
            print(f"\nTEST {node.name}  (line {node.lineno})")
            if doc:
                print(f"  doc: {doc}")
            for sub in ast.walk(node):
                if isinstance(sub, ast.Assert):
                    print(f"  assert@{sub.lineno}: {get_line(sub.lineno).strip()}")


if __name__ == "__main__":
    main()
