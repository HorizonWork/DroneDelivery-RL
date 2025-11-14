import unittest
import ast
import os
from pathlib import Path

class TestSyntaxOnly(unittest.TestCase):

    def test_src_syntax(self):

        src_path = Path("src")
        py_files = list(src_path.rglob(".py"))

        errors = []
        for py_file in py_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source = f.read()
                ast.parse(source)
            except SyntaxError as e:
                errors.append((py_file, str(e)))
            except Exception as e:
                errors.append((py_file, str(e)))

        if errors:
            error_messages = [f"{file}: {error}" for file, error in errors]
            self.fail(f"Found syntax errors in {len(errors)} files:\n" + "\n".join(error_messages))
        else:
            print(f" All {len(py_files)} Python files in src/ have valid syntax!")

    def test_tests_syntax(self):

        test_path = Path("tests")
        py_files = list(test_path.rglob(".py"))

        errors = []
        for py_file in py_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source = f.read()
                ast.parse(source)
            except SyntaxError as e:
                errors.append((py_file, str(e)))
            except Exception as e:
                errors.append((py_file, str(e)))

        if errors:
            error_messages = [f"{file}: {error}" for file, error in errors]
            self.fail(f"Found syntax errors in {len(errors)} files:\n" + "\n".join(error_messages))
        else:
            print(f" All {len(py_files)} Python files in tests/ have valid syntax!")

if __name__ == '__main__':
    unittest.main()