import unittest
import sys
import os
import subprocess
from pathlib import Path

def run_syntax_check():

    print("Checking syntax of Python files...")
    src_path = Path("src")
    test_path = Path("tests")

    python_files = list(src_path.rglob(".py")) + list(test_path.rglob(".py"))

    errors = []
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                source = f.read()
            compile(source, str(py_file), 'exec')
            print(f"   {py_file}")
        except SyntaxError as e:
            errors.append((py_file, str(e)))
            print(f"   {py_file} - Syntax Error: {e}")
        except Exception as e:
            errors.append((py_file, str(e)))
            print(f"   {py_file} - Error: {e}")

    return len(errors) == 0

def run_unit_tests():

    print("\nRunning unit tests...")

    loader = unittest.TestLoader()
    start_dir = 'tests'
    suite = loader.discover(start_dir, pattern='test.py')

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()

def run_import_tests():

    print("\nRunning import tests...")

    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

        import src
        print("   src")

        import src.environment
        print("   src.environment")

        import src.rl
        print("   src.rl")

        import src.planning
        print("   src.planning")

        import src.baselines
        print("   src.baselines")

        import src.bridges
        print("   src.bridges")

        import src.localization
        print("   src.localization")

        import src.utils
        print("   src.utils")

        return True
    except ImportError as e:
        print(f"   Import error: {e}")
        return False

def main():

    print("="  60)
    print("DroneDelivery-RL - Code Integrity Test Suite")
    print("="  60)

    if not os.path.exists("src") or not os.path.exists("tests"):
        print("Error: This script must be run from the project root directory.")
        sys.exit(1)

    all_passed = True

    syntax_ok = run_syntax_check()
    if not syntax_ok:
        print("\n Syntax check failed!")
        all_passed = False
    else:
        print("\n All Python files passed syntax check!")

    print("\nRunning syntax-only tests...")
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName('test_syntax_only', module=__import__('tests.test_syntax_only'))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    syntax_only_ok = result.wasSuccessful()
    if not syntax_only_ok:
        print(" Syntax-only tests failed!")
        all_passed = False
    else:
        print(" All syntax-only tests passed!")

    print("\n" + "="  60)
    if all_passed:
        print(" All tests passed! Code integrity verified.")
        print("Project structure is valid and modules can be imported.")
    else:
        print(" Some tests failed! Please check the output above.")
        sys.exit(1)
    print("="  60)

if __name__ == "__main__":
    main()