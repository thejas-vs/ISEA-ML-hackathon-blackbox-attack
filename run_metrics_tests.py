# -*- coding: utf-8 -*-
"""run_metrics_tests.py
Execute the simple sanity tests for metrics without requiring pytest.
"""
import sys, os, traceback

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'blackbox-attack')) if False else r'C:\Users\theja\Downloads\ISEA ML\blackbox-attack'
code_dir = os.path.join(project_root, 'code')
sys.path.append(code_dir)

# Import the test module directly
test_path = os.path.join(project_root, 'tests', 'test_metrics.py')
import importlib.util
spec = importlib.util.spec_from_file_location('test_metrics', test_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

failed = False
for name in dir(mod):
    if name.startswith('test_'):
        func = getattr(mod, name)
        try:
            func()
            print(f"{name}: PASSED")
        except AssertionError as e:
            failed = True
            print(f"{name}: FAILED (assertion) - {e}")
        except Exception:
            failed = True
            print(f"{name}: FAILED (exception)")
            traceback.print_exc()

sys.exit(1 if failed else 0)
