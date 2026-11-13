"""Regression checks for native creation/execution failures and fallback."""
import ast
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class LoaderTest(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[2] / "python/photara/__init__.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        tree.body.pop()  # Define the loader without loading an actual extension.
        self.ns = {"__file__": str(path)}
        exec(compile(tree, str(path), "exec"), self.ns)
        self.temp = tempfile.TemporaryDirectory(prefix="photara-loader-test-")
        self.addCleanup(self.temp.cleanup)
        directories = [Path(self.temp.name) / name for name in ("first", "second")]
        for directory in directories:
            directory.mkdir()
            (directory / "photara.pyd").touch()
        self.ns["_candidate_native_directories"] = lambda: directories
        self.ns["_add_dll_directories"] = lambda directory: None

    def test_creation_failure_falls_back(self):
        module = SimpleNamespace(__version__="test")
        loader = SimpleNamespace(exec_module=lambda module: None)
        with patch.dict(sys.modules), patch.object(
                importlib.util, "spec_from_file_location", return_value=SimpleNamespace(loader=loader)), \
                patch.object(importlib.util, "module_from_spec",
                             side_effect=[ImportError("missing DLL"), module]) as create:
            self.assertIs(self.ns["_load_native"](), module)
            self.assertEqual(create.call_count, 2)
            self.assertIs(sys.modules["photara"], module)

    def test_execution_failure_restores_module_and_reports_all_attempts(self):
        original = object()
        def fail(module):
            raise ImportError("execution failed")
        loader = SimpleNamespace(exec_module=fail)
        with patch.dict(sys.modules, {"photara": original}), patch.object(
                importlib.util, "spec_from_file_location", return_value=SimpleNamespace(loader=loader)), \
                patch.object(importlib.util, "module_from_spec", side_effect=[SimpleNamespace(), SimpleNamespace()]):
            with self.assertRaises(ImportError) as caught:
                self.ns["_load_native"]()
            self.assertIs(sys.modules["photara"], original)
            message = str(caught.exception)
            self.assertIn("Unable to load the photara extension", message)
            self.assertIn("first", message)
            self.assertIn("second", message)


if __name__ == "__main__":
    unittest.main()
