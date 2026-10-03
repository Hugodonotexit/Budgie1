"""Loads the repository root as the package `budgie`, whatever the checkout directory is called, so the
tests run from a plain clone without `pip install`."""

import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

if "budgie" not in sys.modules:
    spec = importlib.util.spec_from_file_location("budgie", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["budgie"] = module
    spec.loader.exec_module(module)
