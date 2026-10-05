"""Run every project's test suite without needing pytest.

    python run_tests.py            # all projects
    python run_tests.py 02         # only projects whose folder starts with 02

The tests are ordinary pytest-style functions, so `pytest` inside a project
folder works too (each tests/ folder has a conftest.py that adds ../src to
the import path). Each project runs in its own interpreter because several
projects share module names such as generate_plots.py.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

RUNNER = r"""
import importlib.util, inspect, sys, time, traceback
from pathlib import Path
proj = Path(sys.argv[1])
sys.path.insert(0, str(proj / "src"))
passed = failed = 0
for f in sorted((proj / "tests").glob("test_*.py")):
    spec = importlib.util.spec_from_file_location(f.stem, f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if not name.startswith("test_") or fn.__module__ != mod.__name__:
            continue
        t0 = time.time()
        try:
            params = inspect.signature(fn).parameters
            if "tmp_path" in params:
                import tempfile
                with tempfile.TemporaryDirectory() as d:
                    fn(tmp_path=Path(d))
            else:
                fn()
            passed += 1
            print(f"  PASS  {f.stem}::{name}  ({time.time()-t0:.1f}s)")
        except Exception:
            failed += 1
            print(f"  FAIL  {f.stem}::{name}")
            traceback.print_exc()
print(f"  -> {passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
"""


def main() -> int:
    prefix = sys.argv[1] if len(sys.argv) > 1 else ""
    projects = sorted(p for p in ROOT.iterdir()
                      if p.is_dir() and (p / "tests").is_dir() and p.name.startswith(prefix))
    failures = 0
    for proj in projects:
        print(f"\n== {proj.name}")
        failures += subprocess.run([sys.executable, "-c", RUNNER, str(proj)]).returncode != 0
    print(f"\n{len(projects) - failures}/{len(projects)} projects green")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
