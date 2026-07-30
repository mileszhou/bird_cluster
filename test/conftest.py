"""Shared pytest setup. Tests are run from within test/ (see README.md).

The project package is literally named `code`, which shadows the standard
library's `code` module. Under `python3 -m code.embedding.embed` the cwd wins and
this never comes up, but pytest (via its plugins) imports the stdlib `code`
before collecting, so by the time a test module runs, `sys.modules["code"]` is
already the stdlib one and `import code.lib` fails with "not a package".
Dropping the non-package entry lets the repo-root path win instead.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_preloaded = sys.modules.get("code")
if _preloaded is not None and not hasattr(_preloaded, "__path__"):
    del sys.modules["code"]
