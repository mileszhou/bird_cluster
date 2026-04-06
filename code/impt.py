# code/impt.py
#!/usr/bin/env python3

from pathlib import Path
import pathlib
from typing import List



# 1️⃣ Make the `code` and `lib` directories importable
import sys, os
# Absolute path to the folder that contains the module
dir = os.path.abspath('./')
if dir not in sys.path: sys.path.append(dir)
dir = os.path.abspath('./code')
if dir not in sys.path: sys.path.append(dir)
dir = os.path.abspath('./code/lib')
if dir not in sys.path: sys.path.append(dir)

# impt.py
# This wrapper exposes the public API of the `code/lib` package.
# Add or remove imports as needed for your project.

# -----------------------------------------------------------------
# 1️⃣ Add the project root (parent of the ``code/`` folder) to sys.path.
#    image_path.parent.parent is the folder that contains ``code/``.\n
# -----------------------------------------------------------------
root_dir = pathlib.Path(__file__).parents[1]   # .. (up one level from ``code/``)
sys.path.append(str(root_dir))

# Example – adjust to match actual module names in code/lib:
from .lib.label_generator import build_label, pinyin_initials
# from code.lib.some_other_module import some_function, SomeClass --- IGNORE ---

# If you have many modules, you can automatically import all .py files:
# (uncomment the block below if you want an auto‑import for *all* modules)

# import importlib, pkgutil, pathlib
# lib_path = pathlib.Path(__file__).parent / 'code' / 'lib'
# for module in pkgutil.iterdir(lib_path):
#     if module.suffix == ".py" and module.name != "__init__.py":
#         module_name = module.stem
#         globals()[module_name] = importlib.import_module(f"code.lib.{module_name}")

__all__ = [
    "build_label",
    "pinyin_initials"
    # add any other symbols you want to expose here
]

# -----------------------------------------------------------------
# 3️⃣ Optional CLI for quick testing\n
# -----------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate a label from a name and confidence.")
    parser.add_argument("name", help="Chinese name (e.g. 张三)")
    parser.add_argument("conf", type=float, help="Confidence (0‑1 or 0‑100)")
    args = parser.parse_args()
    print(build_label(args.name, args.conf))