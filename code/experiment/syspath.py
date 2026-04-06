from pathlib import Path
from typing import List

# 1️⃣ Make the `code` and `lib` directories importable
import sys, os
# Absolute path to the folder that contains the module
def impt():
    dir = os.path.abspath('./code')
    if dir not in sys.path: sys.path.append(dir)
    dir = os.path.abspath('./lib')
    if dir not in sys.path: sys.path.append(dir)