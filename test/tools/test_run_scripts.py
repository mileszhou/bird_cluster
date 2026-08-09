"""The wrapper scripts must forward the user's arguments.

`./run-vllm --years 2024,2025` silently ran the whole 49k library because
run-vllm ended at its own last flag and never passed "$@" through. Nothing in
the output said so -- argparse saw defaults and reported them faithfully, so
args.json recorded `years: null` and looked correct.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = sorted(p for p in ROOT.glob("run-*") if p.is_file())


def test_there_are_wrapper_scripts():
    assert SCRIPTS, "expected run-* wrappers at the repo root"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_wrapper_forwards_user_arguments(script):
    assert '"$@"' in script.read_text(encoding="utf-8"), (
        f'{script.name} does not forward "$@", so any flag passed to it is '
        f'silently discarded')
