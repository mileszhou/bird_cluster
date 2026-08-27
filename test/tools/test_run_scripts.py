"""The wrapper scripts must forward the user's arguments.

`./run-vllm --years 2024,2025` silently ran the whole 49k library because
run-vllm ended at its own last flag and never passed "$@" through. Nothing in
the output said so -- argparse saw defaults and reported them faithfully, so
args.json recorded `years: null` and looked correct.

All three prefixes are covered, not just `run-`: `run-` is a pipeline stage that
writes to output/, `tool-` reports on the dataset without changing it, and
`server-` starts a long-running process. The bug is a property of being a
wrapper, not of being a stage.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = sorted(p for pre in ("run-*", "tool-*", "server-*")
                 for p in ROOT.glob(pre) if p.is_file())


def test_there_are_wrapper_scripts():
    assert SCRIPTS, "expected run-* wrappers at the repo root"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_wrapper_does_not_silently_discard_arguments(script):
    """Forward everything, or refuse what you do not understand.

    Most of these are thin wrappers and satisfy this by ending in "$@". run-all
    is not thin -- it parses its own flags and drives five stages -- so it
    cannot forward blindly, and instead exits non-zero on an argument it does
    not recognise. Either is fine. What is not fine is accepting a flag and
    doing nothing with it, which is how `./run-vllm --years 2024,2025`
    relabelled the whole library.
    """
    text = script.read_text(encoding="utf-8")
    forwards = '"$@"' in text
    rejects = "unknown argument" in text
    assert forwards or rejects, (
        f"{script.name} neither forwards \"$@\" nor rejects unknown arguments, "
        f"so a flag passed to it is silently discarded")


# --- ./clean is not a wrapper, and needs its own refusal ---------------------

CLEAN = ROOT / "clean"


def _clean_sandbox(tmp_path):
    """A copy of ./clean in a scratch directory, with an output/ to archive.

    The script does `cd "$(dirname "$0")"`, so it always acts on the directory it
    sits in. Testing the real one would archive the live run -- which is exactly
    the accident being guarded against -- so the copy is the point, not a
    convenience: if the guard ever regresses, this test fails instead of moving
    somebody's vectors.
    """
    import shutil

    script = tmp_path / "clean"
    shutil.copy2(CLEAN, script)
    (tmp_path / "output").mkdir()
    (tmp_path / "output" / "marker.txt").write_text("live run\n")
    return script


def _run(script, *args):
    import subprocess

    return subprocess.run([str(script), *args], capture_output=True, text=True)


def test_clean_refuses_an_option_as_a_description(tmp_path):
    """A flag must not become an archive name.

    `./clean --help` archived a live run to output_012_help/ and left an empty
    output/ behind. Nothing was lost -- clean archives and never deletes -- but
    afterwards it is indistinguishable from a deliberate archive. Same shape as
    `./run-vllm --years` relabelling the library, which the test above exists to
    prevent for the wrappers; clean forwards nothing, so it needs its own.
    """
    script = _clean_sandbox(tmp_path)
    done = _run(script, "--force")
    assert done.returncode == 2, done.stdout + done.stderr
    assert not list(tmp_path.glob("output_*")), "an option was archived as a name"
    assert (tmp_path / "output" / "marker.txt").is_file(), "the live run moved"


def test_clean_help_archives_nothing(tmp_path):
    script = _clean_sandbox(tmp_path)
    done = _run(script, "--help")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "archiving" in done.stdout, done.stdout
    assert not list(tmp_path.glob("output_*")), "--help archived the run"
    assert (tmp_path / "output" / "marker.txt").is_file()


def test_clean_still_archives_a_description(tmp_path):
    """The guard must not have broken the thing the script is for."""
    script = _clean_sandbox(tmp_path)
    done = _run(script, "bioclip-taxa")
    assert done.returncode == 0, done.stdout + done.stderr
    archives = list(tmp_path.glob("output_*"))
    assert [p.name for p in archives] == ["output_001_bioclip-taxa"], archives
    assert (archives[0] / "marker.txt").is_file(), "the run was not carried over"
    assert (tmp_path / "output").is_dir() and not list((tmp_path / "output").iterdir())
