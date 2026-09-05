"""A run that labels nothing must not report success.

`process_folder` catches every per-image failure, logs it, and breaks the loop;
the caller printed "Run complete" regardless and exited 0. That is how the
chatgpt backend stayed broken for months -- it raised on the first image of
every run anyone tried, and every run said it had finished.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PY = ROOT / ".venv" / "bin" / "python"

STUB = '''
import json, http.server, sys
MODE = sys.argv[1]
class H(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        if MODE == "broken":
            self.send_response(500); self.send_header("Content-Length","0"); self.end_headers(); return
        c = json.dumps({"category":"bird","label":"Grey Wagtail",
                        "label_cn":"x","confidence":0.9})
        out = json.dumps({"choices":[{"message":{"content":c}}]}).encode()
        self.send_response(200); self.send_header("Content-Type","application/json")
        self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)
    def log_message(self,*a): pass
srv = http.server.HTTPServer(("127.0.0.1", 0), H)
print(srv.server_port, flush=True)
srv.serve_forever()
'''


@pytest.fixture
def data(tmp_path):
    from PIL import Image
    d = tmp_path / "data"
    (d / "jpg" / "Photos-16" / "trip").mkdir(parents=True)
    (d / "xmp").mkdir()
    Image.new("RGB", (8, 8)).save(d / "jpg" / "Photos-16" / "trip" / "a.jpg")
    return d


def _run(mode, data, out, tmp_path):
    stub = tmp_path / "stub.py"
    stub.write_text(STUB)
    proc = subprocess.Popen([str(PY), str(stub), mode], stdout=subprocess.PIPE, text=True)
    try:
        port = proc.stdout.readline().strip()
        env = {**os.environ, "OPENAI_BASE_URL": f"http://127.0.0.1:{port}/v1",
               "OPENAI_API_KEY": "sk-test", "PROJECT_ROOT": str(ROOT)}
        return subprocess.run(
            [str(PY), "-m", "code.bird_label", "--approach", "openai",
             "--data-dir", str(data), "--output-dir", str(out)],
            cwd=ROOT, env=env, capture_output=True, text=True)
    finally:
        proc.kill()


def test_a_run_that_labels_nothing_fails(data, tmp_path):
    out = tmp_path / "out"
    done = _run("broken", data, out, tmp_path)
    assert done.returncode != 0, "a run that labelled nothing reported success"
    assert "labelled nothing" in done.stdout, done.stdout


def test_nothing_is_checkpointed_when_the_model_never_answered(data, tmp_path):
    """So a re-run retries, rather than skipping photos it never labelled."""
    out = tmp_path / "out"
    _run("broken", data, out, tmp_path)
    checkpoint = out / "processed.txt"
    assert not checkpoint.exists() or not checkpoint.read_text().strip()


def test_no_row_is_written_for_an_unanswered_photo(data, tmp_path):
    """The defaults are `scenery/unknown/0.00` -- a plausible label for a real
    photo. Writing them would be a wrong answer, not a missing one."""
    out = tmp_path / "out"
    _run("broken", data, out, tmp_path)
    csv_path = out / "bird_identification_output.csv"
    rows = [l for l in csv_path.read_text(encoding="utf-8-sig").splitlines()[1:] if l.strip()]
    assert rows == [], rows


def test_a_working_run_succeeds_and_says_how_many(data, tmp_path):
    out = tmp_path / "out"
    done = _run("ok", data, out, tmp_path)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "1 images labelled" in done.stdout, done.stdout
    assert (out / "processed.txt").read_text().strip()


def test_limit_caps_the_run_and_the_dry_run_agrees(data, tmp_path):
    """`--limit` exists so a paid model can be tried for pennies.

    It is not scope: which images you get depends on walk order, so a run
    narrowed this way cannot be described or reproduced. `--include-from` is the
    only way to name a population -- `--years` was removed for being a second
    one. The dry run must report the capped figure, or the flag would be
    invisible exactly where it is being checked.
    """
    from PIL import Image
    trip = data / "jpg" / "Photos-16" / "trip"
    for n in range(4):
        Image.new("RGB", (8, 8)).save(trip / f"extra{n}.jpg")

    out = tmp_path / "out"
    stub = tmp_path / "stub.py"
    stub.write_text(STUB)
    proc = subprocess.Popen([str(PY), str(stub), "ok"], stdout=subprocess.PIPE, text=True)
    try:
        port = proc.stdout.readline().strip()
        env = {**os.environ, "OPENAI_BASE_URL": f"http://127.0.0.1:{port}/v1",
               "OPENAI_API_KEY": "sk-test", "PROJECT_ROOT": str(ROOT)}
        base = [str(PY), "-m", "code.bird_label", "--approach", "openai",
                "--data-dir", str(data), "--output-dir", str(out), "--limit", "2"]
        dry = subprocess.run(base + ["--dry-run"], cwd=ROOT, env=env,
                             capture_output=True, text=True)
        assert "2 would be sent to the model" in dry.stdout, dry.stdout
        done = subprocess.run(base, cwd=ROOT, env=env, capture_output=True, text=True)
    finally:
        proc.kill()

    assert done.returncode == 0, done.stdout + done.stderr
    assert "2 images labelled" in done.stdout, done.stdout
    rows = [l for l in (out / "bird_identification_output.csv")
            .read_text(encoding="utf-8-sig").splitlines()[1:] if l.strip()]
    assert len(rows) == 2, rows


def test_the_batch_path_writes_rows(data, tmp_path):
    """The concurrent path, which had no end-to-end cover and was broken by it.

    `write_row` gained a `label_sci` argument; the single-image call was updated
    and the batch one was not, so `--batch-size 8` died on the first batch with a
    missing positional argument. Nothing caught it because every test drove the
    serial path -- and the batch path is the one a real run of any size uses.
    """
    from PIL import Image
    trip = data / "jpg" / "Photos-16" / "trip"
    for n in range(7):
        Image.new("RGB", (8, 8)).save(trip / f"extra{n}.jpg")

    out = tmp_path / "out"
    stub = tmp_path / "stub.py"
    stub.write_text(STUB)
    proc = subprocess.Popen([str(PY), str(stub), "ok"], stdout=subprocess.PIPE, text=True)
    try:
        port = proc.stdout.readline().strip()
        env = {**os.environ, "OPENAI_BASE_URL": f"http://127.0.0.1:{port}/v1",
               "OPENAI_API_KEY": "sk-test", "PROJECT_ROOT": str(ROOT)}
        done = subprocess.run(
            [str(PY), "-m", "code.bird_label", "--approach", "openai",
             "--data-dir", str(data), "--output-dir", str(out), "--batch-size", "4"],
            cwd=ROOT, env=env, capture_output=True, text=True)
    finally:
        proc.kill()

    assert done.returncode == 0, done.stdout + done.stderr
    assert "8 images labelled" in done.stdout, done.stdout
    rows = [l for l in (out / "bird_identification_output.csv")
            .read_text(encoding="utf-8-sig").splitlines()[1:] if l.strip()]
    assert len(rows) == 8, rows
    # and the column the arity bug was about is populated, not shifted
    import csv as _csv
    with open(out / "bird_identification_output.csv", encoding="utf-8-sig") as fh:
        first = next(_csv.DictReader(fh))
    assert first["category"] == "bird", first
    assert first["confidence"] == "0.90", first
