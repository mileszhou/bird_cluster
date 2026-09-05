"""The chatgpt backend, which had rotted quietly while vllm was maintained.

It had never worked since the module split -- `predict_with_gpt4o` was called
unqualified in `bird_label.py` while living in another module, so the first image
raised NameError. Behind that sat a second copy of the vLLM transport that had
drifted: a prompt from before `bird` meant class Aves, `max_tokens` too small for
the JSON now asked for, no timeout, and `json.JSONDecode_decodeError` -- an
attribute that does not exist, so every code-fenced reply became scenery/unknown.

The fix was to delete the copy: one protocol, one implementation, two endpoints.
These tests hold that shut, because the failure mode is silence.
"""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PROJECT_ROOT", str(ROOT))

from code.bird_label import _vllm_chat_completion, predict_with_vllm  # noqa: E402
from code.lib.config import model_name  # noqa: E402


class _Stub(BaseHTTPRequestHandler):
    """Records the request, replies with a code-fenced JSON body."""
    seen: dict = {}
    reply = ("```json\n"
             '{"category":"bird","label":"Grey Wagtail",'
             '"label_cn":"灰鴺鷌","confidence":0.93}'
             "\n```")

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen = {"auth": self.headers.get("Authorization"),
                           "model": body["model"],
                           "max_tokens": body.get("max_tokens"),
                           "path": self.path}
        out = json.dumps({"choices": [{"message": {"content": self.reply}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


@pytest.fixture
def stub():
    srv = HTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1", _Stub
    srv.shutdown()


@pytest.fixture
def jpg(tmp_path):
    from PIL import Image
    p = tmp_path / "test.jpg"
    Image.new("RGB", (8, 8), (30, 60, 90)).save(p)
    return p


def test_api_key_is_sent_when_given(stub, jpg):
    url, cls = stub
    predict_with_vllm(jpg, url, "gpt-4o", 0.6, 0.2, api_key="sk-test")
    assert cls.seen["auth"] == "Bearer sk-test"
    assert cls.seen["path"] == "/v1/chat/completions"


def test_no_key_means_no_header(stub, jpg):
    """A local vLLM or llama.cpp server must not receive an Authorization."""
    url, cls = stub
    predict_with_vllm(jpg, url, "some-local-model", 0.6, 0.2)
    assert cls.seen["auth"] is None


def test_code_fenced_json_is_parsed(stub, jpg):
    """The regression that mattered most: `json.JSONDecode_decodeError`.

    That attribute does not exist, so the fallback for a fenced reply raised
    AttributeError, was swallowed by the outer handler, and the row silently
    became scenery/unknown -- a wrong label rather than a visible failure.
    """
    url, _ = stub
    category, label, label_cn, conf, raw = predict_with_vllm(
        jpg, url, "gpt-4o", 0.6, 0.2, api_key="sk-test")
    assert category == "bird"
    assert label == "grey wagtail"
    assert label_cn == "灰鴺鷌"
    assert conf == 0.93
    assert json.loads(raw)["label"] == "Grey Wagtail"


def test_max_tokens_is_the_maintained_value(stub, jpg):
    """The cloud copy used 80, which truncates the JSON now asked for."""
    url, cls = stub
    predict_with_vllm(jpg, url, "gpt-4o", 0.6, 0.2, api_key="sk-test")
    assert cls.seen["max_tokens"] == 200


def test_chinese_in_the_label_field_is_moved(stub, jpg):
    """Post-processing the cloud copy never had at all."""
    url, cls = stub
    cls.reply = '{"category":"bird","label":"灰鴺鷌","label_cn":"","confidence":0.8}'
    try:
        _, label, label_cn, _, _ = predict_with_vllm(
            jpg, url, "gpt-4o", 0.6, 0.2, api_key="sk-test")
        assert label_cn == "灰鴺鷌"
        assert label == "unknown"
    finally:
        cls.reply = _Stub.__dict__["reply"]


def test_chatgpt_model_comes_from_config():
    """Not a literal in the code: models get retired, and a buried name is one
    nobody edits until a run fails."""
    assert model_name("chatgpt"), "config.toml [models] chatgpt is missing"


def test_server_backends_are_not_configured_with_a_model():
    """vllm and llama.cpp probe their server; a configured name would be a
    second, unchecked source of the same fact."""
    assert model_name("vllm") is None
    assert model_name("llama.cpp") is None
