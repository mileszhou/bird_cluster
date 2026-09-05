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
    assert model_name("openai"), "config.toml [models] openai is missing"


def test_server_backends_are_not_configured_with_a_model():
    """vllm and llama.cpp probe their server; a configured name would be a
    second, unchecked source of the same fact."""
    assert model_name("vllm") is None
    assert model_name("llama.cpp") is None


# --- models that reject the older parameter spelling ------------------------

class _Strict(BaseHTTPRequestHandler):
    """Behaves like GPT-5 / the o-series: rejects max_tokens, then temperature."""
    calls: list = []

    def _err(self, msg):
        out = json.dumps({"error": {"message": msg}}).encode()
        self.send_response(400)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).calls.append(sorted(
            k for k in body if k in ("max_tokens", "max_completion_tokens", "temperature")))
        if "max_tokens" in body:
            return self._err("Unsupported parameter: 'max_tokens' is not supported "
                             "with this model. Use 'max_completion_tokens' instead.")
        if body.get("temperature") not in (None, 1):
            return self._err("Unsupported value: 'temperature' does not support 0.0 "
                             "with this model. Only the default (1) value is supported.")
        content = json.dumps({"category": "bird", "label": "Grey Wagtail",
                              "label_cn": "灰鴺鷌", "confidence": 0.9})
        out = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


@pytest.fixture
def strict():
    import code.bird_label as bl
    bl._PARAM_FIXES.clear()
    _Strict.calls = []
    srv = HTTPServer(("127.0.0.1", 0), _Strict)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1", _Strict
    srv.shutdown()
    bl._PARAM_FIXES.clear()


def test_reasoning_model_parameters_are_corrected(strict, jpg):
    """A hardcoded list of which models want which spelling goes stale.

    GPT-5 and the o-series reject `max_tokens` for `max_completion_tokens`, and
    reject a non-default temperature. The endpoint says so; this reads its 400
    and adapts, so the *first* image succeeds rather than being spent learning.
    """
    url, cls = strict
    category, label, _, _, _ = predict_with_vllm(
        jpg, url, "gpt-5", 0.6, 0.2, api_key="sk-test")
    assert (category, label) == ("bird", "grey wagtail")
    assert cls.calls[0] == ["max_tokens", "temperature"]
    assert cls.calls[-1] == ["max_completion_tokens"]


def test_the_correction_is_learned_once_per_run(strict, jpg):
    """Not re-derived per image: that would triple the request count."""
    url, cls = strict
    for _ in range(3):
        predict_with_vllm(jpg, url, "gpt-5", 0.6, 0.2, api_key="sk-test")
    # 3 for the first image (two rejections plus the success), 1 for each after.
    assert len(cls.calls) == 5, cls.calls


def test_a_400_we_cannot_fix_is_reported_not_swallowed(strict, jpg):
    """An unrecognised rejection must surface as a failed prediction.

    Returning the scenery/unknown defaults would put a plausible wrong label on
    a real photograph and checkpoint it.
    """
    import code.bird_label as bl
    url, cls = strict
    cls.calls = []

    def _bad(detail, payload):
        return None
    original, bl._param_fix = bl._param_fix, _bad
    try:
        _, _, _, _, raw = predict_with_vllm(jpg, url, "gpt-5", 0.6, 0.2, api_key="sk")
    finally:
        bl._param_fix = original
    assert bl.prediction_failed(raw), "an unfixable 400 must be recorded as a failure"


# --- reasoning models spend the budget before they answer -------------------

class _Reasoning(BaseHTTPRequestHandler):
    """GPT-5 shaped: wants max_completion_tokens, and returns nothing unless
    given room *and* a low effort -- reasoning is charged and comes first."""

    def _err(self, msg):
        out = json.dumps({"error": {"message": msg}}).encode()
        self.send_response(400)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_POST(self):
        b = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if "max_tokens" in b:
            return self._err("Unsupported parameter: 'max_tokens' is not supported "
                             "with this model. Use 'max_completion_tokens' instead.")
        if b.get("temperature") not in (None, 1):
            return self._err("Unsupported value: 'temperature' does not support 0.0.")
        budget, effort = b.get("max_completion_tokens", 0), b.get("reasoning_effort")
        if budget < 1000 or effort not in ("minimal", "low"):
            body = {"choices": [{"message": {"content": ""}, "finish_reason": "length"}],
                    "usage": {"completion_tokens_details": {"reasoning_tokens": budget}}}
        else:
            c = json.dumps({"category": "bird", "label": "Grey Wagtail",
                            "label_cn": "x", "confidence": 0.9})
            body = {"choices": [{"message": {"content": c}, "finish_reason": "stop"}]}
        out = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


@pytest.fixture
def reasoning():
    import code.bird_label as bl
    bl._PARAM_FIXES.clear()
    srv = HTTPServer(("127.0.0.1", 0), _Reasoning)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()
    bl._PARAM_FIXES.clear()


def test_a_reasoning_model_gets_room_and_a_low_effort(reasoning, jpg):
    """Renaming the parameter is not enough on its own.

    `max_completion_tokens` covers reasoning *plus* output, and reasoning comes
    first: at the old 200 a GPT-5 run spent the whole allowance thinking and
    returned an empty string, which surfaced as "No JSON found" over a blank
    response. Raising the budget alone would be wrong too -- reasoning tokens are
    billed, so a generous budget across 49,000 images is money spent on
    deliberation nobody reads. Both parts, or neither works.
    """
    category, label, _, _, _ = predict_with_vllm(
        jpg, reasoning, "gpt-5", 0.6, 0.2, api_key="sk-test")
    assert (category, label) == ("bird", "grey wagtail")


def test_an_empty_reply_names_the_token_limit(jpg):
    """Not "No JSON found": that sends the reader hunting a parsing bug."""
    import code.bird_label as bl

    class _Starved(_Reasoning):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            out = json.dumps({
                "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
                "usage": {"completion_tokens_details": {"reasoning_tokens": 2000}}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    srv = HTTPServer(("127.0.0.1", 0), _Starved)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        *_, raw = predict_with_vllm(jpg, f"http://127.0.0.1:{srv.server_port}/v1",
                                    "gpt-5", 0.6, 0.2, api_key="sk-test")
    finally:
        srv.shutdown()
    why = bl.prediction_failed(raw)
    assert why and "token limit" in why and "reasoning tokens" in why, why
