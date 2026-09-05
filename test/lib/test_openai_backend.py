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
    category, label, label_cn, _sci, conf, raw = predict_with_vllm(
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
        _, label, label_cn, *_ = predict_with_vllm(
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
    category, label, *_ = predict_with_vllm(
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
        *_, raw = predict_with_vllm(jpg, url, "gpt-5", 0.6, 0.2, api_key="sk")
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
    category, label, *_ = predict_with_vllm(
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


# --- one bird, one string ---------------------------------------------------

def test_chinese_names_are_normalised_to_simplified():
    """A model answers in whichever script it likes.

    `普通翠鳥` against `普通翠鸟` is one bird under two keywords: a photo manager
    lists it twice and any join on the label splits it. Same fragmentation the
    confidence used to cause in a keyword, so it gets the same answer --
    normalise at the boundary rather than teaching every consumer about it.
    """
    from code.bird_label import to_simplified
    assert to_simplified("普通翠鳥") == "普通翠鸟"
    assert to_simplified("普通翠鸟") == "普通翠鸟"


def test_normalising_leaves_non_chinese_alone():
    from code.bird_label import to_simplified
    for text in ("Grey Wagtail", "", "mrs. gould's sunbird"):
        assert to_simplified(text) == text


def test_a_traditional_reply_is_stored_simplified(stub, jpg):
    """End to end: the model may answer in traditional, the CSV holds one form."""
    url, cls = stub
    cls.reply = ('{"category":"bird","label":"Common Kingfisher",'
                 '"label_cn":"普通翠鳥","confidence":0.9}')
    try:
        _, _, label_cn, *_ = predict_with_vllm(
            jpg, url, "gpt-4o", 0.6, 0.2, api_key="sk-test")
    finally:
        cls.reply = _Stub.__dict__["reply"]
    assert label_cn == "普通翠鸟", label_cn


class _Silent(BaseHTTPRequestHandler):
    """Accepts max_tokens without complaint, then starves on it.

    OpenRouter's shape, and the case the 400-driven adaptation missed entirely:
    the request succeeds, the content is empty, `finish_reason` is "length".
    """
    calls: list = []

    def do_POST(self):
        b = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).calls.append({k: b[k] for k in
                                 ("max_tokens", "max_completion_tokens",
                                  "reasoning_effort") if k in b})
        budget = b.get("max_completion_tokens") or b.get("max_tokens") or 0
        if budget < 1000:
            body = {"choices": [{"message": {"content": ""}, "finish_reason": "length"}],
                    "usage": {"completion_tokens_details": {"reasoning_tokens": budget}}}
        else:
            c = json.dumps({"category": "bird", "label": "Common Kingfisher",
                            "label_cn": "普通翠鸟", "confidence": 0.9})
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
def silent():
    import code.bird_label as bl
    bl._PARAM_FIXES.clear()
    _Silent.calls = []
    srv = HTTPServer(("127.0.0.1", 0), _Silent)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1", _Silent
    srv.shutdown()
    bl._PARAM_FIXES.clear()


def test_a_starved_reply_adapts_even_without_a_rejection(silent, jpg):
    """The 400 route cannot help when nothing is refused.

    An endpoint that accepts `max_tokens: 200` and then lets a reasoning model
    spend all of it thinking returns a *successful* response with empty content.
    Adapting only on rejection left that as broken as before -- just with a
    better error message. The observed failure has to be a trigger too.
    """
    url, cls = silent
    category, label, *_ = predict_with_vllm(
        jpg, url, "qwen/some-flash", 0.6, 0.2, api_key="sk-test")
    assert (category, label) == ("bird", "common kingfisher")
    assert cls.calls[0] == {"max_tokens": 200}, cls.calls
    assert cls.calls[-1]["reasoning_effort"] == "minimal"
    assert cls.calls[-1]["max_tokens"] >= 1000


def test_the_budget_fix_is_learned_once(silent, jpg):
    url, cls = silent
    for _ in range(3):
        predict_with_vllm(jpg, url, "qwen/some-flash", 0.6, 0.2, api_key="sk-test")
    # 2 for the first image (starved, then retried), 1 for each after.
    assert len(cls.calls) == 4, cls.calls


class _RateLimited(BaseHTTPRequestHandler):
    """429 for the first two requests, then answers. A gateway under load."""
    seen = 0
    fail_for = 2

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        type(self).seen += 1
        if type(self).seen <= type(self).fail_for:
            self.send_response(429)
            self.send_header("Retry-After", "0")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        c = json.dumps({"category": "bird", "label": "Common Kingfisher",
                        "label_cn": "普通翠鸟", "confidence": 0.9})
        out = json.dumps({"choices": [{"message": {"content": c},
                                       "finish_reason": "stop"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


@pytest.fixture
def limited():
    _RateLimited.seen = 0
    _RateLimited.fail_for = 2
    srv = HTTPServer(("127.0.0.1", 0), _RateLimited)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1", _RateLimited
    srv.shutdown()


def test_a_rate_limit_is_waited_out_not_treated_as_failure(limited, jpg):
    """429 is a queue, not a refusal.

    On a shared gateway across tens of thousands of images it is a certainty.
    Treating it as permanent drops photos from a paid run -- they are at least
    left uncheckpointed, so a re-run retries them, but a long run bleeds steadily
    and still reports completion.
    """
    url, cls = limited
    category, label, *_ = predict_with_vllm(
        jpg, url, "qwen/flash", 0.6, 0.2, api_key="sk-test")
    assert (category, label) == ("bird", "common kingfisher")
    assert cls.seen == 3, cls.seen


def test_a_persistent_rate_limit_still_fails_rather_than_hanging(limited, jpg):
    import code.bird_label as bl
    url, cls = limited
    cls.fail_for = 99
    *_, raw = predict_with_vllm(jpg, url, "qwen/flash", 0.6, 0.2, api_key="sk-test")
    assert bl.prediction_failed(raw), "a permanent 429 must be recorded as a failure"
    assert cls.seen == bl.TRANSIENT_RETRIES, cls.seen


def test_a_real_refusal_is_not_retried(jpg):
    """4xx other than 429 is a refusal; retrying it wastes a paid call."""
    class _Refuse(_RateLimited):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            type(self).seen += 1
            self.send_response(403)
            self.send_header("Content-Length", "0")
            self.end_headers()

    _Refuse.seen = 0
    srv = HTTPServer(("127.0.0.1", 0), _Refuse)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        predict_with_vllm(jpg, f"http://127.0.0.1:{srv.server_port}/v1",
                          "m", 0.6, 0.2, api_key="sk")
    finally:
        srv.shutdown()
    assert _Refuse.seen == 1, _Refuse.seen


class _Truncating(BaseHTTPRequestHandler):
    """Cuts the answer off mid-JSON at a small budget.

    The commoner shape than an empty reply, and the one that slipped through:
    the model reasons for most of its allowance, starts answering, and is cut off
    leaving `..."confidence":0` with no closing brace.
    """
    budgets: list = []
    FULL = json.dumps({"category": "bird", "label": "Eurasian Spoonbill",
                       "label_cn": "白琵鹭", "label_sci": "Platalea leucorodia",
                       "confidence": 0.94})

    def do_POST(self):
        b = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        budget = b.get("max_completion_tokens") or b.get("max_tokens") or 0
        type(self).budgets.append(budget)
        if budget < 1000:
            body = {"choices": [{"message": {"content": self.FULL[:70]},
                                 "finish_reason": "length"}],
                    "usage": {"completion_tokens_details": {"reasoning_tokens": budget}}}
        else:
            body = {"choices": [{"message": {"content": self.FULL},
                                 "finish_reason": "stop"}]}
        out = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


def test_a_truncated_reply_adapts_like_an_empty_one(jpg):
    """`finish_reason: length` is the signal, whatever the content is.

    Requiring an *empty* answer missed this case entirely -- and adding
    `label_sci` lengthened every reply, which is what began pushing answers over
    the edge on a real run.
    """
    import code.bird_label as bl
    bl._PARAM_FIXES.clear()
    _Truncating.budgets = []
    srv = HTTPServer(("127.0.0.1", 0), _Truncating)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        category, label, _, sci, *_ = predict_with_vllm(
            jpg, f"http://127.0.0.1:{srv.server_port}/v1", "luna", 0.6, 0.2,
            api_key="sk-test")
    finally:
        srv.shutdown()
        bl._PARAM_FIXES.clear()
    assert (category, label) == ("bird", "eurasian spoonbill")
    assert sci == "Platalea leucorodia"
    assert _Truncating.budgets == [200, 2000], _Truncating.budgets
