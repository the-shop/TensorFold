"""Reply metadata for thinking replies (tensorfold.think_closed, --unclosed-think-finish), sampling fields the server
does not apply (tensorfold.ignored_params, --strict-params) and a request's thinking_budget 0 against a server default.
None of these changes a reply's tokens."""

from __future__ import annotations

import json
from typing import Any

import pytest

from http_fakes import post
from test_think_call import events, served_post

PIECES = ["<p>", "<q>", "<a>", "<|im_end|>", "The user says hi. ", "I should greet back.", "</think>", "\n\n",
          "Hello!", "\n"]
EOS = 3
UNCLOSED = [4, 5, EOS]                      # the model ends its reply inside the think block
CLOSED = [4, 5, 6, 7, 8, EOS]
HI = {"messages": [{"role": "user", "content": "hi"}]}


class ScriptTokenizer:
    eos_token_ids = {EOS}

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> list[int]:
        return [0, 1, 2]

    def decode(self, ids: list[int], **_: Any) -> str:
        return "".join(PIECES[int(t)] for t in ids)

    def encode(self, text: str, **_: Any) -> list[int]:
        return [PIECES.index(text)] if text in PIECES else []

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return PIECES.index(token) if token in PIECES else None


def script_app(script: list[int], **kwargs: Any):
    pytest.importorskip("mlx.core")
    from tensorfold.server.app import ChatApp
    from tests.lane_fakes import FakeEngine, FakeFamily

    class ScriptFamily(FakeFamily):
        """Writes ``script`` after the three-token prompt, then its last token, whatever it is fed."""

        def hidden(self, inputs: Any, cache: list[Any], parents: Any = None) -> Any:
            import mlx.core as mx
            import numpy as np

            history, out = cache[0].rows[0], []
            for token in np.array(inputs).reshape(-1).tolist():
                history.append(int(token))
                out.append(script[min(len(history) - 3, len(script) - 1)])
            return mx.array(out, dtype=mx.float32).reshape(1, -1, 1)

    family = ScriptFamily()
    options = {"enable_thinking": True, **kwargs}
    return ChatApp(None, ScriptTokenizer(), served_name="fake", lanes=1, max_rows=16, max_draft=4,
                   default_max_tokens=32, checkpoint_slots=0, use_proposer=True,
                   engine_factory=lambda model, **kw: FakeEngine(family, **kw), **options)


def both(app, body: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """(the non-streamed reply, the streamed reply's final chunk) for one request."""

    status, text = served_post(app, body)
    assert status == 200, text
    status, streamed = served_post(app, {**body, "stream": True})
    assert status == 200, streamed
    return json.loads(text), events(streamed)[-1]


# -- 1. tensorfold.think_closed ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("draft", [True, False])
def test_think_closed_says_whether_the_block_closed(draft):
    for script, closed, content in ((UNCLOSED, False, ""), (CLOSED, True, "Hello!")):
        app = script_app(script)
        try:
            reply, last = both(app, {**HI, "draft": draft})
        finally:
            app.close()
        choice = reply["choices"][0]
        assert choice["finish_reason"] == "stop" and choice["message"]["content"] == content
        assert choice["message"]["reasoning_content"].startswith("The user says hi.")
        assert reply["tensorfold"]["think_closed"] is closed
        assert last["tensorfold"]["think_closed"] is closed and last["choices"][0]["finish_reason"] == "stop"


def test_think_closed_is_null_without_thinking():
    app = script_app(UNCLOSED)
    try:
        reply, last = both(app, {**HI, "chat_template_kwargs": {"enable_thinking": False}})
    finally:
        app.close()
    assert reply["choices"][0]["message"]["content"] == "The user says hi. I should greet back."
    assert reply["tensorfold"]["think_closed"] is None and last["tensorfold"]["think_closed"] is None


def test_think_closed_helper():
    from tensorfold.server.text import CHANNEL_MARKERS, think_closed

    assert think_closed("a</think>b", thinking=True) is True
    assert think_closed("a b", thinking=True) is False
    assert think_closed("a b", thinking=False) is None
    assert think_closed("<|channel>thought\nx<channel|>y", thinking=True, markers=CHANNEL_MARKERS) is True
    assert think_closed("<|channel>thought\nx", thinking=True, markers=CHANNEL_MARKERS) is False
    assert think_closed("no channel", thinking=True, markers=CHANNEL_MARKERS) is True   # never opened: none open


# -- 2. --unclosed-think-finish ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("draft", [True, False])
def test_unclosed_think_finish_length_reports_length(draft):
    shas = {}
    for policy in ("stop", "length"):
        app = script_app(UNCLOSED, unclosed_think_finish=policy)
        try:
            reply, last = both(app, {**HI, "draft": draft})
        finally:
            app.close()
        assert reply["choices"][0]["finish_reason"] == policy == last["choices"][0]["finish_reason"]
        assert reply["tensorfold"]["think_closed"] is False
        shas[policy] = (reply["tensorfold"]["token_sha"], reply["usage"]["completion_tokens"])
    assert shas["stop"] == shas["length"]                   # the same tokens: only the reported finish differs


def test_unclosed_think_finish_length_leaves_other_replies_alone():
    for script, body in ((CLOSED, HI), (UNCLOSED, {**HI, "chat_template_kwargs": {"enable_thinking": False}})):
        app = script_app(script, unclosed_think_finish="length")
        try:
            reply, last = both(app, body)
        finally:
            app.close()
        assert reply["choices"][0]["finish_reason"] == "stop" == last["choices"][0]["finish_reason"]


def test_unclosed_think_finish_rule_and_cli():
    from tensorfold import cli
    from tensorfold.server.text import unclosed_think_finish

    assert unclosed_think_finish("stop", closed=False, content="", policy="length") == "length"
    assert unclosed_think_finish("stop", closed=False, content="", policy="stop") == "stop"
    assert unclosed_think_finish("stop", closed=False, content="<tool_call>x", policy="length") == "stop"
    assert unclosed_think_finish("stop", closed=True, content="", policy="length") == "stop"
    assert unclosed_think_finish("stop", closed=None, content="", policy="length") == "stop"
    assert unclosed_think_finish("length", closed=False, content="", policy="length") == "length"
    parser = cli.build_parser()
    assert parser.parse_args(["serve", "owner/model"]).unclosed_think_finish == "stop"
    assert parser.parse_args(["serve", "owner/model", "--unclosed-think-finish", "length"]).unclosed_think_finish \
        == "length"
    with pytest.raises(SystemExit):
        parser.parse_args(["serve", "owner/model", "--unclosed-think-finish", "error"])
    with pytest.raises(ValueError, match="unclosed_think_finish"):
        script_app(CLOSED, unclosed_think_finish="error")


# -- 3. sampling fields the server does not apply ---------------------------------------------------------------------

class _App:
    served_name = "fake"
    model_ids = ["fake"]
    max_batch_size = 1
    exact_mode = {"mode": "exact"}
    accepts_sampling = True

    def __init__(self, strict: bool = False) -> None:
        self.strict_params = strict
        self.calls = 0

    def chat(self, messages, **kwargs):
        self.calls += 1
        return {"content": "Hi", "finish_reason": "stop", "prompt_tokens": 3, "cached_tokens": 0,
                "completion_tokens": 1, "runtime": {"token_sha": "x"}}


@pytest.fixture
def fresh_warnings(monkeypatch):
    from tensorfold.server import http

    monkeypatch.setattr(http, "_WARNED", set(), raising=False)


def test_ignored_params_are_reported_and_warned_once(fresh_warnings, capsys):
    app = _App()
    body = {**HI, "presence_penalty": 0.5, "frequency_penalty": 0.3, "repetition_penalty": 1.1, "min_p": 0.05}
    for _ in range(2):
        status, text = post(app, body)
        assert status == 200
        assert json.loads(text)["tensorfold"] == {"token_sha": "x", "ignored_params": [
            "frequency_penalty", "min_p", "presence_penalty", "repetition_penalty"]}
    warnings = [line for line in capsys.readouterr().out.splitlines() if "does not apply" in line]
    assert len(warnings) == 4 and sum("presence_penalty" in line for line in warnings) == 1


def test_neutral_values_are_not_reported(fresh_warnings, capsys):
    app = _App(strict=True)
    status, text = post(app, {**HI, "presence_penalty": 0, "frequency_penalty": 0.0, "repetition_penalty": 1,
                              "logit_bias": {}, "n": 1, "logprobs": False, "min_p": None})
    assert status == 200 and "ignored_params" not in json.loads(text)["tensorfold"]
    assert "does not apply" not in capsys.readouterr().out


def test_strict_params_refuses(fresh_warnings):
    app = _App(strict=True)
    status, text = post(app, {**HI, "presence_penalty": 0.5, "logit_bias": {"42": -100}})
    error = json.loads(text)["error"]
    assert status == 400 and app.calls == 0
    assert error["type"] == "invalid_request_error" and error["code"] == "unsupported_parameter"
    assert error["param"] == "logit_bias" and "logit_bias, presence_penalty" in error["message"]
    from tensorfold import cli

    assert cli.build_parser().parse_args(["serve", "owner/model", "--strict-params"]).strict_params
    assert not cli.build_parser().parse_args(["serve", "owner/model"]).strict_params


def test_ignored_params_reach_the_chat_app_reply_streamed_or_not(fresh_warnings):
    app = script_app(CLOSED)
    try:
        plain, _ = both(app, HI)
        reply, last = both(app, {**HI, "presence_penalty": 1.5})
    finally:
        app.close()
    assert reply["tensorfold"]["ignored_params"] == ["presence_penalty"] == last["tensorfold"]["ignored_params"]
    assert "ignored_params" not in plain["tensorfold"]
    assert reply["tensorfold"]["token_sha"] == plain["tensorfold"]["token_sha"]
    assert reply["choices"][0]["message"] == plain["choices"][0]["message"]


# -- 4. thinking_budget 0 opts out of the server's default -----------------------------------------------------------

@pytest.mark.parametrize("sent, budget", [({}, 2), ({"thinking_budget": None}, 2), ({"thinking_budget": 0}, 0),
                                          ({"thinking_budget": 5}, 5)])
def test_a_request_budget_of_zero_opts_out_of_the_default(sent, budget):
    app = script_app(UNCLOSED + [EOS] * 4, thinking_budget=2)
    jobs = []
    submit = app.scheduler.submit
    app.scheduler.submit = lambda job: (jobs.append(job), submit(job))[1]
    try:
        reply, _ = both(app, {**HI, **sent})
    finally:
        app.close()
    assert jobs and all(job.think_budget == budget for job in jobs)
    # a budget of 2 closes the block at the second reply token; without one the model's reply ends inside it
    reasoning = "The user says hi." if budget == 2 else "The user says hi. I should greet back."
    assert reply["choices"][0]["message"]["reasoning_content"] == reasoning
