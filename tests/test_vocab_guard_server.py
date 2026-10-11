"""Prompts with token ids past the model's embedding are refused (HTTP 400) before any work; a load-time warning names the gap."""

import types

import pytest

from tensorfold.server.app import model_vocab, warn_vocab_mismatch
from tensorfold.server.http import RequestError
from tests.test_lane_server import FakeTokenizer, make_app
from tests.test_server_openai_compat import post_json, serve_fake


def _ids(app, text):
    return app.render([{"role": "user", "content": text}])[0]


def test_out_of_vocab_prompt_is_refused_before_submission(monkeypatch):
    app = make_app()
    try:
        app.model_vocab = max(_ids(app, "ab")) + 1
        assert max(_ids(app, "abw")) >= app.model_vocab
        monkeypatch.setattr(app.scheduler, "submit", lambda job: pytest.fail("job submitted"))
        with pytest.raises(RequestError, match=rf"token id {max(_ids(app, 'abw'))}.*vocabulary of {app.model_vocab}"):
            app.chat([{"role": "user", "content": "abw"}], max_tokens=2)
        assert not app.engine.prefill_calls
    finally:
        app.close()


def test_http_out_of_vocab_is_400_and_next_request_works():
    app = make_app()
    app.model_vocab = max(_ids(app, "ab")) + 1
    server = serve_fake(app)
    try:
        for stream in (False, True):
            status, body = post_json(server, "/v1/chat/completions",
                                     {"messages": [{"role": "user", "content": "abw"}], "max_tokens": 2,
                                      "stream": stream})
            # a stream has sent its headers already: the refusal is its invalid_request_error event, as for any
            # other RequestError (context length)
            assert status == (200 if stream else 400) and "outside this model's vocabulary" in body
            assert "invalid_request_error" in body
        assert not app.engine.prefill_calls
        status, body = post_json(server, "/v1/chat/completions",
                                 {"messages": [{"role": "user", "content": "ab"}], "max_tokens": 2})
        assert status == 200
    finally:
        server.shutdown()
        server.server_close()
        app.close()


def test_model_vocab_reads_args():
    assert model_vocab(None) == 0
    assert model_vocab(types.SimpleNamespace(args=types.SimpleNamespace(vocab_size=248320))) == 248320


def test_load_warning_only_when_tokenizer_exceeds_vocab(capsys):
    class Sized(FakeTokenizer):
        def __len__(self):
            return 248077

    warn_vocab_mismatch(Sized(), 248320)
    warn_vocab_mismatch(Sized(), 0)
    assert capsys.readouterr().out == ""
    warn_vocab_mismatch(Sized(), 65536)
    out = capsys.readouterr().out
    assert "warning" in out and "248,077" in out and "65,536" in out and "HTTP 400" in out
