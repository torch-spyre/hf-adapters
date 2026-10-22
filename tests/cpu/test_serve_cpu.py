# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""``hf_adapters.serve`` HTTP surface over mocked backends (no model, no Spyre)."""

import base64
import io
import json
import sys
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest
import torch
from PIL import Image

from hf_adapters import serve


def _data_url(size=(8, 6)):
    buf = io.BytesIO()
    Image.new("RGB", size, (255, 0, 0)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class _Tokenizer:
    """Just enough tokenizer for ``TextIteratorStreamer``: one char per token id."""

    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids)


class _Processor:
    tokenizer = _Tokenizer()

    def __init__(self):
        self.conversations = []

    def apply_chat_template(self, conv, **kwargs):
        self.conversations.append(conv)
        return {
            "input_ids": torch.ones(1, 5, dtype=torch.long),
            "pixel_values": torch.zeros(1, 3, 2, 2),
        }


class _VLMModel:
    """Streams ``reply`` one token at a time, like ``hf_common.generate``."""

    reply = "Hi!"

    def __init__(self):
        self.kwargs = None
        self.pixel_dtype = None

    def generate(self, input_ids, pixel_values, streamer, **kwargs):
        self.kwargs, self.pixel_dtype = kwargs, pixel_values.dtype
        streamer.put(input_ids)
        new = [ord(c) for c in self.reply][: kwargs["max_new_tokens"]]
        for t in new:
            streamer.put(torch.tensor([t]))
        streamer.end()
        return torch.cat([input_ids, torch.tensor([new])], dim=1)


class _ST:
    def encode(self, items, convert_to_numpy):
        return np.array(
            [[1.0, 0.0] if isinstance(x, str) else [0.0, float(x.width)] for x in items]
        )

    def tokenize(self, texts):
        return {"attention_mask": torch.ones(len(texts), 4)}


class _TextST(_ST):
    modalities = ["text"]

    def encode(self, items, convert_to_numpy):
        if any(not isinstance(x, str) for x in items):
            raise ValueError("Modality 'image' is not supported")
        return super().encode(items, convert_to_numpy)


def _vlm(monkeypatch):
    def init(self, model_path, trust_remote_code):
        self.dtype, self.model, self.processor = (
            torch.float16,
            _VLMModel(),
            _Processor(),
        )

    monkeypatch.setattr(serve.VLM, "__init__", init)


def _clip(monkeypatch, st=_ST):
    def init(self, model_path, trust_remote_code):
        self.st = st()

    monkeypatch.setattr(serve.CLIP, "__init__", init)


class _LMTokenizer(_Tokenizer):
    def __init__(self, template):
        self.chat_template = template
        self.seen = []

    def apply_chat_template(self, conv, **kwargs):
        self.seen.append(("template", conv))
        return {"input_ids": torch.ones(1, 4, dtype=torch.long), "extra": 1}

    def __call__(self, text, return_tensors):
        self.seen.append(("plain", text))
        return {
            "input_ids": torch.ones(1, 2, dtype=torch.long),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
        }


class _LMModel(_VLMModel):
    def generate(self, input_ids, streamer, attention_mask=None, **kwargs):
        self.kwargs = kwargs
        streamer.put(input_ids)
        new = [ord(c) for c in self.reply][: kwargs["max_new_tokens"]]
        for t in new:
            streamer.put(torch.tensor([t]))
        streamer.end()
        return torch.cat([input_ids, torch.tensor([new])], dim=1)


class _RerankTokenizer:
    def __call__(self, queries, text_pair, **kwargs):
        self.pairs = list(zip(queries, text_pair))
        n = len(queries)
        return {
            "input_ids": torch.ones(n, 3, dtype=torch.long),
            "attention_mask": torch.ones(n, 3, dtype=torch.long),
        }


class _RerankModel:
    def __init__(self, logits):
        self.logits = torch.tensor(logits)

    def __call__(self, input_ids, attention_mask, return_dict):
        return type("Out", (), {"logits": self.logits})()


def _lm(monkeypatch, template="{{ messages }}"):
    def init(self, model_path, trust_remote_code):
        self.model, self.tokenizer = _LMModel(), _LMTokenizer(template)

    monkeypatch.setattr(serve.CausalLM, "__init__", init)


def _rerank(monkeypatch, logits=((0.0,), (2.0,), (-3.0,))):
    def init(self, model_path, trust_remote_code):
        self.model, self.tokenizer = _RerankModel(list(logits)), _RerankTokenizer()

    monkeypatch.setattr(serve.Rerank, "__init__", init)


@pytest.fixture
def start(monkeypatch):
    servers = []

    def _start(task, load=True, st=_ST, template="{{ messages }}"):
        if task == "clip":
            _clip(monkeypatch, st)
        elif task == "chat":
            _lm(monkeypatch, template)
        elif task == "rerank":
            _rerank(monkeypatch)
        else:
            _vlm(monkeypatch)
        srv = serve.Server(("127.0.0.1", 0), task, "org/model", "m")
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        if load:
            srv.load("unused", False)
        return srv, f"http://127.0.0.1:{srv.server_address[1]}"

    yield _start
    for srv in servers:
        srv.shutdown()
        srv.server_close()


def _get(url):
    try:
        with urllib.request.urlopen(url) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _post(url, body):
    req = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.headers.get("Content-Type"), r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type"), e.read().decode()


def _chat(text="What is this?", **extra):
    content = [
        {"type": "image_url", "image_url": {"url": _data_url()}},
        {"type": "text", "text": text},
    ]
    return {"model": "m", "messages": [{"role": "user", "content": content}], **extra}


def test_health_waits_for_load_and_warmup(start):
    srv, url = start("vlm", load=False)
    assert _get(url + "/health")[0] == 503
    assert _post(url + "/v1/chat/completions", _chat())[0] == 503
    srv.load("unused", False)
    assert _get(url + "/health") == (200, {"status": "ok"})
    # Warmup ran one short greedy generation.
    assert srv.backend.model.kwargs == {"max_new_tokens": 2, "do_sample": False}


def test_models_lists_served_name(start):
    _, url = start("clip")
    status, body = _get(url + "/v1/models")
    assert status == 200
    assert body["data"][0]["id"] == "m"
    assert body["data"][0]["root"] == "org/model"
    assert body["data"][0]["task"] == "clip"


def test_wrong_task_route_is_404(start):
    _, url = start("clip")
    assert _post(url + "/v1/chat/completions", _chat())[0] == 404


def test_chat_completion_converts_image_parts(start):
    srv, url = start("vlm")
    status, ctype, text = _post(
        url + "/v1/chat/completions", _chat(max_tokens=8, temperature=0.5)
    )
    assert status == 200 and ctype == "application/json"
    body = json.loads(text)
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "Hi!"}
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"] == {
        "prompt_tokens": 5,
        "completion_tokens": 3,
        "total_tokens": 8,
    }
    parts = srv.backend.processor.conversations[-1][0]["content"]
    assert isinstance(parts[0]["image"], Image.Image) and parts[0]["image"].size == (
        8,
        6,
    )
    assert parts[1] == {"type": "text", "text": "What is this?"}
    assert srv.backend.model.pixel_dtype == torch.float16
    assert srv.backend.model.kwargs == {
        "max_new_tokens": 8,
        "do_sample": True,
        "temperature": 0.5,
        "top_p": 1.0,
    }


def test_chat_stream_emits_tokens_finish_and_usage(start):
    _, url = start("vlm")
    status, ctype, text = _post(
        url + "/v1/chat/completions",
        _chat(max_tokens=2, stream=True, stream_options={"include_usage": True}),
    )
    assert status == 200 and ctype == "text/event-stream"
    events = [line[6:] for line in text.split("\n\n") if line]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    deltas = [c["choices"][0]["delta"] for c in chunks if c["choices"]]
    assert deltas[0] == {"role": "assistant", "content": ""}
    assert "".join(d.get("content", "") for d in deltas) == "Hi"
    assert chunks[-2]["choices"][0]["finish_reason"] == "length"
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"]["completion_tokens"] == 2


@pytest.mark.parametrize(
    "messages, error",
    [
        ([], "messages is required"),
        ([{"role": "user", "content": "no image"}], "at least one image_url"),
        (
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "http://x/a.png"}}
                    ],
                }
            ],
            "data: URLs",
        ),
    ],
)
def test_chat_rejects_bad_requests(start, messages, error):
    _, url = start("vlm")
    status, _, text = _post(url + "/v1/chat/completions", {"messages": messages})
    assert status == 400
    assert error in json.loads(text)["error"]["message"]


def test_embeddings_keep_input_order_across_text_and_images(start):
    _, url = start("clip")
    image = {"type": "image_url", "image_url": {"url": _data_url((7, 7))}}
    status, _, text = _post(
        url + "/v1/embeddings",
        {"input": ["a dog", image, {"type": "text", "text": "a cat"}]},
    )
    assert status == 200
    body = json.loads(text)
    assert [d["index"] for d in body["data"]] == [0, 1, 2]
    assert [d["embedding"] for d in body["data"]] == [
        [1.0, 0.0],
        [0.0, 7.0],
        [1.0, 0.0],
    ]
    assert body["usage"] == {"prompt_tokens": 8, "total_tokens": 8}


def test_embeddings_accept_a_single_string(start):
    _, url = start("clip")
    status, _, text = _post(url + "/v1/embeddings", {"input": "hello"})
    assert status == 200
    assert len(json.loads(text)["data"]) == 1


def test_text_only_embedder_loads_and_refuses_images(start):
    _, url = start("clip", st=_TextST)
    image = {"type": "image_url", "image_url": {"url": _data_url((7, 7))}}
    assert _post(url + "/v1/embeddings", {"input": "hello"})[0] == 200
    status, _, text = _post(url + "/v1/embeddings", {"input": ["hi", image]})
    assert status == 400
    assert "embeds text only" in json.loads(text)["error"]["message"]


def test_yolo_reports_unsupported_build(monkeypatch):
    # A None entry makes the import raise ImportError, as in a build without the module.
    monkeypatch.setitem(sys.modules, "hf_adapters.ultralytics_yolo", None)
    with pytest.raises(SystemExit, match="unsupported in this build"):
        serve.YOLO("yolov8n-seg.pt", False)


def _text_chat(text="Hi", **extra):
    return {"model": "m", "messages": [{"role": "user", "content": text}], **extra}


def test_causal_lm_chat_uses_the_chat_template(start):
    srv, url = start("chat")
    status, _, text = _post(url + "/v1/chat/completions", _text_chat(max_tokens=8))
    assert status == 200
    body = json.loads(text)
    assert body["choices"][0]["message"]["content"] == "Hi!"
    assert body["usage"] == {
        "prompt_tokens": 4,
        "completion_tokens": 3,
        "total_tokens": 7,
    }
    assert srv.backend.tokenizer.seen[-1] == (
        "template",
        [{"role": "user", "content": "Hi"}],
    )
    assert srv.backend.model.kwargs["max_new_tokens"] == 8


def test_causal_lm_without_a_chat_template_gets_a_plain_prompt(start):
    srv, url = start("chat", template=None)
    status, _, _ = _post(
        url + "/v1/chat/completions",
        {
            "messages": [
                {"role": "system", "content": "A"},
                {"role": "user", "content": "B"},
            ]
        },
    )
    assert status == 200
    assert srv.backend.tokenizer.seen[-1] == ("plain", "A\nB")


def test_causal_lm_refuses_images(start):
    _, url = start("chat")
    status, _, text = _post(url + "/v1/chat/completions", _chat())
    assert status == 400 and "text content only" in json.loads(text)["error"]["message"]


def test_rerank_orders_by_relevance_and_keeps_the_input_index(start):
    srv, url = start("rerank")
    status, _, text = _post(
        url + "/v1/rerank", {"query": "q", "documents": ["a", "b", "c"], "top_n": 2}
    )
    assert status == 200
    results = json.loads(text)["results"]
    assert [(r["index"], r["document"]["text"]) for r in results] == [
        (1, "b"),
        (0, "a"),
    ]
    assert results[0]["relevance_score"] == pytest.approx(0.8808, abs=1e-4)
    assert srv.backend.tokenizer.pairs == [("q", "a"), ("q", "b"), ("q", "c")]


@pytest.mark.parametrize(
    "body, error",
    [
        ({"documents": ["a"]}, "query is required"),
        ({"query": "q", "documents": []}, "non-empty list"),
        ({"query": "q", "documents": ["a", 1]}, "non-empty list"),
    ],
)
def test_rerank_rejects_bad_requests(start, body, error):
    _, url = start("rerank")
    status, _, text = _post(url + "/v1/rerank", body)
    assert status == 400 and error in json.loads(text)["error"]["message"]
