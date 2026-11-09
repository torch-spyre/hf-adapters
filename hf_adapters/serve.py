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

"""Minimal OpenAI-style HTTP server for one hf-adapters model on Spyre.

    python -m hf_adapters.serve --task vlm --model ibm-granite/granite-vision-4.1-4b

Every task serves ``GET /health`` (200 once the model is loaded and warmed up)
and ``GET /v1/models``, plus one task endpoint:

- ``vlm``:  ``POST /v1/chat/completions`` with OpenAI content parts; images are
  ``image_url`` parts holding base64 ``data:`` URLs. Supports ``stream`` and
  ``stream_options.include_usage``.
- ``chat``: ``POST /v1/chat/completions`` over a causal LM, text parts only; a
  checkpoint without a chat template is sent the messages joined as a plain prompt.
- ``rerank``: ``POST /v1/rerank`` ``{"query": ..., "documents": [...], "top_n": n}``
  -> ``results`` of ``index``, ``document`` and ``relevance_score``, best first.
- ``clip``: ``POST /v1/embeddings``; each ``input`` item is a string, a
  ``{"type": "text", "text": ...}`` part or an ``image_url`` part.
- ``yolo``: ``POST /v1/segment`` ``{"image": "data:...", "conf": 0.25,
  "iou": 0.45}`` -> boxes, classes, scores and mask polygons in image pixels.
  Needs ``hf_adapters.ultralytics_yolo``, which not every build ships.

Requests run one at a time: the model owns a single card and its generation
state is not reentrant.
"""

import argparse
import base64
import io
import json
import os
import sys
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from PIL import Image

MAX_BODY_BYTES = 64 << 20
DEFAULT_MAX_TOKENS = 256


class BadRequestError(ValueError):
    pass


def decode_image(part):
    """PIL image from an ``image_url`` part, its ``{"url": ...}`` or a bare URL."""
    url = part.get("image_url", part) if isinstance(part, dict) else part
    url = url.get("url") if isinstance(url, dict) else url
    # Only inline images: fetching remote URLs from the pod would be an SSRF hole.
    if not isinstance(url, str) or not url.startswith("data:") or "," not in url:
        raise BadRequestError("images must be base64 data: URLs")
    try:
        return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert(
            "RGB"
        )
    except Exception as e:
        raise BadRequestError(f"cannot decode image: {e}") from e


def stream_generate(tokenizer, model, batch, gen, on_text):
    """``model.generate(**batch)`` on a worker thread, calling ``on_text`` per decoded piece.

    Returns ``(prompt_tokens, completion_tokens)``.
    """
    from transformers import TextIteratorStreamer

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    result = {}

    def run():
        try:
            with torch.no_grad():
                result["sequences"] = model.generate(**batch, streamer=streamer, **gen)
        except BaseException as e:
            result["error"] = e
            streamer.end()

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    for text in streamer:
        if text:
            on_text(text)
    worker.join()
    if "error" in result:
        raise result["error"]
    prompt_len = batch["input_ids"].shape[1]
    return prompt_len, result["sequences"].shape[1] - prompt_len


class Chat:
    """``/v1/chat/completions`` over ``hf_messages`` and ``complete`` of the subclass."""

    path = "/v1/chat/completions"

    def handle(self, h, body):
        conv = self.hf_messages(body.get("messages"))
        max_new = int(
            body.get("max_completion_tokens")
            or body.get("max_tokens")
            or DEFAULT_MAX_TOKENS
        )
        temperature = float(body.get("temperature") or 0.0)
        gen = {"max_new_tokens": max_new, "do_sample": temperature > 0}
        if temperature > 0:
            gen.update(temperature=temperature, top_p=float(body.get("top_p", 1.0)))

        rid, created = f"chatcmpl-{uuid.uuid4().hex}", int(time.time())

        def chunk(choices, **extra):
            return {
                "id": rid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": h.server.model_name,
                "choices": choices,
                **extra,
            }

        stream = bool(body.get("stream"))
        pieces = []
        if stream:
            h.start_sse()
            h.sse(chunk([{"index": 0, "delta": {"role": "assistant", "content": ""}}]))

        def on_text(text):
            pieces.append(text)
            if stream:
                h.sse(chunk([{"index": 0, "delta": {"content": text}}]))

        prompt_tokens, completion_tokens = self.complete(conv, gen, on_text)
        finish = "length" if completion_tokens >= max_new else "stop"
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        if not stream:
            h.send_json(
                {
                    "id": rid,
                    "object": "chat.completion",
                    "created": created,
                    "model": h.server.model_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "".join(pieces),
                            },
                            "finish_reason": finish,
                        }
                    ],
                    "usage": usage,
                }
            )
            return
        h.sse(chunk([{"index": 0, "delta": {}, "finish_reason": finish}]))
        if (body.get("stream_options") or {}).get("include_usage"):
            h.sse(chunk([], usage=usage))
        h.sse("[DONE]")


class VLM(Chat):

    def __init__(self, model_path, trust_remote_code):
        from transformers import AutoProcessor

        from hf_adapters import AutoSpyreModelForImageTextToText
        from hf_adapters.auto_spyre_model import dtype_for_model_path

        self.dtype = dtype_for_model_path(
            model_path, target_device="spyre", trust_remote_code=trust_remote_code
        )
        self.model = AutoSpyreModelForImageTextToText.from_pretrained(
            model_path, dtype=self.dtype, trust_remote_code=trust_remote_code
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        self.processor.tokenizer.padding_side = "left"

    def warmup(self):
        self.complete(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": Image.new("RGB", (448, 448))},
                        {"type": "text", "text": "Describe this image."},
                    ],
                }
            ],
            {"max_new_tokens": 2, "do_sample": False},
            lambda _: None,
        )

    @staticmethod
    def hf_messages(messages):
        if not isinstance(messages, list) or not messages:
            raise BadRequestError("messages is required")
        out, images = [], 0
        for m in messages:
            content = m.get("content")
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            if not isinstance(content, list):
                raise BadRequestError(
                    "message content must be a string or a list of parts"
                )
            parts = []
            for p in content:
                kind = p.get("type")
                if kind == "text":
                    parts.append({"type": "text", "text": p.get("text", "")})
                elif kind == "image_url":
                    parts.append({"type": "image", "image": decode_image(p)})
                    images += 1
                else:
                    raise BadRequestError(f"unsupported content part type {kind!r}")
            out.append({"role": m.get("role", "user"), "content": parts})
        # The Spyre VLM prefill always consumes image features.
        if not images:
            raise BadRequestError("this model needs at least one image_url part")
        return out

    def complete(self, conv, gen, on_text):
        """Generate for ``conv``, calling ``on_text`` per decoded piece.

        Returns ``(prompt_tokens, completion_tokens)``.
        """
        batch = self.processor.apply_chat_template(
            conv,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        if "pixel_values" in batch:
            batch["pixel_values"] = batch["pixel_values"].to(self.dtype)
        return stream_generate(
            self.processor.tokenizer, self.model, batch, gen, on_text
        )


class CausalLM(Chat):
    """Text generation. A checkpoint without a chat template gets the conversation as a plain prompt."""

    def __init__(self, model_path, trust_remote_code):
        from transformers import AutoTokenizer

        from hf_adapters import AutoSpyreModelForCausalLM
        from hf_adapters.auto_spyre_model import dtype_for_model_path

        self.model = AutoSpyreModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype_for_model_path(
                model_path, target_device="spyre", trust_remote_code=trust_remote_code
            ),
            trust_remote_code=trust_remote_code,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )

    def warmup(self):
        self.complete(
            [{"role": "user", "content": "Hello"}],
            {"max_new_tokens": 2, "do_sample": False},
            lambda _: None,
        )

    @staticmethod
    def hf_messages(messages):
        if not isinstance(messages, list) or not messages:
            raise BadRequestError("messages is required")
        out = []
        for m in messages:
            content = m.get("content")
            if isinstance(content, list):
                if any(p.get("type") != "text" for p in content):
                    raise BadRequestError("this model takes text content only")
                content = "".join(p.get("text", "") for p in content)
            if not isinstance(content, str):
                raise BadRequestError(
                    "message content must be a string or a list of text parts"
                )
            out.append({"role": m.get("role", "user"), "content": content})
        return out

    def encode(self, conv):
        tok = self.tokenizer
        if tok.chat_template:
            return tok.apply_chat_template(
                conv,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        return tok("\n".join(m["content"] for m in conv), return_tensors="pt")

    def complete(self, conv, gen, on_text):
        batch = {
            k: v
            for k, v in self.encode(conv).items()
            if k in ("input_ids", "attention_mask")
        }
        return stream_generate(self.tokenizer, self.model, batch, gen, on_text)


class Rerank:
    path = "/v1/rerank"
    max_length = 512

    def __init__(self, model_path, trust_remote_code):
        from transformers import AutoTokenizer

        from hf_adapters import AutoSpyreModelForSequenceClassification
        from hf_adapters.auto_spyre_model import dtype_for_model_path

        self.model = AutoSpyreModelForSequenceClassification.from_pretrained(
            model_path,
            dtype=dtype_for_model_path(
                model_path, target_device="spyre", trust_remote_code=trust_remote_code
            ),
            trust_remote_code=trust_remote_code,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )

    def warmup(self):
        self.score("warmup", ["warmup"])

    def score(self, query, documents):
        """``(relevance scores, token count)``; a single-logit head goes through a sigmoid, as in vLLM."""
        encoded = self.tokenizer(
            [query] * len(documents),
            text_pair=documents,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            padding_side="right",
        )
        with torch.no_grad():
            logits = self.model(**encoded, return_dict=True).logits.float()
        scores = (
            logits[:, 0].sigmoid()
            if logits.shape[1] == 1
            else logits.softmax(-1)[:, -1]
        )
        return scores.tolist(), int(encoded["attention_mask"].sum())

    def handle(self, h, body):
        query, docs = body.get("query"), body.get("documents")
        if not isinstance(query, str) or not query.strip():
            raise BadRequestError("query is required")
        if (
            not isinstance(docs, list)
            or not docs
            or not all(isinstance(d, str) and d.strip() for d in docs)
        ):
            raise BadRequestError("documents must be a non-empty list of strings")
        scores, tokens = self.score(query, docs)
        ranked = sorted(range(len(docs)), key=lambda i: -scores[i])
        top = int(body.get("top_n") or len(docs))
        h.send_json(
            {
                "id": f"rerank-{uuid.uuid4().hex}",
                "model": h.server.model_name,
                "usage": {"total_tokens": tokens},
                "results": [
                    {
                        "index": i,
                        "document": {"text": docs[i]},
                        "relevance_score": scores[i],
                    }
                    for i in ranked[:top]
                ],
            }
        )


class CLIP:
    path = "/v1/embeddings"

    def __init__(self, model_path, trust_remote_code):
        from sentence_transformers import SentenceTransformer

        import hf_adapters.st_backend  # noqa: F401  (registers backend="spyre")

        self.st = SentenceTransformer(
            model_path, backend="spyre", trust_remote_code=trust_remote_code
        )

    def warmup(self):
        # A text-only model (any plain SentenceTransformer) raises on an image input.
        self.images = "image" in getattr(self.st, "modalities", ["text", "image"])
        self.embed(["warmup"] + ([Image.new("RGB", (224, 224))] if self.images else []))

    def embed(self, items):
        """Embeddings in input order; texts and images go in separate batches."""
        vectors: list = [None] * len(items)
        for kind in (str, Image.Image):
            idx = [i for i, x in enumerate(items) if isinstance(x, kind)]
            if idx:
                encoded = self.st.encode([items[i] for i in idx], convert_to_numpy=True)
                for i, v in zip(idx, encoded):
                    vectors[i] = v.tolist()
        return vectors

    def handle(self, h, body):
        raw = body.get("input")
        raw = raw if isinstance(raw, list) else [raw]
        items = []
        for x in raw:
            if isinstance(x, dict) and x.get("type") == "text":
                x = x.get("text")
            if isinstance(x, str):
                if not x:
                    raise BadRequestError("input strings must be non-empty")
                items.append(x)
            elif isinstance(x, dict) and x.get("type") == "image_url":
                if not self.images:
                    raise BadRequestError(f"{h.server.model_name} embeds text only")
                items.append(decode_image(x))
            else:
                raise BadRequestError(
                    "input items must be strings, text or image_url parts"
                )
        texts = [x for x in items if isinstance(x, str)]
        tokens = int(self.st.tokenize(texts)["attention_mask"].sum()) if texts else 0
        h.send_json(
            {
                "object": "list",
                "model": h.server.model_name,
                "data": [
                    {"object": "embedding", "index": i, "embedding": v}
                    for i, v in enumerate(self.embed(items))
                ],
                "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
            }
        )


class YOLO:
    path = "/v1/segment"
    size = 640

    def __init__(self, model_path, trust_remote_code):
        try:
            from hf_adapters.ultralytics_yolo import (  # type: ignore[import-untyped]
                load_model,
                prepare_input,
            )
        except ImportError as e:
            raise SystemExit(
                f"task 'yolo' is unsupported in this build "
                f"(hf_adapters.ultralytics_yolo unavailable: {e})"
            ) from e
        self.prepare_input = prepare_input
        model = load_model(model_path, device="spyre")
        self.names = model.names
        self.model = torch.compile(model, dynamic=False)

    def warmup(self):
        self.detect(Image.new("RGB", (self.size, self.size)), 0.25, 0.45)

    def detect(self, image, conf, iou):
        import numpy as np
        from ultralytics.data.augment import LetterBox  # type: ignore[import-not-found]
        from ultralytics.utils import ops  # type: ignore[import-not-found]
        from ultralytics.utils.nms import (  # type: ignore[import-not-found]
            non_max_suppression,
        )

        # Gray padding is channel-symmetric, so letterboxing RGB directly is fine.
        boxed = LetterBox((self.size, self.size), auto=False)(image=np.asarray(image))
        x = torch.from_numpy(np.ascontiguousarray(boxed.transpose(2, 0, 1)) / 255.0)
        with torch.no_grad():
            out = self.model(self.prepare_input(x.float().unsqueeze(0), "spyre"))
        first = out[0]
        pred, proto = (
            (first[0], first[1]) if isinstance(first, (tuple, list)) else (first, None)
        )
        pred = pred.cpu().float()
        # nc=0 lets NMS infer the class count for detect heads.
        det = non_max_suppression(
            pred, conf, iou, nc=len(self.names) if proto is not None else 0
        )[0]
        shape = (image.height, image.width)
        segments = []
        if proto is not None and len(det):
            masks = ops.process_mask(
                proto.cpu().float()[0],
                det[:, 6:],
                det[:, :4],
                (self.size, self.size),
                upsample=True,
            )
            segments = [
                ops.scale_coords((self.size, self.size), s, shape, normalize=False)
                for s in ops.masks2segments(masks)
            ]
        boxes = ops.scale_boxes((self.size, self.size), det[:, :4].clone(), shape)
        return [
            {
                "box": [round(float(v), 1) for v in boxes[i]],
                "class_id": int(det[i, 5]),
                "class": self.names[int(det[i, 5])],
                "score": round(float(det[i, 4]), 4),
                **(
                    {
                        "polygon": [
                            [round(float(a), 1), round(float(b), 1)]
                            for a, b in segments[i]
                        ]
                    }
                    if segments
                    else {}
                ),
            }
            for i in range(len(det))
        ]

    def handle(self, h, body):
        image = decode_image(body.get("image"))
        detections = self.detect(
            image, float(body.get("conf", 0.25)), float(body.get("iou", 0.45))
        )
        h.send_json(
            {
                "model": h.server.model_name,
                "width": image.width,
                "height": image.height,
                "detections": detections,
            }
        )


TASKS = {"vlm": VLM, "chat": CausalLM, "rerank": Rerank, "clip": CLIP, "yolo": YOLO}


class Handler(BaseHTTPRequestHandler):
    server: "Server"

    def send_json(self, obj, status=HTTPStatus.OK):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def error(self, status, message):
        self.send_json(
            {"error": {"message": message, "type": status.phrase, "code": status}},
            status,
        )

    def start_sse(self):
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.streaming, self.disconnected = True, False

    def sse(self, obj):
        # A client that went away must not abort a generation still holding the card.
        if self.disconnected:
            return
        try:
            payload = obj if isinstance(obj, str) else json.dumps(obj)
            self.wfile.write(f"data: {payload}\n\n".encode())
            self.wfile.flush()
        except OSError:
            self.disconnected = True

    def do_GET(self):
        srv = self.server
        if self.path == "/health":
            if srv.ready.is_set():
                self.send_json({"status": "ok"})
            else:
                self.error(HTTPStatus.SERVICE_UNAVAILABLE, "model is loading")
        elif self.path == "/v1/models":
            self.send_json(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": srv.model_name,
                            "object": "model",
                            "created": srv.created,
                            "owned_by": "hf-adapters",
                            "root": srv.model_id,
                            "task": srv.task,
                        }
                    ],
                }
            )
        else:
            self.error(HTTPStatus.NOT_FOUND, f"no route {self.path}")

    def do_POST(self):
        srv = self.server
        if self.path != TASKS[srv.task].path:
            return self.error(HTTPStatus.NOT_FOUND, f"no route {self.path}")
        if not srv.ready.is_set():
            return self.error(HTTPStatus.SERVICE_UNAVAILABLE, "model is loading")
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY_BYTES:
            return self.error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "body too large")
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body, dict):
                raise BadRequestError("body must be a JSON object")
        except ValueError as e:
            return self.error(HTTPStatus.BAD_REQUEST, f"invalid JSON: {e}")
        self.streaming = False
        try:
            with srv.lock:
                srv.backend.handle(self, body)
        except Exception as e:
            status = (
                HTTPStatus.BAD_REQUEST
                if isinstance(e, BadRequestError)
                else HTTPStatus.INTERNAL_SERVER_ERROR
            )
            if status == HTTPStatus.INTERNAL_SERVER_ERROR:
                self.log_error("request failed: %r", e)
            if self.streaming:
                self.sse({"error": {"message": str(e), "code": status}})
            else:
                self.error(status, str(e))


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, task, model_id, model_name):
        super().__init__(address, Handler)
        self.task, self.model_id, self.model_name = task, model_id, model_name
        self.created = int(time.time())
        self.backend = None
        self.lock = threading.Lock()
        self.ready = threading.Event()

    def load(self, model_path, trust_remote_code):
        try:
            t0 = time.time()
            backend = TASKS[self.task](model_path, trust_remote_code)
            t1 = time.time()
            backend.warmup()
            print(
                f"loaded {self.model_id} in {t1 - t0:.1f}s, "
                f"warmup {time.time() - t1:.1f}s",
                flush=True,
            )
            self.backend = backend
            self.ready.set()
        except BaseException as e:
            print(f"failed to load {self.model_id}: {e!r}", file=sys.stderr, flush=True)
            # Exit the process so the pod restarts visibly instead of never turning ready.
            os._exit(1)


def main(argv=None):
    env = os.environ.get
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--task", choices=sorted(TASKS), default=env("HF_SERVE_TASK"))
    p.add_argument("--model", default=env("HF_SERVE_MODEL"), help="Hub id or path")
    p.add_argument("--revision", default=env("HF_SERVE_REVISION"))
    p.add_argument("--served-model-name", default=env("HF_SERVE_MODEL_NAME"))
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--host", default=env("HF_SERVE_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(env("HF_SERVE_PORT", "8000")))
    args = p.parse_args(argv)
    if not args.task or not args.model:
        p.error("--task and --model are required")

    model_path = args.model
    # YOLO weights are a file path, not a Hub repo.
    if args.task != "yolo" and not os.path.isdir(model_path):
        from huggingface_hub import snapshot_download

        model_path = snapshot_download(model_path, revision=args.revision)

    server = Server(
        (args.host, args.port),
        args.task,
        args.model,
        args.served_model_name or args.model,
    )
    threading.Thread(
        target=server.load, args=(model_path, args.trust_remote_code), daemon=True
    ).start()
    print(f"serving {args.task} {args.model} on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
