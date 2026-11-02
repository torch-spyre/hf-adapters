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

"""Greedy tree speculative decoding for Gemma 4 26B-A4B with the XPress drafter.

Batch 1, greedy. Each step:

1. the XPress drafter proposes a tree of ``width`` nodes rooted at the last
   emitted token (``_xpress_tree``), reading the target hidden states of the
   rows committed by the previous step;
2. the target scores the whole tree in one forward (``_gemma4_tree``);
3. the longest root-to-node path matching the target's own greedy argmax is
   accepted, plus the target's next token after it (always emitted);
4. the accepted rows are committed to the target's KV caches, and their tapped
   hidden states become the drafter's next context rows.

Every emitted token is the target's argmax, so the output is the target's
greedy continuation up to the floating-point differences of a tree forward.
Speed comes from emitting several tokens per target forward.

Options that change the target's numerics are separate and off by default:
``SPYRE_MOE_UNION_CAP`` (expert cap, ``hf_common``) and ``SPYRE_GEMMA4_FP8``
(W8A8 linears, ``_gemma4_fp8``).

Usage::

    from hf_adapters import AutoSpyreModelForCausalLM, hf_xpress_qwen3
    from hf_adapters.gemma4_spec_decode import TreeSpeculativeDecoder

    target = AutoSpyreModelForCausalLM.from_pretrained(
        "google/gemma-4-26B-A4B-it", dtype=torch.float16)
    drafter = hf_xpress_qwen3.load_model(
        "UIUC-SSAIL/gemma4-26b-a4b-XPress-b16",
        target_path="google/gemma-4-26B-A4B-it", dtype=torch.float32)
    decoder = TreeSpeculativeDecoder(target, drafter, width=16)
    out = decoder.generate(input_ids, max_new_tokens=256)
"""

import time
from dataclasses import dataclass, field

import torch

from hf_adapters._gemma4_tree import TreeVerifier, accept_greedy
from hf_adapters._xpress_tree import XPressTreeDrafter
from hf_adapters.hf_common import text_config


@dataclass
class SpecDecodeStats:
    """Per-request timings (ms) and acceptance; ``steps`` excludes the prefill."""

    prefill_ms: float = 0.0
    draft_ms: list = field(default_factory=list)
    verify_ms: list = field(default_factory=list)
    commit_ms: list = field(default_factory=list)
    accepted: list = field(default_factory=list)  # tokens emitted per step

    @property
    def steps(self):
        return len(self.accepted)

    @property
    def mean_accepted(self):
        return sum(self.accepted) / max(1, len(self.accepted))

    def decode_tps(self):
        """Emitted tokens per second of decode wall time (prefill excluded)."""
        step_ms = [
            d + v + c for d, v, c in zip(self.draft_ms, self.verify_ms, self.commit_ms)
        ]
        tokens = sum(self.accepted)
        return 1000.0 * tokens / max(sum(step_ms), 1e-9)


def _sync():
    sync = getattr(getattr(torch, "spyre", None), "synchronize", None)
    if sync is not None:
        sync()


class TreeSpeculativeDecoder:
    """Drafter + verifier pair for one prepared target and one XPress drafter."""

    def __init__(
        self,
        target,
        drafter,
        *,
        width=16,
        draft_vocab=None,
        frontier=4,
        branching=4,
        depth=7,
        drafter_kwargs=None,
    ):
        self.target = target
        self.width = int(width)
        taps = tuple(drafter.target_layer_ids)
        self.verifier = TreeVerifier(target, self.width, tap_layers=taps)
        self.drafter = XPressTreeDrafter(
            drafter,
            draft_vocab=draft_vocab,
            frontier=frontier,
            branching=branching,
            depth=depth,
            **(drafter_kwargs or {}),
        )
        self.taps = taps
        cfg = text_config(target.config)
        self.vocab_size = int(cfg.vocab_size)
        self.stats = None

    @staticmethod
    def _host(t):
        # Copy in the device dtype, then cast on the host: a materialized
        # fp16 -> fp32 cast on Spyre permutes lanes (see hf_xpress_qwen3).
        return t.to("cpu").float()

    def _features(self, taps):
        """Concatenate tapped layer outputs ``{layer: [1, n, H]}`` in tap order."""
        return torch.cat([self._host(taps[i]) for i in self.taps], dim=-1)[0]

    def _pad_tree(self, tokens, parents):
        """Fill a short tree to ``width`` nodes with extra root children.

        The verify graph has a fixed width. Filler children carry tokens unused
        among the root's children, so ``accept_greedy`` stays unique; their
        predictions are ignored unless the target picks them.
        """
        tokens, parents = list(tokens), list(parents)
        used = {t for t, p in zip(tokens, parents) if p == 0}
        filler = 0
        while len(tokens) < self.width:
            while filler in used or filler == tokens[0]:
                filler += 1
            tokens.append(filler)
            parents.append(0)
            used.add(filler)
        return tokens, parents

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=256, eos_token_id=None):
        """Greedy continuation ``[1, n]`` of a ``[1, P]`` prompt (batch 1)."""
        if input_ids.shape[0] != 1:
            raise ValueError("tree speculative decoding is batch-1 only")
        eos = (
            set()
            if eos_token_id is None
            else set([eos_token_id] if isinstance(eos_token_id, int) else eos_token_id)
        )
        stats = self.stats = SpecDecodeStats()
        v, d = self.verifier, self.drafter
        t0 = time.perf_counter()
        logits = v.prefill(input_ids, max_new_tokens)
        first = int(self._host(logits[0, -1, : self.vocab_size]).argmax())
        prompt_rows = self._features(v.state.taps)
        d.reset(prompt_rows.shape[0] + max_new_tokens + self.width)
        d.append(prompt_rows)
        _sync()
        stats.prefill_ms = 1000 * (time.perf_counter() - t0)
        out = [first]
        prompt_last = int(input_ids[0, -1])
        new_rows = None
        while len(out) < max_new_tokens and out[-1] not in eos:
            t0 = time.perf_counter()
            # The refiner's slot-0 input: the token before the anchor.
            predecessor = out[-2] if len(out) > 1 else prompt_last
            tokens, parents = d.propose(out[-1], predecessor, self.width, new_rows)
            tokens, parents = self._pad_tree(tokens, parents)
            t1 = time.perf_counter()
            logits = v.verify(tokens, parents)
            predictions = (
                self._host(logits[0, :, : self.vocab_size]).argmax(-1).tolist()
            )
            t2 = time.perf_counter()
            path, bonus = accept_greedy(tokens, parents, predictions)
            taps = v.tap_buffers()
            new_rows = self._features(taps)[path]
            v.commit(path)
            _sync()
            t3 = time.perf_counter()
            emitted = [tokens[p] for p in path[1:]] + [bonus]
            for token in emitted:
                if len(out) >= max_new_tokens or out[-1] in eos:
                    break
                out.append(token)
            stats.draft_ms.append(1000 * (t1 - t0))
            stats.verify_ms.append(1000 * (t2 - t1))
            stats.commit_ms.append(1000 * (t3 - t2))
            stats.accepted.append(len(emitted))
        return torch.tensor([out[:max_new_tokens]])
