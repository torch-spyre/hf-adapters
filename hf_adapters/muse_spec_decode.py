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

"""Greedy tree speculative decoding for Muse Glimmer 30B with its DFlash assistant.

Batch 1, greedy. Each step:

1. the DFlash assistant drafts a block from the last emitted token and
   proposes a ``width``-node tree (``_muse_dflash``), reading the target
   hidden states of the rows committed by the previous step;
2. the target scores the whole tree in one forward (``_muse_tree``);
3. the longest root-to-node path matching the target's own greedy argmax is
   accepted, plus the target's next token after it (always emitted);
4. the accepted rows are committed, and their tapped hidden states become the
   assistant's next context rows.

Every emitted token is the target's argmax, so the output is the target's
greedy continuation up to the floating-point differences of a tree forward.

Usage::

    from transformers import MuseGlimmerAssistantModel
    from hf_adapters import AutoSpyreModelForCausalLM
    from hf_adapters.muse_spec_decode import TreeSpeculativeDecoder

    target = AutoSpyreModelForCausalLM.from_pretrained(
        "meta-models/Muse-Glimmer-30B", dtype=torch.float16)
    assistant = MuseGlimmerAssistantModel.from_pretrained(
        "meta-models/Muse-Glimmer-30B-assistant", dtype=torch.float32)
    decoder = TreeSpeculativeDecoder(target, assistant, width=16)
    out = decoder.generate(input_ids, max_new_tokens=256)
"""

import time
from dataclasses import dataclass, field

import torch

from hf_adapters._muse_dflash import MuseDFlashDrafter
from hf_adapters._muse_tree import MuseTreeVerifier
from hf_adapters._spec_tree import accept_greedy, pad_tree
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
        return 1000.0 * sum(self.accepted) / max(sum(step_ms), 1e-9)


def _sync():
    sync = getattr(getattr(torch, "spyre", None), "synchronize", None)
    if sync is not None:
        sync()


class TreeSpeculativeDecoder:
    """Assistant + verifier pair for one prepared Muse Glimmer target."""

    def __init__(
        self,
        target,
        assistant,
        *,
        width=16,
        top_k=4,
        depth=None,
        prefill_chunk=None,
        drafter_kwargs=None,
    ):
        self.width = int(width)
        taps = tuple(assistant.config.target_layer_ids)
        self.verifier = MuseTreeVerifier(
            target, self.width, tap_layers=taps, prefill_chunk=prefill_chunk
        )
        self.drafter = MuseDFlashDrafter(
            assistant, target, top_k=top_k, depth=depth, **(drafter_kwargs or {})
        )
        self.vocab_size = int(text_config(target.config).vocab_size)
        self.draft_capacity = None  # rows; set for one compile across prompts
        self.stats = None

    @staticmethod
    def _host(t):
        return t.to("cpu").float()

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens=256,
        eos_token_id=None,
    ):
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
        # Copy whole, crop on the host (the head's vocabulary is padded).
        first = int(self._host(logits)[0, -1, : self.vocab_size].argmax())
        prompt_rows = v.tap_features()
        need = prompt_rows.shape[0] + max_new_tokens + self.width
        # A fixed drafter capacity keeps its graph shapes across requests.
        d.reset(max(need, self.draft_capacity or 0))
        d.append(prompt_rows)
        _sync()
        stats.prefill_ms = 1000 * (time.perf_counter() - t0)
        out = [first]
        new_rows = None
        while len(out) < max_new_tokens and out[-1] not in eos:
            t0 = time.perf_counter()
            tokens, parents = d.propose(out[-1], self.width, new_rows)
            tokens, parents = pad_tree(tokens, parents, self.width)
            t1 = time.perf_counter()
            logits = v.verify(tokens, parents)
            predictions = (
                self._host(logits)[0, :, : self.vocab_size].argmax(-1).tolist()
            )
            t2 = time.perf_counter()
            path, bonus = accept_greedy(tokens, parents, predictions)
            new_rows = v.tap_features()[path]
            v.commit(path)
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
