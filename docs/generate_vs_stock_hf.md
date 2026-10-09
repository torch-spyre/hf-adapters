# `generate()` gaps vs. stock HF `model.generate()`

Our [`generate()`](../hf_adapters/hf_common.py) in `hf_common.py` covers greedy +
temperature/top-k/top-p sampling with HF-matching parameter precedence and EOS
stopping, but diverges from stock HF in several ways worth documenting.

## Stock-shaped inputs and basic output

- **Inputs are pre-tokenized.** Callers pass `input_ids` and an optional
  `attention_mask`, as with stock HF. `hf_adapters.encode_prompts()` is provided
  for canonical model-aware tokenization: it applies the checkpoint's chat
  template for instruct models and ordinary tokenizer post-processing for base
  models. Ordinary contiguous left or right caller padding is removed on CPU,
  the logical prompts are compacted, and each row is right-aligned in the
  internal 64-token block layout before prefill. Sparse masks with holes are
  rejected rather than silently reinterpreted.
- **The default return is a token tensor.** It contains the caller's exact
  `input_ids` prefix followed by generated tokens, including EOS. Decode the
  continuation outside `generate()` with the tokenizer, as with stock HF.
  Internal block padding is never returned; rows that finish early receive the
  configured pad token while the rest of the batch continues.
- **Basic rich decoder output is supported.** With
  `return_dict_in_generate=True`, generation returns stock HF's
  `GenerateDecoderOnlyOutput`. `output_scores=True` adds the processed scores
  used for token selection, and `output_logits=True` adds raw logits. Both are
  CPU tuples with one `[batch_size, vocab_size]` tensor per generated step;
  padded Spyre LM-head rows are cropped to the configured vocabulary. As in
  stock HF, output flags alone do not change the default tensor return.
- Length resolution follows stock HF conventions: `max_new_tokens` takes
  precedence over `max_length`, while `max_length` limits the total returned
  sequence width. If neither is configured, generation uses HF's model-agnostic
  default of 20 new tokens. `min_new_tokens` suppresses EOS until the minimum
  continuation length is reached.

## Profiling generation

[`scripts/profile_e2e_spyre.py`](../scripts/profile_e2e_spyre.py) uses
`encode_prompts()` and the shared checkpoint dtype policy. For example:

```bash
python scripts/profile_e2e_spyre.py --model gemma4_26b_a4b --max-new-tokens 10 --batch 1
```

Chat templates supply the instruct model's turn delimiters and generation
prefix. Gemma 4's direct tokenizer does not add BOS, so a raw prompt can produce
repetitive output even when stock CPU inference receives the same input.
Use `--raw-prompt` to explicitly request direct tokenization for experiments.
`--prompt-tokens` creates a synthetic fixed-length workload by repeating or
truncating tokenized input, including any chat framing; use the natural prompt
length when checking response correctness.

## Internal block alignment

- Autoregressive generation projects only the final prompt token through the
  vocabulary head. Chunked text prefill runs every chunk through the backbone
  to populate KV caches, then runs the head once. Multimodal prefill also keeps
  only the final row for the head. The prepared head still applies model-specific
  scaling, softcapping, and TP vocabulary gathering. Direct full-forward helpers
  retain all-token logits by default. Diffusion decoding uses its own loop and
  needs logits for every canvas position.
- Attention and KV shapes stay fixed across a prompt's prefill chunks to avoid
  compiling a different graph for every populated prefix length.

- External left and right padding are both accepted when each row has one
  contiguous valid span. The caller's padding width does not affect KV-cache
  capacity.
- Internally, prompts are still right-aligned and left-padded up to a
  `BLOCK_SIZE` multiple. This ensures `logits[:, -1, :]` predicts from a real
  position and lets the static Spyre decode scheduler share physical cache
  coordinates across a heterogeneous batch. This internal padding is not part
  of the logical prompt and is not decoded.
- Custom `position_ids`, arbitrary sparse/higher-rank masks, and `inputs_embeds`
  are not supported by this input path.

## Unsupported decoding modes

Only **greedy** and **top-k / top-p / temperature sampling** are implemented.
Not supported:

- **Beam search** (`num_beams > 1`), group/diverse beam search, contrastive
  search, assisted/speculative decoding.
- **`num_return_sequences > 1`.**
- **Logits processors / warpers** beyond top-k/top-p and EOS suppression for
  `min_new_tokens`: no `repetition_penalty`, `no_repeat_ngram_size`,
  `bad_words_ids`, `min_p`, `typical_p`, `epsilon_cutoff`, `eta_cutoff`,
  `exponential_decay_length_penalty`, etc. `forced_bos_token_id` is supported,
  including its stock interaction with `begin_suppress_tokens`.
- **Custom `StoppingCriteria` / `stopping_criteria`** — only EOS-token stopping
  is implemented (matching `EosTokenCriteria`); no stop-strings, no `max_time`.
- **`LogitsProcessorList` / `logits_processor` injection**, `streamer`,
  `prefix_allowed_tokens_fn`, `forced_eos_token_id`, etc.

## Other behavioral notes

- Rich output is deliberately minimal: attentions and hidden states remain
  unsupported, and `past_key_values` is `None` because the internal Spyre tensor
  caches are not a stock HF `Cache`. Text and multimodal generation share this
  output contract; VLM callers decode the returned continuation with the processor.
- Supported generation settings use stock precedence (explicit kwarg > caller
  `generation_config` > model config > HF default) through
  `_prepare_generation_config`. Active settings are checked against an explicit
  allowlist; unknown arguments and non-neutral unsupported generation-config
  options are rejected instead of being silently ignored.
