# `generate()` gaps vs. stock HF `model.generate()`

Our [`generate()`](../hf_adapters/hf_common.py) in `hf_common.py` covers greedy +
temperature/top-k/top-p sampling with HF-matching parameter precedence and EOS
stopping, but diverges from stock HF in several ways worth documenting.

## Partially stock-shaped signature

- **Inputs are pre-tokenized.** Callers pass `input_ids` and an optional
  `attention_mask`, as with stock HF. Ordinary contiguous left or right caller
  padding is removed on CPU, the logical prompts are compacted, and each row is
  right-aligned in the internal 64-token block layout before prefill. Sparse
  masks with holes are rejected rather than silently reinterpreted.
- **A keyword-only `tokenizer` is still required for output decoding.** The
  implementation temporarily returns generated strings, so final
  `tokenizer.decode(..., skip_special_tokens=True)` remains inside `generate()`.
  Moving decoding outside and returning stock token tensors is a follow-up.
- **`max_new_tokens` is required**, not optional. HF resolves a default length
  via `max_length` (prompt + new); our block-decode loop doesn't implement
  `max_length`, so callers must always state a new-token budget.

## Internal block alignment

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
- **Logits processors / warpers** beyond top-k/top-p: no `repetition_penalty`,
  `no_repeat_ngram_size`, `min_new_tokens`, `bad_words_ids`, `min_p`,
  `typical_p`, etc.
- **Custom `StoppingCriteria` / `stopping_criteria`** — only EOS-token stopping
  is implemented (matching `EosTokenCriteria`); no stop-strings, no `max_time`.
- **`LogitsProcessorList` / `logits_processor` injection**, `streamer`,
  `prefix_allowed_tokens_fn`, `forced_bos/eos_token_id`, etc.

## Other behavioral notes

- Returns a `list[str]` only — no `GenerateOutput`, no `output_scores` /
  `output_hidden_states` / `return_dict_in_generate`.
- Sampling/EOS precedence (explicit kwarg > `generation_config` > HF default)
  *does* match stock HF via `_prepare_generation_config`, so that part is
  faithful.
