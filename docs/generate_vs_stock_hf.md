# `generate()` gaps vs. stock HF `model.generate()`

Our [`generate()`](../hf_adapters/hf_common.py#L794) in `hf_common.py` covers greedy +
temperature/top-k/top-p sampling with HF-matching parameter precedence and EOS
stopping, but diverges from stock HF in several ways worth documenting.

## Different signature

- **We take `(tokenizer, prompts)`, not `input_ids`.** Stock HF takes
  pre-tokenized `input_ids`/`inputs_embeds`; tokenization and decoding happen
  *outside* `generate()`. Here tokenization and final
  `tokenizer.decode(..., skip_special_tokens=True)` are baked in, so the caller
  passes raw strings and gets back strings
  ([hf_common.py:861](../hf_adapters/hf_common.py#L861),
  [hf_common.py:1091](../hf_adapters/hf_common.py#L1091)).
- **`max_new_tokens` is required**, not optional. HF resolves a default length
  via `max_length` (prompt + new); our block-decode loop doesn't implement
  `max_length`, so callers must always state a new-token budget
  ([hf_common.py:818](../hf_adapters/hf_common.py#L818)).

## Forced left-padding + block alignment

- Inputs are **always left-padded** (`padding_side="left"`), then further padded
  up to a `BLOCK_SIZE` multiple
  ([hf_common.py:858-882](../hf_adapters/hf_common.py#L858-L882)). Required so
  `logits[:, -1, :]` predicts from a real position and all sequences end at the
  same index. Stock HF supports either padding side and doesn't impose block
  alignment. Position IDs and attention masks are constructed to compensate.

## Unsupported decoding modes

Only **greedy** and **top-k / top-p / temperature sampling** are implemented
([hf_common.py:1024-1043](../hf_adapters/hf_common.py#L1024-L1043)). Not
supported:

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
  *does* match stock HF via `_prepare_generation_config`
  ([hf_common.py:763](../hf_adapters/hf_common.py#L763)), so that part is
  faithful.
