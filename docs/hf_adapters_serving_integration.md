# HF-adapters serving integration

In supported adapters, the same model preparation and backbone code used for standalone HF adapters
execution can also be used for serving via Spyre implementations of vLLM and SGLang.

An engine-independent API lets each serving integration
supply attention and request handling. Supported tasks are text generation and
dense text embeddings, subject to adapter and engine capabilities.

## Ownership and execution

| Component | Responsibility |
|---|---|
| HF-adapters | Weight transformations, embeddings, Q/K/V projections, positional encoding, normalization, residuals, MLPs, backbone driver and block compilation hooks |
| Spyre-inference (vLLM) or SGLang-Spyre | Model loading integration, attention binding, tensor-layout conversion, device placement and serving lifecycle |
| Serving engine and its Spyre backend | Request preprocessing, scheduling, attention masks, cache allocation, shape buckets, warmup, output head, logits processing, sampling or pooling |

In supported adapters, the attention logic - between Q/K/V preparation and the output projection - is replaceable.
Standalone decoders use `DenseAttention` with adapter-owned dense KV caches.
Serving uses `ExternalAttention` with a bound kernel and no adapter caches.
The
serving wrapper calls the adapter backbone directly, bypassing stock HF forward
and its attention-dispatch hooks.

HF-adapters imports neither serving engine nor
its integration package.

## Shared API

Prepare the backbone, bind attention, place the model and compile:

```python
from hf_adapters.execution import prepare_adapter_execution

prepared = prepare_adapter_execution(
    loaded_cpu_backbone.eval(), max_model_len=context_length, task="generate"
)
# Use task="embed" for embeddings; "generate" is the default.

# make_engine_attention() and place_prepared_body() are integration helpers.
for executor in prepared.runtime.attentions:
    executor.bind(make_engine_attention(executor.spec))
place_prepared_body(prepared.model)

prepared.runtime.compile_body(prepared.model)
hidden_states = prepared.forward(input_ids, position_ids, token_type_ids=None)
```

- `PreparedExecution` holds `model`, `runtime`, `driver` and `encoder_only`.
  `forward()` returns final hidden states, before the LM head or pooling. Decoder
  drivers receive `None` for the mask, cache index and each layer's KV caches;
  encoder drivers receive no adapter mask. Token-type IDs require encoder support.
- `AttentionSpec` describes each layer: `layer_idx`, `num_heads`,
  `num_kv_heads`, final `head_size`, original softmax `scale`, `is_causal`,
  `uses_cache` and optional `position_bias`. Descriptors follow layer order; for
  MHA, KV-head count defaults to query-head count.
- `ExternalRuntime.attention(attn, *, head_size=None, scale=None)` derives a
  descriptor from an attention module. `register_attention(spec)` accepts an
  explicit descriptor. Both return an `ExternalAttention` executor.
- `executor.bind(kernel)` binds once. The kernel takes `(q, k, v)` and returns
  attention output, all in adapter layout `[batch, heads, tokens, head_size]`.
  The adapter calls the executor with
  `(q, k, v, mask, key_cache, value_cache, cache_index)`. It returns
  `(output, None, None)`; all mask/cache arguments must be `None`.
- `runtime.wrap(function)` records a callable for deferred compilation.
  `compile_body(model)` installs `torch.compile(dynamic=False, fullgraph=True)`
  wrappers once, after all executors are bound. The runtime also carries
  `max_model_len`, `task`, `is_causal` and optional `logits_scale`.

## Preparation and serving lifecycle

1. **Load.** Validate serving options and load weights into an unprepared,
   eval-mode CPU backbone. Preserve projection biases and tied embeddings; bypass
   native transformations that duplicate adapter preparation.
2. **Prepare once.** The shared API splits projections, pads heads and applies
   model adaptations, skipping standalone head/cache setup. Freeze rotary tables
   for the configured context and register prepared buffers for device placement.
3. **Publish geometry.** Validate descriptors before creating attention modules
   or sizing caches. Preserve the HF configuration object's identity, layer/head
   counts, hidden size and vocabulary size. Head padding must preserve the
   original attention scale, including when the model already scales Q.
4. **Bind, place and compile.** Register and bind engine attention modules, then
   place the prepared body with adapter-aware device/layout handling. Compile the
   deferred callables, bypassing native body compilation. Keep adapter-specific
   eager final norms.
5. **Warm up.** Warm declared body, attention and output shapes before serving.
   Bucket and pad inputs outside the backbone; discard padded
   output rows. Use legal padding IDs and exclude padding from request lengths
   and real KV slots. Pooler warmup must match runtime device layouts and gathers.

Generation uses the engine's LM head and logits processor. Resolve tied
embeddings before head packing and apply `runtime.logits_scale` exactly once;
Granite supplies `1 / config.logits_scaling`. Embeddings use engine pooling and
load no LM head.

## Generation and embedding contracts

| Task | Causality | Persistent KV cache |
|---|---|---|
| Generation | Causal | Engine-owned paged cache |
| Decoder embeddings | Causal or bidirectional, per model configuration | None |
| Encoder embeddings | Bidirectional; local/relative bias may vary by layer where supported | None |

### Generation

Pack IDs and positions as `[1, total_tokens]`. Attention metadata separates
requests; the rest of the supported backbone computation is tokenwise.
Use engine-provided positions, including cached-prefix and chunk-continuation
offsets. Packed row numbers are not token positions. Layout bridges convert Q/K/V
to the engine layout and restore the adapter output layout. Mutable request
metadata stays outside compiled block graphs.

### Embeddings

Run each request in full with cache-free attention. Packed or rectangular
batches must preserve request boundaries and exclude padding from attention and
pooling. Honor request-relative positions, position-table offsets and supported
token-type IDs. Decoder embeddings honor `is_causal=False` and
`use_bidirectional_attention=True`.

Relative or local attention uses `AttentionSpec.position_bias`, a registered
model-owned module returning `[1, heads_or_1, extent, extent]`. The extent must
cover padded request lengths. The serving kernel slices and adds the bias per
request; gathering or scattering padding rows must preserve real outputs. Local
bias restricts attention to the window, but computation remains dense.

Pooling returns one vector per request using CLS, MEAN or LAST, with optional
normalization. Supported SentenceTransformers pipelines contain `Transformer`, one
`Pooling` and optionally `Normalize`. Learned Dense projections, custom modules,
combined, weighted, sqrt-length or prompt-excluding pooling require additional
support. Follow the checkpoint's instructions, special tokens and truncation rules.

## Engine integration and configuration

Use a version of Spyre-inference or SGLang-Spyre that includes the HF-adapters integration, and ensure each worker can import `hf_adapters`.
Select adapter mode explicitly; raise errors for unsupported requests.
Preserve unrelated caller configuration overrides.

### vLLM / Spyre-inference

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen3-0.6B",
    model_impl="transformers",
    max_model_len=384,
    additional_config={"spyre_model_preparation": "hf_adapters"},
)
```

Adapter mode selects `TransformersForCausalLM`. For embeddings, use an embedding
checkpoint with `runner="pooling"` to select `TransformersEmbeddingModel`, then call
`llm.embed(texts)`. Explicit `pooler_config` overrides SentenceTransformers
metadata; bare backbones require a pooling configuration. Embedding mode disables
chunked prefill and retains the pooling runner's prefix-cache policy.

The provider maps shared execution to `AttentionBinding` and `PreparedBody`.
Generation binds `PagedAdapterAttention`; embeddings bind
cache-free causal or encoder attention. The v1 runner retains its scheduling,
cache and warmup machinery. `enforce_eager=True` skips adapter body compilation.

### SGLang / SGLang-Spyre

```python
import json
import sglang as sgl

engine = sgl.Engine(
    model_path="Qwen/Qwen3-0.6B",
    model_impl="transformers",
    json_model_override_args=json.dumps(
        {"spyre_model_preparation": "hf_adapters"}
    ),
    device="cpu",
    attention_backend="spyre_paged",
    dtype="float16",
    page_size=128,
    context_length=384,
    chunked_prefill_size=256,
    max_running_requests=1,
    disable_cuda_graph=True,
)
```

`spyre_model_preparation` activates adapter placement, preserving HF architecture
and model type. Initialize Spyre before importing HF-adapters in the worker. A
Transformers wrapper extension handles generation; an embedding wrapper supplies
cache-free attention, rectangular packing and CLS/MEAN/LAST pooling. Embedding
mode disables chunked prefill and prefix caching.

Generation binds registered `RadixAttention` modules through Spyre's opaque
attention boundary, converting Q/K/V to `[T, heads * head_size]`. SGLang owns
request slots and RadixCache prefix lifetimes; its Spyre backend owns physical KV
pages. Keep `ForwardBatch` outside compiled graphs and clean up global attention
registration on shutdown. Output selection runs on CPU; the Spyre LM head
projects selected hidden states on-device and returns host logits.

## Adding an adapter

1. **Accept the runtime.** Add `prepare_for_spyre(model, *, runtime=None)` and pass
   `runtime` through preparation and block factories; `None` keeps standalone behavior.
   Reuse `prepare_standard_gqa()`, `make_decoder_block()` or `make_encoder_block()`
   where their model math applies.
2. **Expose the backbone.** `_run_backbone_forward()` returns final hidden states.
   Its decoder signature is `(model, input_ids, position_ids, attn_mask, key_caches,
   value_caches, cache_index)`; the encoder signature is
   `(model, input_ids, attn_mask, position_ids, token_type_ids)`. Exclude head and
   pooling work; accept absent external mask/cache state.
3. **Inject attention** between Q/K/V preparation and output projection. Register
   executors in layer order through `runtime.attention()` or `register_attention()`.
   Publish post-padding geometry and the original scale. Preserve GQA, Q/K
   normalization, partial RoPE and residual semantics.
4. **Defer compilation.** Use `compile_block(function, runtime)` from
   `hf_adapters.hf_common` or equivalent standalone/deferred wrapping. Register
   final normalization unless it must stay eager. External preparation must not
   trace blocks, allocate standalone caches or configure the LM head.
5. **Validate before mutation.** Check context and position-table bounds, head
   alignment, attention semantics and output width. Register all
   positional/bias buffers. Supply output scaling through `runtime.logits_scale`.
   Encoder adapters declare `_is_encoder_only = True`; active sliding attention
   requires `_supports_external_sliding_attention` and a compatible engine kernel.
6. **Register support.** Add the family/class mapping in
   `prepare_adapter_execution()` and a model registry entry. Put shared checks in
   the execution API, family checks in the adapter and serving-option checks in
   the provider. Extend providers only for new semantics or loading requirements.
7. **Verify.** Run stock HF before preparation, then compare standalone and
   external outputs against that reference. Cover loading/tying, padded-head
   scaling, positions, packed requests, cache continuation or pooling, repeated/reordered
   requests, deferred compilation and placement. Test each target engine on Spyre,
   including warmup and request/cache lifecycle. Check serving-stack import isolation.

## Scope boundaries

Serving supports a single device (`TP=PP=DP=1`), FP16, with standard unquantized
checkpoint loading. Generation requires uniform dense
attention geometry and full attention within the configured context; sliding
windows must cover that context. Local attention for embeddings requires explicit
bias support. Supported RoPE variants are default RoPE and LongRoPE within its
original short-context range. Other scaling schemes and attention softcapping
require further runtime support. Learned positions and head padding are
adapter-specific.

LoRA, speculative decoding, prepared-weight reload, MoE/hybrid state,
encoder-decoder models, multimodal models, reranking and sparse/late-interaction
embeddings are outside this scope. Engine-native or standalone support does not
establish adapter-mode support; each family must meet the shared and
engine-specific requirements above.
