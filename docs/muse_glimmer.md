# Muse Glimmer text generation

Load `meta-models/Muse-Glimmer-30B` with `AutoSpyreModelForCausalLM` and bf16.
The adapter uses the stock Hugging Face conditional-generation model and removes
its vision-only modules. Image and video tensors are not supported by this API.

```python
import torch
from transformers import AutoTokenizer

from hf_adapters import AutoSpyreModelForCausalLM

model = AutoSpyreModelForCausalLM.from_pretrained(
    "meta-models/Muse-Glimmer-30B", dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained("meta-models/Muse-Glimmer-30B")
inputs = tokenizer("The capital of France is", return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=32, do_sample=False)
```

Prompts longer than 512 real tokens automatically use prefill chunks of 512
tokens. This avoids the slow compilation observed with larger one-shot prompts.
Short prompts retain 64-token alignment. An `attention_mask` excludes caller
padding when choosing the default, including right-padded batches.

An explicit `prefill_chunk_size` keyword overrides `generation_config`, which
overrides the adapter's automatic choice. For example:

```python
output = model.generate(**inputs, max_new_tokens=32, prefill_chunk_size=512)
```

Chunk sizes must be positive multiples of 64. Explicitly passing
`prefill_chunk_size=None` requests one-shot prefill; larger one-shot shapes can
take much longer to compile on the current Spyre backend.

The cache policy includes the requested continuation length. Requests whose
allocated generation cache fits within the 2,048-token sliding window use a
fused SDPA decoder path. Larger requests use compact sliding-window caches and
retain full caches for global-attention layers. Each generation selects its own
cache policy, so a loaded model can alternate between short and long requests.
