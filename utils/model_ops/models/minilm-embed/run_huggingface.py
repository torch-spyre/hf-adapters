# Copyright 2025 The Torch-Spyre Authors.
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

"""Trace and generate YAML test cases for IBM Granite Embedding 278M Multilingual.

This script loads ``ibm-granite/granite-embedding-278m-multilingual`` from the
Hugging Face Hub, runs a single forward pass under :class:`TorchOpCollector`,
and writes the captured operator configurations to a YAML file consumable by
the downstream test harness.

Usage::

    python run_huggingface.py

Requirements:
    - A CUDA-capable GPU (enforced by ``require_cuda()``).
    - The model weights accessible via the Hugging Face Hub or a local cache.

Output:
    A YAML file named after the model (``granite-embedding-278m-multilingual``)
    containing one test-case entry per unique operator signature observed during
    the traced forward pass.  Supported dtypes written to the file:
    ``float16``, ``float32``, ``float64``, ``bfloat16``, ``half``,
    ``int32``, ``int64``, ``bool``.
"""

import os

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from utils.torchop_yaml import TorchOpCollector, require_cuda, setup_logging


def main():
    setup_logging()
    require_cuda()

    model_path = "sentence-transformers/all-MiniLM-L6-v2"
    prompt = "Hello " * 512

    device = "cuda"
    #base_model = SentenceTransformer(model_path, model_kwargs={"torch_dtype": torch.float16}, device=device)
    base_model = SentenceTransformer(model_path, device=device).to(torch.float16)

    model = torch.compile(base_model)

    features = base_model.tokenize(prompt)
    features = {
        n: v.to(device) if isinstance(v, torch.Tensor) else v for n, v in features.items()
    }

    with TorchOpCollector() as ctx:
        with torch.no_grad():
            outputs = model(features)
            embeddings = outputs["sentence_embedding"]
            embeddings = F.normalize(embeddings, p=2, dim=1)

    # print traced torch ops
    for op in ctx.ops_list:
        print(op)
    print(f"Total ops traced: {len(ctx.ops_list)}")

    # List of ops with generated test cases
    print("List of ops with test cases generated")
    for op in ctx.test_gen_ops:
        print(op, ctx.test_case_count[op])
    print(f"Total ops with test configs generated: {len(ctx.test_gen_ops)}")

    ctx.write_yaml(
        os.path.basename(model_path),
    )


if __name__ == "__main__":
    main()
