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

import os

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util
from PIL import Image
from utils.torchop_yaml import TorchOpCollector, require_cuda, setup_logging


def main():
    setup_logging()
    require_cuda()

    model_path = "sentence-transformers/clip-ViT-B-32"

    device = "cuda"
    base_model = SentenceTransformer(model_path, device=device).to(torch.float16)

    model = torch.compile(base_model)

    def _encode(inputs):
        features = base_model.tokenize(inputs)
        features = {
            n: v.to(device) if isinstance(v, torch.Tensor) else v for n, v in features.items()
        }

        with torch.inference_mode():
            embeddings = model(features)["sentence_embedding"]
            return F.normalize(embeddings, p=2, dim=1)

    with TorchOpCollector() as ctx:
        with Image.open("models/clip-ViT-B-32/two_dogs_in_snow.jpg") as image:
            img_emb = _encode([image.convert("RGB")]) 

        text_emb = _encode(
            ["Two dogs in the snow", "A cat on a table", "A picture of London at night"]
        )

    similarity_scores = util.cos_sim(img_emb, text_emb)

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
