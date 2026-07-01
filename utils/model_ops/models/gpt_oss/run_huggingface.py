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
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
from utils.torchop_yaml import TorchOpCollector, require_cuda, setup_logging


def main():
    setup_logging()
    require_cuda()

    model_path = "openai/gpt-oss-20b"
    prompt = "Where is the Thomas J. Watson Research Center located?"

    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        dtype=torch.bfloat16,
        experts_implementation="batched_mm",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    encoded_input = tokenizer(prompt, return_tensors="pt").to(device)

    past_key_values = StaticCache(config=model.config, max_cache_len=2048)

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    model.forward = torch.compile(model.forward)

    with TorchOpCollector() as ctx:
        with torch.no_grad():
            model.generate(
                **encoded_input, past_key_values=past_key_values, use_cache=True
            )

    # print traced torch op
    for op in ctx.ops_list:
        print(op)
    print(f"Total ops traced: {len(ctx.ops_list)}")

    # List of ops with generated test cases
    print("List of ops with test cases generated")
    for op in ctx.test_gen_ops:
        print(op, ctx.test_case_count[op])
    print(f"Total ops with test configs generated: {len(ctx.test_gen_ops)}")

    ctx.write_yaml(os.path.basename(model_path))


if __name__ == "__main__":
    main()
