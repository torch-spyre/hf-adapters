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
from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM,
    StaticCache,
)
from utils.torchop_yaml import TorchOpCollector, require_cuda, setup_logging


def main():
    setup_logging()
    require_cuda()

    model_path = "google/gemma-4-12B-it"

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Where is the Thomas J. Watson Research Center located?",
                },
            ],
        },
    ]

    device = "cuda"
    model = AutoModelForMultimodalLM.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
    )
    tokenizer = AutoProcessor.from_pretrained(model_path)
    encoded_input = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True, 
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False
    ).to(device)
    input_ids = encoded_input["input_ids"]
    batch, input_len = input_ids.shape[0], input_ids.shape[-1]

    past_key_values = StaticCache(config=model.config, max_cache_len=2048)

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    model.forward = torch.compile(model.forward)

    with TorchOpCollector() as ctx:
        with torch.no_grad():
            outputs = model.generate(
                **encoded_input,
                max_new_tokens=512,
                past_key_values=past_key_values,
                use_cache=True,
            )
    output_len = len(outputs[0]) - input_len

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
