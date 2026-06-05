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
from datetime import datetime, timedelta

import torch
from huggingface_hub import hf_hub_download
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from transformers import Mistral3ForConditionalGeneration, StaticCache
from utils.torchop_yaml import TorchOpCollector, require_cuda, setup_logging


def load_system_prompt(repo_id: str, filename: str) -> str:
    file_path = hf_hub_download(repo_id=repo_id, filename=filename)
    with open(file_path, "r") as file:
        system_prompt = file.read()
    today = datetime.today().strftime("%Y-%m-%d")
    yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    model_name = repo_id.split("/")[-1]
    return system_prompt.format(name=model_name, today=today, yesterday=yesterday)


def main():
    setup_logging()
    require_cuda()

    model_path = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
    SYSTEM_PROMPT = load_system_prompt(model_path, "SYSTEM_PROMPT.txt")

    tokenizer = MistralTokenizer.from_hf_hub(model_path)

    device = "cuda"
    model = Mistral3ForConditionalGeneration.from_pretrained(
        model_path, device_map="auto", torch_dtype=torch.bfloat16
    )

    image_url = "https://static.wikia.nocookie.net/essentialsdocs/images/7/70/Battle.png/revision/latest?cb=20220523172438"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What action do you think I should take in this situation? List all the possible actions and explain why you think they are good or bad.",
                },
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        },
    ]

    tokenized = tokenizer.encode_chat_completion(
        ChatCompletionRequest(messages=messages)
    )

    input_ids = torch.tensor([tokenized.tokens]).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    pixel_values = (
        torch.tensor(tokenized.images[0], dtype=torch.bfloat16).unsqueeze(0).to(device)
    )
    image_sizes = torch.tensor([pixel_values.shape[-2:]]).to(device)

    past_key_values = StaticCache(config=model.config, max_cache_len=2048)

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    model.forward = torch.compile(model.forward)

    with TorchOpCollector() as ctx:
        with torch.no_grad():
            model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                max_new_tokens=2048,
                past_key_values=past_key_values,
                use_cache=True,
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

    print("=== START dump ===", flush=True)
    ctx.write_yaml(os.path.basename(model_path))
    print("=== END dump ===", flush=True)


if __name__ == "__main__":
    main()
