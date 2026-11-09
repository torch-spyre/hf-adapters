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

"""Generate the Gemma4 op-test YAML by loading the model through spyre-inference.

Unlike ``run_huggingface.py`` this runs on a Spyre host, inside the
spyre-inference image (vLLM and the ``spyre_inference`` plugin are not
dependencies of this repository).
"""

import os

from utils.torchop_yaml import TorchOpCollector, setup_logging

MODEL_PATH = "google/gemma-4-26B-A4B-it"
PROMPT = "Say hello in one word."
MAX_TOKENS = 8


def main():
    setup_logging()

    # The engine must run in this process so the compile hook sees the graphs.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    # A compile-cache hit would skip compile_fx and record nothing.
    os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"

    with TorchOpCollector() as ctx:
        # Import vLLM only after the hook is installed, so it cannot bind the
        # original compile_fx first.
        from vllm import LLM, SamplingParams

        # No enforce_eager: nothing would be compiled, so nothing recorded.
        llm = LLM(
            model=MODEL_PATH,
            max_model_len=3072,
            max_num_seqs=8,
            tensor_parallel_size=1,
        )
        request_outputs = llm.generate([PROMPT], SamplingParams(max_tokens=MAX_TOKENS))

    request_output = request_outputs[0]
    input_len = len(request_output.prompt_token_ids)
    output_len = len(request_output.outputs[0].token_ids)

    # print traced torch op
    for op in ctx.ops_list:
        print(op)
    print(f"Total ops traced: {len(ctx.ops_list)}")

    # List of ops with generated test cases
    print("List of ops with test cases generated")
    for op in ctx.test_gen_ops:
        print(op, ctx.test_case_count[op])
    print(f"Total ops with test configs generated: {len(ctx.test_gen_ops)}")

    ctx.write_yaml(
        os.path.basename(MODEL_PATH),
        batch=1,
        input_len=input_len,
        output_len=output_len,
    )


if __name__ == "__main__":
    main()
