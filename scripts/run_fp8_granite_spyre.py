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

"""Load an FP8 (E4M3) Granite checkpoint on Spyre, verify FP8 is active, generate.

Usage::

    python scripts/run_fp8_granite_spyre.py
    python scripts/run_fp8_granite_spyre.py --prompt "Hello" --max-new-tokens 16

Exits 0 when FP8 is active and generation completes, 1 otherwise. Output
correctness is not checked here; see tests/spyre/test_e2e_token_compare_spyre.py.
"""

import argparse
import sys
import time

from transformers import AutoTokenizer

from hf_adapters import AutoSpyreModelForCausalLM
from hf_adapters.fp8_linear import fp8_status
from hf_adapters.hf_common import encode_prompts

DEFAULT_MODEL = "ibm-granite/granite-3.3-8b-instruct-FP8"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()

    t0 = time.time()
    model = AutoSpyreModelForCausalLM.from_pretrained(args.model)
    print(f"Load time: {time.time() - t0:.1f}s")

    status = fp8_status(model)
    print(f"FP8 status: {status}")
    checks = {
        "FP8Linear present (swap ran)": status["n_fp8"] > 0,
        "no unswapped E4M3 nn.Linear": status["n_unswapped_e4m3"] == 0,
        "FP8Linear weights are [in, out]": status["orientation_ok"],
        "prequantized flag matches weight dtype": (
            status["n_prequantized"] == status["n_weight_fp8"]
        ),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        print("FAIL: " + "; ".join(failed))
        return 1

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    encoded = encode_prompts(tokenizer, [args.prompt])
    t0 = time.time()
    sequences = model.generate(
        **encoded, max_new_tokens=args.max_new_tokens, do_sample=False, timing=True
    )
    print(f"Generate time: {time.time() - t0:.1f}s")

    text = tokenizer.decode(
        sequences[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True
    )
    print(f"Output: {text!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
