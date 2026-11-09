# Automated YAML File Generation

This module (and its submodules) enable automated generation of a yaml file with entries configuring test cases for operators needed for specific LLMs.

Each LLM for a yaml config file is to be generated has a driver scripts in folder models/\<model_name\>

Currently, we confirmed that a yaml file can be generated for the following models:

- granite 3.3
- granite 4.0 hybrid
- granite 4.1
- gpt-oss
- llama 3.1
- mistral small
- ministral3-14b


## How to generate a yaml file

If running for the first time, install the parent project together with the
`models-ops` dependency group from the repository root:

```
uv sync --group models-ops
```

The driver scripts require an NVIDIA GPU, so install a CUDA-enabled build of
PyTorch separately. The exact index URL depends on your CUDA version (replace
`cu130` with the build that matches your driver, e.g. `cu121`, `cu124`):

```
uv pip install --upgrade --force-reinstall "torch==2.13.0+cu130" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
uv pip install mistral_common[opencv]
```

Then change directory into `utils/models_ops/` to run the drivers (the absolute
import `from utils.torchop_yaml import ...` resolves against this directory).

Run the following command with an NVIDIA GPU. Multiple GPUs environment is not supported now.
More details on the yaml files can be found in [RFC](https://github.com/torch-spyre/rfcs/blob/main/0186-TestFrameworks/0186-TestFrameworks.md), [RFC](https://github.com/torch-spyre/rfcs/blob/main/1287-SpyreTestFramework/1287-SpyreTestFrameworkRFC.md), and [document](https://github.com/torch-spyre/torch-spyre/blob/main/tests/docs/input_args_enablement.md).

```
uv run --no-sync python -m models.<model folder>.run_huggingface
```

The desired level of logging can be controlled via the environment variable **TEST_GEN_LOGGING_LEVEL**, which can be set to standard python logging levels, namely, one of **DEBUG**, **INFO**, **WARNING**, **ERROR**, and **CRITICAL**.

The variable can be defined via command line or **.env** file in the current folder.

## Generating a yaml file through spyre-inference (Spyre host)

`models/gemma4-26b-a4b/run_spyre_inference.py` generates the Gemma4 yaml by loading
`google/gemma-4-26B-A4B-it` through spyre-inference (vLLM with the `spyre_inference`
plugin) instead of stock HuggingFace on CUDA. The generated yaml is kept in
[torch-spyre/spyre-inference](https://github.com/torch-spyre/spyre-inference), not in this repository.

It requires a Spyre host and the spyre-inference image. vLLM and the plugin are **not**
dependencies of this repository (as with the vLLM script in `utils/module_discovery`), so
do not run `uv sync` for it. Run it from `utils/model_ops/`:

```
python -m models.gemma4-26b-a4b.run_spyre_inference
```

The yaml is written to the current directory. Notes:

- The model is gated, so `HF_TOKEN` must be set.
- The first run downloads about 52 GB of weights and is slow.
- Only one process may use a Spyre card at a time.
- vLLM runs in-process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) and with its compile cache
  disabled (`VLLM_DISABLE_COMPILE_CACHE=1`); otherwise the collector would see no graphs.
