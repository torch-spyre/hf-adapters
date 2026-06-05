# Automated YAML File Generation

This module (and its submodules) enable automated generation of a yaml file with entries configuring test cases for operators needed for specific LLMs.

Each LLM for a yaml config file is to be generated has a driver scripts in folder models/\<model_name\>

Currently, we confirmed that a yaml file can be generated for the following models:

- granite 3.3
- granite 4.0 hybrid
- gpt-oss
- llama 3.1
- mistral small
- ministral3-14b


## How to generate a yaml file

If running for the first time, install the parent project together with the
`models-ops` extra from the repository root:

```
pip install -e ".[models-ops]"
```

The driver scripts require an NVIDIA GPU, so install a CUDA-enabled build of
PyTorch separately. The exact index URL depends on your CUDA version (replace
`cu128` with the build that matches your driver, e.g. `cu121`, `cu124`):

```
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128
```

Then change directory into `utils/models_ops/` to run the drivers (the absolute
import `from utils.torchop_yaml import ...` resolves against this directory).

Run the following command with an NVIDIA GPU. Multiple GPUs environment is not supported now.
More details on the yaml files can be found in [RFC](https://github.com/torch-spyre/rfcs/blob/main/0186-TestFrameworks/0186-TestFrameworks.md), [RFC](https://github.com/torch-spyre/rfcs/blob/main/1287-SpyreTestFramework/1287-SpyreTestFrameworkRFC.md), and [document](https://github.com/torch-spyre/torch-spyre/blob/main/tests/docs/input_args_enablement.md).

```
python -m models.<model folder>.run_huggingface
```

The desired level of logging can be controlled via the environment variable **TEST_GEN_LOGGING_LEVEL**, which can be set to standard python logging levels, namely, one of **DEBUG**, **INFO**, **WARNING**, **ERROR**, and **CRITICAL**.

The variable can be defined via command line or **.env** file in the current folder.

