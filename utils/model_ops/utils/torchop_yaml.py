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


import copy
import logging
import math
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

import regex as re
import torch
import torch._inductor.compile_fx
import yaml
from dotenv import load_dotenv
from torch.fx.experimental.symbolic_shapes import has_free_symbols, is_concrete_int


USE_OLDFORMAT = os.getenv("USE_OLDFORMAT", None) is not None

def _get_transformers_version():
    ver = os.getenv("TRANSFORMERS_VERSION", None)
    if ver:
        return ver
    try:
        import transformers

        return transformers.__version__
    except ImportError:
        return "main"


_TRANSFORMERS_VERSION = _get_transformers_version()
_TRANSFORMERS_PATH_RE = re.compile(r"[^\s]*site-packages/transformers/(.*?\.py):(\d+)")


def setup_logging():
    """Configure logging for TorchOpCollector.

    Reads ``TEST_GEN_LOGGING_LEVEL`` and ``LOGGING_METHOD`` from the environment
    (after loading ``.env``), attaches a single ``StreamHandler`` to the
    ``TorchOpCollector`` logger, and updates
    ``TorchOpCollector.log_function`` / ``TorchOpCollector.log_mthd`` so existing
    call sites pick up the chosen method. Idempotent: safe to call more than
    once.

    Must be called explicitly by the application entry point. Importing this
    module no longer configures any logger.
    """
    # Suppress warnings/errors from symbolic_shapes and recording
    logging.getLogger("torch.fx.experimental.symbolic_shapes").setLevel(
        logging.CRITICAL
    )
    logging.getLogger("torch.fx.experimental.recording").setLevel(logging.CRITICAL)

    load_dotenv()

    logger = logging.getLogger("TorchOpCollector")
    log_level_str = os.getenv("TEST_GEN_LOGGING_LEVEL", "DEBUG")
    log_level_int = logging.getLevelName(log_level_str)
    logger.setLevel(log_level_int)
    if not any(getattr(h, "_torchop_yaml", False) for h in logger.handlers):
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch._torchop_yaml = True  # type: ignore[attr-defined]
        logger.addHandler(ch)

    logger.debug(
        f"Current logging level: {logging.getLevelName(logger.getEffectiveLevel())}"
    )

    TorchOpCollector.log_function = {
        "PRINT": print,
        "LOGGER": logger.debug,
    }
    TorchOpCollector.log_mthd = os.getenv("LOGGING_METHOD", "PRINT")
    TorchOpCollector.logger = logger
    return logger


def require_cuda():
    """Abort early if no CUDA-enabled PyTorch build is available.

    The driver scripts run on an NVIDIA GPU; install a CUDA wheel of torch
    (e.g. ``pip install torch --index-url https://download.pytorch.org/whl/cu128``)
    before invoking them.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. The YAML generators require an NVIDIA GPU "
            "and a CUDA-enabled build of PyTorch. Install one with, e.g.: "
            "pip install --upgrade torch --index-url "
            "https://download.pytorch.org/whl/cu128"
        )
    if "," in os.getenv("CUDA_VISIBLE_DEVICES", ""):
        raise RuntimeError("CUDA_VISIBLE_DEVICES should specify only one card")


def _convert_transformers_path_to_url(comments):
    """Replace site-packages/transformers/path/to/file.py:LINE with a GitHub URL."""
    if "site-packages" not in comments:
        return comments

    def _replace(m):
        rel_path = m.group(1)
        lineno = m.group(2)
        url = (
            f"https://github.com/huggingface/transformers/blob/"
            f"v{_TRANSFORMERS_VERSION}/src/transformers/{rel_path}#L{lineno}"
        )
        if os.getenv("CHECK_URL_BY_REQUEST", None):
            try:
                req = urllib.request.Request(url, method="HEAD")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status >= 400:
                        logging.getLogger("TorchOpCollector").warning(
                            f"URL returned HTTP {resp.status}: {url}"
                        )
            except urllib.error.HTTPError as e:
                logging.getLogger("TorchOpCollector").warning(
                    f"URL validation failed ({e.code}): {url}"
                )
            except Exception as e:
                logging.getLogger("TorchOpCollector").warning(
                    f"URL validation error ({e}): {url}"
                )
        return url

    return _TRANSFORMERS_PATH_RE.sub(_replace, comments)


class FlowList(list):
    pass


def repr_flow_seq(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


yaml.add_representer(FlowList, repr_flow_seq)


def sanitize_arg(
    arg, shape, dtype, randintlim=1000, stride=None, storage_offset=None, device=None
):
    logging.getLogger("TorchOpCollector").debug(
        f"In sanitize_arg: {arg}, {shape}, {dtype}, {randintlim}, {stride}, {storage_offset}"
    )

    if isinstance(arg, torch.fx.node.Node):
        # Include stride and storage_offset in the representation for deduplication
        stride_str = f"_stride{stride}" if stride is not None else ""
        offset_str = f"_offset{storage_offset}" if storage_offset is not None else ""
        device_str = f", device={device}" if device is not None else ""

        if "float" in str(dtype):
            return f"torch.rand(torch.Size({shape}), dtype={dtype}{device_str}){stride_str}{offset_str}"
        elif "int" in str(dtype):
            return f"torch.randint({randintlim}, torch.Size({shape}), dtype={dtype}{device_str}){stride_str}{offset_str}"
        elif "bool" in str(dtype):
            return f"torch.rand(torch.Size({shape}){device_str}) < 0.9{stride_str}{offset_str}"
        else:
            print(f"Unhandled dtype {dtype}")
            return str(arg)

    return str(arg)


def format_tensor_details(
    arg, shape, stride, storage_offset, dtype, device, randintlim=1000
):
    logging.getLogger("TorchOpCollector").debug(
        f"In format_tensor_details: {arg}, {shape}, {stride}, {storage_offset}, {dtype}, {device}, {randintlim}"
    )
    tensor_details = {}

    if isinstance(arg, torch.fx.node.Node):
        tensor_details["shape"] = FlowList(shape)
        tensor_details["stride"] = FlowList(stride)
        tensor_details["storage_offset"] = storage_offset
        tensor_details["dtype"] = str(dtype)
        tensor_details["device"] = str(device)
        if "float" in str(dtype):
            tensor_details["init"] = "rand"
        elif "int" in str(dtype):
            tensor_details["init"] = "randint"
            tensor_details["init_args"] = {"high": randintlim}
        elif "bool" in str(dtype):
            pass
        else:
            print(f"Unhandled dtype {dtype}")
            return {}

        return tensor_details


def add_test_case_yaml(
    node_name,
    target_name,
    inputs_list,
    kwmap,
    func_comments,
    logger=None,
    unnorm_args_script=True,
    old_format=True,
    **kwargs,
):
    """
    kwargs: a list of boolean variables round_up, to_float16_fromfp32, to_float16_frombf16_dim_redn, and const_arg whether the operator as found in the model includes arguments that need  padding to
    be a multiple of stick size, are FP32, exceed 3d, and are constants, respectively.
    """
    test_case_yaml: dict[str, Any] = {}
    if not old_format:
        op_name = "name"
        inputs = "sample_inputs_func"
        inputs_list = {"args": inputs_list}
        test_case_yaml = {
            op_name: target_name,
            inputs: inputs_list,
            "description": func_comments,
        }
    else:
        op_name = "op"
        inputs = "inputs"
        test_case_yaml = {
            "name": node_name,
            op_name: target_name,
            inputs: inputs_list,
            "description": func_comments,
        }

    if unnorm_args_script:
        markers = []
        if "round_up" in kwargs:
            if kwargs["round_up"]:
                markers.append("paddedtensor")
        if "to_float16_fromfp32" in kwargs:
            if kwargs["to_float16_fromfp32"]:
                markers.append("fp32operation")
        if "to_float16_frombf16" in kwargs:
            if kwargs["to_float16_frombf16"]:
                markers.append("bf16operation")
        if "dim_redn" in kwargs:
            if kwargs["dim_redn"]:
                markers.append("largedimtensor")
        if "const_arg" in kwargs:
            if kwargs["const_arg"]:
                markers.append("constant")
        if "tensors_cpu" in kwargs:
            if kwargs["tensors_cpu"]:
                markers.append("tensorsoncpu")
        if not old_format:
            markers.append(node_name)

        if len(markers) > 0:
            if not old_format:
                test_case_yaml["tags"] = markers
            else:
                if len(markers) == 1:
                    test_case_yaml["marks"] = markers[0]
                else:
                    test_case_yaml["marks"] = markers
    else:
        markers = []
        if "const_arg" in kwargs:
            if kwargs["const_arg"]:
                markers.append("constant")
        if "tensors_cpu" in kwargs:
            if kwargs["tensors_cpu"]:
                markers.append("tensorsoncpu")
        if not old_format:
            markers.append(node_name)

        if len(markers) > 0:
            if not old_format:
                test_case_yaml["tags"] = markers
            else:
                if len(markers) == 1:
                    test_case_yaml["marks"] = markers[0]
                else:
                    test_case_yaml["marks"] = markers

    if len(kwmap) > 0:
        if not old_format:
            test_case_yaml["kwargs"] = kwmap
        else:
            test_case_yaml["kwmap"] = kwmap

    return test_case_yaml


@dataclass
class _ArgResult:
    skip_this_node: bool = False
    tensors_cpu: bool = False
    const_arg: bool = False
    round_up: bool = False
    dim_redn: bool = False
    to_float16_fromfp32: bool = False
    to_float16_frombf16: bool = False
    arg_comments: str = ""
    san_arg: object = None
    san_arg_norm: object = None
    san_arg_comp: object = None
    yaml_input: dict[str, Any] = field(default_factory=dict)
    yaml_input_norm: dict[str, Any] = field(default_factory=dict)
    orig_shape: object = None
    shrunk_shape: object = None
    new_saved_shape: object = None


class TorchOpCollector:
    DEFAULT_YAML_DEFAULTS = {"dtype": "fp16", "seed": 123, "atol": 5e-3, "rtol": 5e-3}

    graph_id = 0
    op_seq_num = 0
    test_gen_ops: list[Any] = []
    test_gen_ops_set: set[Any] = set()
    test_case_count: dict[Any, Any] = {}
    op_param_map: dict[Any, Any] = {}
    test_cases_yaml: list[dict[str, Any]] = []
    test_cases_norm_yaml: list[dict[str, Any]] = []

    # Defaults; ``setup_logging()`` overwrites these once the application entry
    # point configures logging. The class-attribute interface is preserved so
    # that existing ``TorchOpCollector.log_function[TorchOpCollector.log_mthd]``
    # call sites keep working without modification.
    log_function = {
        "PRINT": print,
        "LOGGER": logging.getLogger("TorchOpCollector").debug,
    }
    log_mthd = "PRINT"
    logger = logging.getLogger("TorchOpCollector")

    # ---- helpers for collect_torchops ----------------------------------------

    @staticmethod
    def _resolve_node_comments(comments):
        if comments is None:
            return None
        st_index = comments.find("site-packages")
        if st_index != -1:
            comments = comments[st_index:]
        return _convert_transformers_path_to_url(comments)

    _INDEX_INTLIMIT_OPS = [
        "torch.scatter",
        "torch.scatter_",
        "torch.scatter_add",
        "torch.scatter_reduce",
        "torch.index_add",
        "torch.index_add_",
        "torch.index_copy",
        "torch.index_copy_",
        "torch.index_reduce",
        "torch.index_select",
        "torch.select_scatter",
    ]
    _SPECIAL_INTLIMIT_OPS = _INDEX_INTLIMIT_OPS + ["torch.getitem"]

    @staticmethod
    def _compute_randintlimit(op_name, i, dtype, saved_shape, san_args):
        if op_name not in TorchOpCollector._SPECIAL_INTLIMIT_OPS:
            return 1000
        if i == 0:
            return 1000
        if op_name == "torch.getitem" and "int" in str(dtype):
            TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                f"i: {i}, saved_shape: {saved_shape}, op_name: {op_name}, dtype: {dtype}, san_args: {san_args}"
            )
            return saved_shape[0]
        if op_name in TorchOpCollector._INDEX_INTLIMIT_OPS:
            dim_index = 2 if op_name == "torch.select_scatter" else 1
            if i == dim_index + 1 and "int" in str(dtype):
                TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                    f"i: {i}, saved_shape: {saved_shape}, op_name: {op_name}, dtype: {dtype}, san_args: {san_args}"
                )
                limit = saved_shape[int(san_args[dim_index])]
                TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                    f"randintlimit for {op_name}: {limit}"
                )
                return limit
        return 1000

    @staticmethod
    def _process_node_arg(arg, op_name, i, saved_shape, san_args, out_device):
        result = _ArgResult()
        op_name_arg, shape, stride, storage_offset, dtype, device, comments = (
            TorchOpCollector.get_node_shape(arg)
        )
        if out_device is not None and device is not None and "cpu" in device:
            result.tensors_cpu = True
        comments = TorchOpCollector._resolve_node_comments(comments)
        if comments is not None:
            result.arg_comments = "\n" + comments
        TorchOpCollector.log_function[TorchOpCollector.log_mthd](
            f"Arg COMMENTS: {comments}"
        )
        TorchOpCollector.log_function[TorchOpCollector.log_mthd](
            f"Arg #{i}: {op_name_arg}, {shape}, {stride}, {storage_offset}, {dtype}, {device}"
        )
        if op_name_arg is None:
            TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                f"Skipping arg {i} for {op_name} as the op_name of this arg is empty"
            )
            return result
        if shape is not None and isinstance(shape, str) and "symbolic" in shape:
            TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                f"Skipping {op_name} as one of its args has symbolic shape with free symbols"
            )
            result.skip_this_node = True
            return result

        # track saved_shape for scatter/index ops at i==0
        if op_name in TorchOpCollector._SPECIAL_INTLIMIT_OPS and i == 0:
            result.new_saved_shape = shape
            TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                f"Saving shape for scatter: {shape}"
            )

        randintlimit = TorchOpCollector._compute_randintlimit(
            op_name, i, dtype, saved_shape, san_args
        )

        if dtype is not None and "scalar" in str(dtype):
            TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                f"adding {shape} for {arg} to san_args"
            )
            result.const_arg = True
            result.san_arg = shape
            result.san_arg_norm = shape
            result.san_arg_comp = shape
            result.yaml_input = {"value": shape}
            result.yaml_input_norm = {"value": shape}
        elif isinstance(shape, tuple) and isinstance(dtype, tuple):
            assert len(shape) == len(dtype)
            args_tuple = ()
            args_tuple_norm = ()
            tensor_list_input = []
            tensor_list_input_norm = []
            for idx in range(len(shape)):
                shrunk_shape, mod_dtype = TorchOpCollector.normalize_shape_type(
                    shape[idx], dtype[idx]
                )
                (
                    result.round_up,
                    result.dim_redn,
                    result.to_float16_fromfp32,
                    result.to_float16_frombf16,
                ) = TorchOpCollector.get_normalization_flags(
                    shape[idx], dtype[idx], shrunk_shape, mod_dtype
                )
                args_tuple = args_tuple + (
                    sanitize_arg(
                        arg,
                        shape[idx],
                        dtype[idx],
                        stride=stride[idx],
                        storage_offset=storage_offset[idx],
                        device=device[idx],
                    ),
                )
                args_tuple_norm = args_tuple_norm + (
                    sanitize_arg(
                        arg,
                        shrunk_shape,
                        mod_dtype,
                        stride=stride[idx],
                        storage_offset=storage_offset[idx],
                        device=device[idx],
                    ),
                )
                tensor = format_tensor_details(
                    arg,
                    shape[idx],
                    stride[idx],
                    storage_offset[idx],
                    dtype[idx],
                    device[idx],
                )
                tensor_list_input.append(tensor)
                tensor_norm = format_tensor_details(
                    arg,
                    shrunk_shape,
                    stride[idx],
                    storage_offset[idx],
                    mod_dtype,
                    device[idx],
                )
                tensor_list_input_norm.append(tensor_norm)
            result.san_arg = args_tuple
            result.san_arg_comp = args_tuple
            result.yaml_input = {"tensor_list": tensor_list_input}
            if op_name in ["torch.getitem"]:
                result.san_arg_norm = args_tuple_norm[len(args_tuple_norm) - 3 :]
                result.yaml_input_norm = {
                    "tensor_list": tensor_list_input_norm[len(args_tuple_norm) - 3 :]
                }
            else:
                result.san_arg_norm = args_tuple_norm
                result.yaml_input_norm = {"tensor_list": tensor_list_input_norm}
            TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                f"Tuple of san_args: {result.san_arg_norm}"
            )
        elif shape is None and dtype is None:
            result.skip_this_node = True
        else:
            assert isinstance(shape, list)
            this_san_arg = sanitize_arg(
                arg,
                shape,
                dtype,
                randintlimit,
                stride=stride,
                storage_offset=storage_offset,
                device=device,
            )
            tensor = format_tensor_details(
                arg, shape, stride, storage_offset, dtype, device, randintlimit
            )
            shrunk_shape, mod_dtype = TorchOpCollector.normalize_shape_type(
                shape, dtype
            )
            TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                f"After normalization: {shape}, {dtype}, {shrunk_shape}, {mod_dtype}"
            )
            (
                result.round_up,
                result.dim_redn,
                result.to_float16_fromfp32,
                result.to_float16_frombf16,
            ) = TorchOpCollector.get_normalization_flags(
                shape, dtype, shrunk_shape, mod_dtype
            )
            result.san_arg = this_san_arg
            result.san_arg_comp = this_san_arg
            result.san_arg_norm = sanitize_arg(
                arg,
                shrunk_shape,
                mod_dtype,
                randintlimit,
                stride=stride,
                storage_offset=storage_offset,
                device=device,
            )
            result.yaml_input = {"tensor": tensor}
            result.yaml_input_norm = {
                "tensor": format_tensor_details(
                    arg,
                    shrunk_shape,
                    stride,
                    storage_offset,
                    mod_dtype,
                    device,
                    randintlimit,
                )
            }
            result.orig_shape = shape
            result.shrunk_shape = shrunk_shape
        return result

    @staticmethod
    def _process_scalar_arg(arg, op_name):
        result = _ArgResult()
        result.const_arg = True
        input_type_str = "value"
        str_arg = arg
        if arg is not None and (
            isinstance(arg, torch.device)
            or isinstance(arg, tuple)
            or isinstance(arg, slice)
        ):
            str_arg = str(arg)
        tensor_list_input = []
        tensor_list_input_norm = []
        arg_list = []
        tuple_input = ()
        if isinstance(arg, tuple):
            for t_arg in arg:
                TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                    f"Tuple member: {t_arg}, type: {type(t_arg)}"
                )
                if isinstance(t_arg, slice):
                    input_type_str = "py"
                    TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                        f"Params of slice: {t_arg.start}:{type(t_arg.start)}, {t_arg.step}:{type(t_arg.step)}, {t_arg.stop}:{type(t_arg.stop)}"
                    )
                    if (
                        TorchOpCollector.is_symbolic_list(t_arg.start)
                        or TorchOpCollector.is_symbolic_list(t_arg.step)
                        or TorchOpCollector.is_symbolic_list(t_arg.stop)
                    ):
                        result.skip_this_node = True
                elif isinstance(t_arg, int):
                    tuple_input = tuple_input + (t_arg,)
                if isinstance(t_arg, torch.fx.node.Node):
                    TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                        f"Node opcode: {t_arg.op} Target type: {type(t_arg.target)} {t_arg.name}, {t_arg.args}"
                    )
                    TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                        "Node element of tuple"
                    )
                    TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                        f"Node opcode: {t_arg.op} Target type: {type(t_arg.target)} {t_arg.name}, {t_arg.args}"
                    )
                    if TorchOpCollector.is_symbolic_list(t_arg):
                        result.skip_this_node = True
                    (
                        opel,
                        shapel,
                        stridel,
                        storage_offsetl,
                        dtypel,
                        devicel,
                        comments,
                    ) = TorchOpCollector.get_node_shape(t_arg)
                    TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                        f"Node tuple member details: {opel}, {shapel}, {dtypel}"
                    )
                    skip_this_node2, skip_arg_and_continue, is_scalar, is_list = (
                        TorchOpCollector.preprocess_node_contents(opel, shapel, dtypel)
                    )
                    if skip_this_node2:
                        result.skip_this_node = True
                        break
                    if devicel is not None and "cpu" in devicel:
                        result.tensors_cpu = True
                    if is_scalar:
                        tuple_input = tuple_input + (shapel,)
                    elif is_list:
                        arg_list.append(
                            sanitize_arg(
                                t_arg,
                                shapel,
                                dtypel,
                                stride=stridel,
                                storage_offset=storage_offsetl,
                                device=devicel,
                            )
                        )
                        tensor_details = format_tensor_details(
                            t_arg, shapel, stridel, storage_offsetl, dtypel, devicel
                        )
                        tensor_list_input.append(tensor_details)
                        shrunk_shape, mod_dtype = TorchOpCollector.normalize_shape_type(
                            shapel, dtypel
                        )
                        tensor_details_norm = format_tensor_details(
                            t_arg,
                            shrunk_shape,
                            stridel,
                            storage_offsetl,
                            mod_dtype,
                            devicel,
                        )
                        tensor_list_input_norm.append(tensor_details_norm)
                        TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                            f"After normalization: {shapel}, {dtypel}, {shrunk_shape}, {mod_dtype}"
                        )
                        (
                            result.round_up,
                            result.dim_redn,
                            result.to_float16_fromfp32,
                            result.to_float16_frombf16,
                        ) = TorchOpCollector.get_normalization_flags(
                            shapel, dtypel, shrunk_shape, mod_dtype
                        )
                    if comments is not None:
                        result.arg_comments += "\n" + comments
        if len(tensor_list_input) > 0:
            result.san_arg = arg_list
            result.san_arg_norm = arg_list
            result.san_arg_comp = arg_list
            if len(tensor_list_input) > 1 or op_name == "torch.cat":
                result.yaml_input = {"tensor_list": tensor_list_input}
                result.yaml_input_norm = {"tensor_list": tensor_list_input_norm}
            else:
                result.yaml_input = {"tensor": tensor_list_input[0]}
                result.yaml_input_norm = {"tensor": tensor_list_input_norm[0]}
        elif len(tuple_input) > 0:
            result.san_arg = tuple_input
            result.san_arg_norm = tuple_input
            result.san_arg_comp = tuple_input
            TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                f"tuple_input: {tuple_input}"
            )
            result.yaml_input = {input_type_str: str(tuple_input)}
            result.yaml_input_norm = {input_type_str: str(tuple_input)}
        else:
            result.san_arg = str_arg
            result.san_arg_norm = str_arg
            result.san_arg_comp = type(arg)
            TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                f"str_arg: {str_arg}"
            )
            # Convert torch.dtype objects to strings for YAML serialization
            yaml_value = str(str_arg) if isinstance(arg, torch.dtype) else str_arg
            result.yaml_input = {input_type_str: yaml_value}
            result.yaml_input_norm = {input_type_str: yaml_value}
        return result

    @staticmethod
    def _process_list_arg(arg, op_name):
        result = _ArgResult()
        TorchOpCollector.log_function[TorchOpCollector.log_mthd](
            f"Immutable list arg: {arg}"
        )
        are_all_ints = True
        arg_list = []
        tensor_list_input = []
        tensor_list_input_norm = []
        for el in arg:
            TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                f"element {el} is of type {type(el)}"
            )
            if isinstance(el, (int, float)):
                arg_list.append(el)
            else:
                if isinstance(el, torch.fx.node.Node):
                    (
                        opel,
                        shapel,
                        stridel,
                        storage_offsetl,
                        dtypel,
                        devicel,
                        comments,
                    ) = TorchOpCollector.get_node_shape(el)
                    TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                        f"Node list member details: {opel}, {shapel}, {dtypel}, {devicel}"
                    )
                    skip_this_node2, skip_arg_and_continue, is_scalar, is_list = (
                        TorchOpCollector.preprocess_node_contents(opel, shapel, dtypel)
                    )
                    if is_scalar:
                        arg_list.append(shapel)
                        continue
                    are_all_ints = False
                    if skip_this_node2:
                        result.skip_this_node = True
                    if skip_arg_and_continue:
                        continue
                    if devicel is not None and "cpu" in devicel:
                        result.tensors_cpu = True
                    if is_list:
                        arg_list.append(
                            sanitize_arg(
                                el,
                                shapel,
                                dtypel,
                                stride=stridel,
                                storage_offset=storage_offsetl,
                                device=devicel,
                            )
                        )
                        tensor_details = format_tensor_details(
                            el, shapel, stridel, storage_offsetl, dtypel, devicel
                        )
                        tensor_list_input.append(tensor_details)
                        shrunk_shape, mod_dtype = TorchOpCollector.normalize_shape_type(
                            shapel, dtypel
                        )
                        tensor_details_norm = format_tensor_details(
                            el,
                            shrunk_shape,
                            stridel,
                            storage_offsetl,
                            mod_dtype,
                            devicel,
                        )
                        tensor_list_input_norm.append(tensor_details_norm)
                        TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                            f"After normalization: {shapel}, {dtypel}, {shrunk_shape}, {mod_dtype}, {devicel}"
                        )
                        (
                            result.round_up,
                            result.dim_redn,
                            result.to_float16_fromfp32,
                            result.to_float16_frombf16,
                        ) = TorchOpCollector.get_normalization_flags(
                            shapel, dtypel, shrunk_shape, mod_dtype
                        )
                        if comments is not None:
                            result.arg_comments += "\n" + comments
                else:
                    TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                        f"Elements of immutable list neither numeric nor node: type: {type(el)}"
                    )
        if are_all_ints:
            TorchOpCollector.log_function[TorchOpCollector.log_mthd]("Numeric list arg")
            result.san_arg = arg_list
            result.san_arg_norm = arg_list
            result.san_arg_comp = arg_list
            result.yaml_input = {"value": FlowList(arg_list)}
            result.yaml_input_norm = {"value": FlowList(arg_list)}
            TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                f"yaml_inputs: {result.yaml_input}, {result.yaml_input_norm}"
            )
        elif not result.skip_this_node and len(tensor_list_input) > 0:
            TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                f"List of non-numeric arguments: {arg_list}"
            )
            result.san_arg = arg_list
            result.san_arg_norm = arg_list
            result.san_arg_comp = arg_list
            keyword = "tensor"
            if len(tensor_list_input) > 1 or op_name == "torch.cat":
                keyword = "tensor_list"
            else:
                tensor_list_input = tensor_list_input[0]
            result.yaml_input = {keyword: tensor_list_input}
            result.yaml_input_norm = {keyword: tensor_list_input_norm}
        return result

    @staticmethod
    def _pack_reshape_args(
        op_name,
        san_args,
        san_args_norm,
        yaml_inputs,
        yaml_inputs_norm,
        dim_redn,
        round_up,
        orig_shape,
        shrunk_shape,
    ):
        if op_name not in ["torch.reshape", "torch.view", "torch.permute"]:
            return san_args, san_args_norm, yaml_inputs, yaml_inputs_norm
        if len(san_args) <= 1:
            return san_args, san_args_norm, yaml_inputs, yaml_inputs_norm
        TorchOpCollector.log_function[TorchOpCollector.log_mthd](
            f"Converting scalar args {san_args} to tuple for {op_name}"
        )
        start_index_norm = 1
        if dim_redn:
            TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                f"Dim reduction: {orig_shape}, {shrunk_shape}, {start_index_norm}"
            )
            dims_removed = len(orig_shape) - len(shrunk_shape)
            start_index_norm = max(dims_removed, 0) + 1
        val_tuple = tuple(int(ii) for ii in san_args[1:])
        val_tuple_norm = tuple(int(ii) for ii in san_args[start_index_norm:])
        if op_name == "torch.reshape":
            if round_up:
                TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                    f"Round up changes: {val_tuple}, {val_tuple_norm}, {orig_shape}"
                )
            if -1 not in val_tuple_norm:
                tot_elems = math.prod(el for el in shrunk_shape)
                reshaped_elems = math.prod(el for el in val_tuple_norm)
                TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                    f"Adjusting total elem count: {tot_elems} {reshaped_elems}"
                )
                if reshaped_elems < tot_elems:
                    val_tuple_norm = tuple(
                        (
                            int(val_tuple_norm[ii])
                            if ii > 0
                            else int(val_tuple_norm[0] * tot_elems / reshaped_elems)
                        )
                        for ii in range(len(val_tuple_norm))
                    )
        elif op_name == "torch.permute":
            val_tuple_pos = tuple(
                i if i >= 0 else i + len(orig_shape) for i in val_tuple
            )
            is_val_tuple_neg = tuple(0 if i >= 0 else 1 for i in val_tuple)
            val_tuple_norm = tuple(
                (
                    val_tuple_pos[i] - len(shrunk_shape)
                    if is_val_tuple_neg[i]
                    else val_tuple_pos[i]
                )
                for i in range(len(val_tuple_pos))
                if val_tuple_pos[i] < len(shrunk_shape)
            )
        yaml_inputs = [yaml_inputs[0], {"value": str(val_tuple)}]
        yaml_inputs_norm = [yaml_inputs_norm[0], {"value": str(val_tuple_norm)}]
        san_args = [san_args[0], val_tuple]
        san_args_norm = [san_args[0], val_tuple_norm]
        return san_args, san_args_norm, yaml_inputs, yaml_inputs_norm

    @staticmethod
    def _emit_test_case(
        op_name,
        yaml_inputs,
        yaml_inputs_norm,
        kwmap,
        node_kwargs,
        node_comments,
        san_args_comp,
        round_up,
        to_float16_fromfp32,
        to_float16_frombf16,
        dim_redn,
        const_arg,
        tensors_cpu,
    ):
        if len(yaml_inputs) == 0:
            TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                "Skipping operator since no valid input has been detected"
            )
            return
        if (
            op_name in TorchOpCollector.op_param_map
            and san_args_comp in TorchOpCollector.op_param_map[op_name]
        ):
            TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                f"Args already handled for {op_name}"
            )
            return
        TorchOpCollector.log_function[TorchOpCollector.log_mthd](
            f"Arguments seen for first time for {op_name}. Adding a new test case"
        )
        op_name_with_seqno = (
            op_name
            + "."
            + (
                str(TorchOpCollector.test_case_count[op_name] + 1)
                if op_name in TorchOpCollector.test_case_count
                else "1"
            )
        )
        TorchOpCollector.log_function[TorchOpCollector.log_mthd](
            f"Test case name: {op_name_with_seqno}"
        )
        tc_yaml = add_test_case_yaml(
            op_name_with_seqno,
            op_name,
            yaml_inputs,
            kwmap,
            node_comments,
            TorchOpCollector.logger,
            True,
            old_format=USE_OLDFORMAT,
            round_up=round_up,
            to_float16_fromfp32=to_float16_fromfp32,
            to_float16_frombf16=to_float16_frombf16,
            dim_redn=dim_redn,
            const_arg=const_arg,
            tensors_cpu=tensors_cpu,
        )
        TorchOpCollector.test_cases_yaml.append(tc_yaml)
        if node_kwargs:
            for k, v in node_kwargs.items():
                if k == "dtype":
                    if "bfloat16" in str(v):
                        to_float16_frombf16 = True
                    elif "float32" in str(v):
                        to_float16_fromfp32 = True
        if (
            round_up
            or to_float16_fromfp32
            or to_float16_frombf16
            or dim_redn
            or const_arg
            or tensors_cpu
        ):
            TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                "Creating round_up version of the yaml"
            )
            kwmap_norm = copy.deepcopy(kwmap)
            if (
                len(kwmap_norm) > 0
                and "dtype" in kwmap_norm
                and (to_float16_fromfp32 or to_float16_frombf16)
            ):
                kwmap_norm["dtype"] = str(torch.float16)
                if "fill_value" in kwmap_norm:
                    fill_val = kwmap_norm["fill_value"]
                    if isinstance(fill_val, (int, float)):
                        FLOAT16_MAX = torch.finfo(torch.float16).max
                        FLOAT16_MIN = torch.finfo(torch.float16).min
                        if fill_val > FLOAT16_MAX:
                            TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                                f"Capping fill_value from {fill_val} to {FLOAT16_MAX} for float16 compatibility"
                            )
                            kwmap_norm["fill_value"] = FLOAT16_MAX
                        elif fill_val < FLOAT16_MIN:
                            TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                                f"Capping fill_value from {fill_val} to {FLOAT16_MIN} for float16 compatibility"
                            )
                            kwmap_norm["fill_value"] = FLOAT16_MIN
            tc_yaml = add_test_case_yaml(
                op_name_with_seqno + "_spyre",
                op_name,
                yaml_inputs_norm,
                kwmap_norm,
                node_comments,
                TorchOpCollector.logger,
                False,
                old_format=USE_OLDFORMAT,
                round_up=round_up,
                to_float16_fromfp32=to_float16_fromfp32,
                to_float16_frombf16=to_float16_frombf16,
                dim_redn=dim_redn,
                const_arg=const_arg,
                tensors_cpu=tensors_cpu,
            )
        TorchOpCollector.test_cases_norm_yaml.append(tc_yaml)
        if tc_yaml != {}:
            TorchOpCollector.op_seq_num += 1
            TorchOpCollector.test_gen_ops_set.add(op_name)
            if op_name in TorchOpCollector.test_case_count:
                TorchOpCollector.test_case_count[op_name] += 1
                TorchOpCollector.op_param_map[op_name].append(san_args_comp)
            else:
                TorchOpCollector.test_case_count[op_name] = 1
                TorchOpCollector.op_param_map[op_name] = [san_args_comp]

    # ---- main collector (coordinator) ----------------------------------------

    @staticmethod
    def collect_torchops(gm, ops_set, print_output):
        output_list = [""]
        TorchOpCollector.log_function[TorchOpCollector.log_mthd](
            f"New Graph: graph {TorchOpCollector.graph_id}"
        )
        TorchOpCollector.log_function[TorchOpCollector.log_mthd](
            f"New Graph: graph {TorchOpCollector.graph_id}"
        )
        for node in gm.graph.nodes:
            if str(node.op) != "call_function" and str(node.op) != "call_method":
                continue
            skip_this_node = False
            round_up = to_float16_fromfp32 = to_float16_frombf16 = dim_redn = (
                const_arg
            ) = tensors_cpu = False
            orig_shape = shrunk_shape = None
            TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                f"Start processing new node of graph {TorchOpCollector.graph_id}"
            )
            TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                f"op: {node.op} node.name: {node.name}"
            )

            (
                op_name,
                out_shape,
                out_stride,
                out_storage_offset,
                out_dtype,
                out_device,
                comments,
            ) = TorchOpCollector.get_node_shape(node)
            if out_device is not None and "cpu" in out_device:
                tensors_cpu = True
            node_comments = TorchOpCollector._resolve_node_comments(comments) or ""

            if (op_name is None) or (
                out_shape is not None
                and isinstance(out_shape, str)
                and "symbolic" in out_shape
            ):
                if op_name is not None:
                    TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                        f"Skipping op {op_name} as its shape contains unbound symbolic elements"
                    )
                continue
            TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                f"MAIN OP DETAILS: {op_name}, {out_shape}, {out_stride}, {out_storage_offset}, {out_dtype}, {out_device}"
            )
            output_list.append(f"{node.op} {op_name}")
            ops_set.add(op_name)

            san_args = []
            san_args_norm = []
            san_args_comp = []
            yaml_inputs = []
            yaml_inputs_norm = []
            arg_comments = ""

            TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                f"node.args: {node.args}"
            )
            saved_shape = []
            for i, arg in enumerate(node.args):
                TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                    f"Arg #{i}: {node.args[i]}, {arg} arg type: {type(arg)}"
                )
                if isinstance(arg, torch.fx.Node):
                    result = TorchOpCollector._process_node_arg(
                        arg, op_name, i, saved_shape, san_args, out_device
                    )
                    if result.san_arg is not None and not result.skip_this_node:
                        output_list.append(
                            f"  arg[{i}] (Node: {arg.name}) -> shape: {result.san_arg} {arg.name}"
                        )
                elif not isinstance(arg, torch.fx.immutable_collections.immutable_list):
                    assert not isinstance(arg, torch.Tensor)
                    TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                        f"Scalar arg: {arg}, type: {type(arg)}"
                    )
                    output_list.append(f"  arg[{i}] (value) -> {arg}")
                    result = TorchOpCollector._process_scalar_arg(arg, op_name)
                else:
                    result = TorchOpCollector._process_list_arg(arg, op_name)

                if result.skip_this_node:
                    skip_this_node = True
                    break
                if result.tensors_cpu:
                    tensors_cpu = True
                if result.const_arg:
                    const_arg = True
                if result.round_up:
                    round_up = True
                if result.dim_redn:
                    dim_redn = True
                if result.to_float16_fromfp32:
                    to_float16_fromfp32 = True
                if result.to_float16_frombf16:
                    to_float16_frombf16 = True
                if result.arg_comments:
                    arg_comments += result.arg_comments
                if result.new_saved_shape is not None:
                    saved_shape = result.new_saved_shape
                if result.orig_shape is not None:
                    orig_shape = result.orig_shape
                if result.shrunk_shape is not None:
                    shrunk_shape = result.shrunk_shape
                if result.san_arg is not None:
                    san_args.append(result.san_arg)
                if result.san_arg_norm is not None:
                    san_args_norm.append(result.san_arg_norm)
                if result.san_arg_comp is not None:
                    san_args_comp.append(result.san_arg_comp)
                if result.yaml_input is not None:
                    yaml_inputs.append(result.yaml_input)
                if result.yaml_input_norm is not None:
                    yaml_inputs_norm.append(result.yaml_input_norm)

            if skip_this_node:
                continue

            san_args, san_args_norm, yaml_inputs, yaml_inputs_norm = (
                TorchOpCollector._pack_reshape_args(
                    op_name,
                    san_args,
                    san_args_norm,
                    yaml_inputs,
                    yaml_inputs_norm,
                    dim_redn,
                    round_up,
                    orig_shape,
                    shrunk_shape,
                )
            )

            kwmap = {}
            if node.kwargs:
                san_args_comp.append(node.kwargs)
                output_list.append("  kwargs:")
                for k, v in node.kwargs.items():
                    output_list.append(f"    {k}: {v}")
                    if "attn_mask" not in k:
                        kwmap[k] = v
                        if isinstance(v, torch.dtype) or k == "device" or v is None:
                            kwmap[k] = str(v)

            output_list.append(
                f"  output shape: {out_shape}, {out_stride}, {out_storage_offset}, {out_dtype}, {out_device} @ {node.name}"
            )
            output_list.append("")
            TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                f"san_args: {san_args} san_args_comp: {san_args_comp}"
            )

            TorchOpCollector._emit_test_case(
                op_name,
                yaml_inputs,
                yaml_inputs_norm,
                kwmap,
                node.kwargs,
                node_comments,
                san_args_comp,
                round_up,
                to_float16_fromfp32,
                to_float16_frombf16,
                dim_redn,
                const_arg,
                tensors_cpu,
            )

        output: str = "\n".join(output_list)
        if print_output:
            print(output)
        TorchOpCollector.graph_id = TorchOpCollector.graph_id + 1
        return output

    orig_compile_fx = None
    ops_set: set[Any] = set()
    print_output = False
    print_graph_module = False
    graph_module_idx = 0

    @staticmethod
    def _extract_stacktrace_comments(node):
        stacktrace = (
            torch.fx.graph._parse_stack_trace(node.stack_trace)
            if node.stack_trace is not None
            else None
        )
        comments = ""
        if stacktrace is not None:
            TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                f"STACK TRACE: {stacktrace.file} SUMMARY STR: {stacktrace.get_summary_str()} CODE: {stacktrace.code}"
            )
            comments = stacktrace.get_summary_str()
            if comments is not None:
                comm_idx = comments.find("File: ")
                if comm_idx > 0:
                    comments = comments[comm_idx:-2]
        return comments, stacktrace

    @staticmethod
    def _extract_meta_info(meta_val):
        if (
            isinstance(meta_val, torch.Tensor)
            and meta_val.layout
            not in (
                torch.sparse_csc,
                torch.sparse_csr,
            )
            or isinstance(meta_val, torch.fx.passes.shape_prop.TensorMetadata)
        ):
            shape = meta_val.shape
            stride = (
                meta_val.stride()
                if isinstance(meta_val, torch.Tensor)
                else meta_val.stride
            )
            storage_offset = (
                meta_val.storage_offset() if isinstance(meta_val, torch.Tensor) else 0
            )
            dtype = meta_val.dtype
            device = (
                str(meta_val.device) if isinstance(meta_val, torch.Tensor) else "???"
            )
            TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                f"Case 1: {shape}, {stride}, {storage_offset}, {dtype}, {device}"
            )
            try:
                shape: list[int] = [int(s) for s in shape]
                stride: list[int] = [int(s) for s in stride]
                storage_offset = int(storage_offset)
            except (TypeError, ValueError):
                for ss in shape:
                    if not is_concrete_int(ss) and has_free_symbols(ss):
                        TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                            f"{ss} has free symbols"
                        )
                        shape = stride = "symbolic"
                        storage_offset = -1
                        device = "???"
                        break
        elif isinstance(meta_val, torch.fx.experimental.proxy_tensor.py_sym_types):
            # if not is_concrete_int(meta_val) and has_free_symbols(meta_val):
            TorchOpCollector.log_function[TorchOpCollector.log_mthd]("Case 3")
            if has_free_symbols(meta_val):
                TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                    f"{meta_val} has free symbols"
                )
                shape = stride = "symbolic"
                storage_offset = -1
                dtype = device = "???"
            else:
                # If the shape symbolic variable is bound, use the bound value instead of symbol
                if is_concrete_int(meta_val):
                    TorchOpCollector.log_function[TorchOpCollector.log_mthd]("Case 3A")
                    shape = int(meta_val)
                    stride = [1]
                    storage_offset = 0
                    dtype = "scalar int"
                    device = "???"
                else:
                    TorchOpCollector.log_function[TorchOpCollector.log_mthd]("Case 3B")
                    shape = stride = dtype = device = None
                    storage_offset = -1
            TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                f"shape: {shape}, dtype: {dtype}"
            )
        elif meta_val is None:
            TorchOpCollector.log_function[TorchOpCollector.log_mthd]("Case 4")
            shape = stride = storage_offset = dtype = device = None
        # if meta_val is an instance of a tuple, iterate thro' the tuple elements
        # and store them in an output tuple after resolving symbols. Unbound symbols
        # will result in skipping the main operator
        elif isinstance(meta_val, tuple):
            # workaround for torch.chunk() in Granite 4
            shape = stride = storage_offset = dtype = device = None
            TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                f"Case 5: shape - {shape} dtype - {dtype}"
            )
            TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                "Types of elements in tuple"
            )
            shapes_tuple = ()
            strides_tuple = ()
            storage_offsets_tuple = ()
            dtypes_tuple = ()
            devices_tuple = ()
            for vv in meta_val:
                TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                    f"{vv}: {type(vv)} type: {vv.dtype} shape: {vv.shape}"
                )
                if not TorchOpCollector.is_symbolic_list(list(vv.shape)):
                    shapes_tuple = shapes_tuple + (vv.shape,)
                    stride = vv.stride() if isinstance(vv, torch.Tensor) else vv.stride
                    strides_tuple = strides_tuple + (stride,)
                    storage_offset = (
                        vv.storage_offset() if isinstance(vv, torch.Tensor) else 0
                    )
                    storage_offsets_tuple = storage_offsets_tuple + (storage_offset,)
                    dtypes_tuple = dtypes_tuple + (vv.dtype,)
                    devices_tuple = devices_tuple + (vv.device,)
                else:
                    break
            # Update shape and dtype only if shapes tuple and dtypes tuple are non-empty
            # Else leave shape and dtype unaltered as None
            if shapes_tuple and dtypes_tuple:
                shape = shapes_tuple
                stride = strides_tuple
                storage_offset = storage_offsets_tuple
                dtype = dtypes_tuple
                device = devices_tuple
        elif isinstance(meta_val, (int, float)):
            # workaround for Gemma3
            TorchOpCollector.log_function[TorchOpCollector.log_mthd]("Case 6")
            shape = stride = storage_offset = dtype = device = None
        else:
            TorchOpCollector.log_function[TorchOpCollector.log_mthd]("Case 7: unknown")
            raise RuntimeError(f"{type(meta_val)} : {meta_val}")
        return shape, stride, storage_offset, dtype, device

    @staticmethod
    def _extract_op_name(node):
        op = ""
        if node.op == "call_function":
            assert callable(node.target)
            if getattr(node.target, "__module__", "") == "_operator":
                TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                    f"Function type is _operator: {node.target.__name__}"
                )
                target_name = node.target.__name__
                if target_name in torch.fx.graph.magic_methods:
                    TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                        f"Operator {target_name} is magic method"
                    )
                    return "torch." + target_name
                if target_name in torch.fx.graph.inplace_methods:
                    TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                        f"Operator {target_name} is an inplace method"
                    )
                    if target_name[0] == "i":
                        # op = "torch." + target_name[1:] + "_"
                        op = "torch.Tensor." + target_name[1:]
                    elif target_name == "setitem":
                        op = "torch." + target_name
                    return op
            op = torch.fx.node._get_qualified_name(node.target)
            TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                f"target name: {node.target.__name__} {node.target} Qualified Name: {op}"
            )
            if op.find("torch._C._nn") == 0:
                op = "torch.nn.functional" + op[12:]
        elif node.op == "call_method":
            TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                f"Operator Function type is _operator of type call_method, has a qualified name of {node.op} and will be invoked as {node.target}"
            )
            assert isinstance(node.target, str)
            if node.target in ["expand", "copy_", "contiguous", "view"]:
                op = "torch.Tensor." + node.target
            # elif (node.target in ['float', 'long']):
            # op = node.target
            else:
                op = "torch." + node.target
        return op

    @staticmethod
    def get_node_shape(node):
        assert isinstance(node, torch.fx.Node)
        # derived from emit_node() at torch/fx/graph/graph.py
        meta_val = node.meta.get(
            "val",
            node.meta.get("tensor_meta", node.meta.get("example_value", None)),
        )
        TorchOpCollector.log_function[TorchOpCollector.log_mthd](
            f"Node args: {node.args}"
        )
        TorchOpCollector.log_function[TorchOpCollector.log_mthd](
            f"Node kw args: {node.kwargs}"
        )

        comments, stacktrace = TorchOpCollector._extract_stacktrace_comments(node)

        # Case 2: dynamo node — skip (workaround for torch.autograd.function.FunctionCtx
        # in autograd and torch._C._log_api_usage_once in dynamo)
        if stacktrace is not None and "torch/_dynamo" in stacktrace.file:
            TorchOpCollector.log_function[TorchOpCollector.log_mthd]("Case 2")
            return None, None, None, None, None, None, None

        shape, stride, storage_offset, dtype, device = (
            TorchOpCollector._extract_meta_info(meta_val)
        )
        op = TorchOpCollector._extract_op_name(node)
        return op, shape, stride, storage_offset, dtype, device, comments

    @staticmethod
    def is_symbolic_list(ll):
        if isinstance(ll, list):
            for elem in ll:
                # if not is_concrete_int(elem) and has_free_symbols(elem):
                if has_free_symbols(elem):
                    TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                        f"is_symbolic_list: {elem} is symbolic"
                    )
                    return True
        else:
            if (
                isinstance(ll, torch.fx.node.Node)
                and ll.op == "placeholder"
                and isinstance(ll.target, str)
            ):
                return True
            TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                f"is_symbolic_list: type: {type(ll)}"
            )
        return False

    @staticmethod
    def normalize_shape_type(shape, dtype):
        if len(shape) == 0:
            return shape, dtype
        shrunk_shape = [s for s in shape]
        mod_dtype = dtype
        ###if (len(shape) > 3):
        ###  shrunk_shape = shape[len(shape)-3:]
        # mult_of_64 = any(s/64 == math.floor(s/64) for s in shrunk_shape)
        shrunk_shape[-1] / 64 == math.floor(shrunk_shape[-1] / 64)
        ###Do not shrink a shape
        ###if not mult_of_64:
        ###  shrunk_shape[-1] = int(math.ceil(shrunk_shape[-1]/64))*64
        if "float32" in str(mod_dtype):
            mod_dtype = torch.float16
        if "bfloat16" in str(mod_dtype):
            mod_dtype = torch.float16
        return shrunk_shape, mod_dtype

    @staticmethod
    def get_normalization_flags(shape, dtype, shrunk_shape, mod_dtype):
        round_up, dim_redn, to_float16_fromfp32, to_float16_frombf16 = (
            False,
            False,
            False,
            False,
        )
        # if shrunk_shape != shape[len(shape)-3:]:
        #    round_up = True
        # if len(shrunk_shape) < len(shape):
        #    dim_redn = True
        if str(mod_dtype) != str(dtype) and "float16" in str(mod_dtype):
            if "float32" in str(dtype):
                to_float16_fromfp32 = True
            if "bfloat16" in str(dtype):
                to_float16_frombf16 = True
        return round_up, dim_redn, to_float16_fromfp32, to_float16_frombf16

    @staticmethod
    def preprocess_node_contents(op, shape, dtype):
        skip, is_scalar, is_list, skip_arg_and_cont = False, False, False, False
        if (op is None) or (
            shape is not None and isinstance(shape, str) and "symbolic" in shape
        ):
            if op is not None:
                TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                    f"Skipping {op} as one of its args has symbolic shape with free symbols"
                )
                skip = True
        elif shape is None and dtype is None:
            TorchOpCollector.log_function[TorchOpCollector.log_mthd](
                f"Skipping argument {op} as one of it is a node but with no shape or type"
            )
            skip_arg_and_cont = True
        elif dtype is not None and "scalar" in str(dtype):
            is_scalar = True
        elif isinstance(shape, list):
            is_list = True
        return skip, skip_arg_and_cont, is_scalar, is_list

    @staticmethod
    def _compile_fx(
        model_,
        example_inputs_,
        inner_compile=torch._inductor.compile_fx.compile_fx_inner,
        config_patches=None,
        decompositions=None,
        *args,
        **kwargs,
    ):
        model_.print_readable(print_output=TorchOpCollector.print_graph_module)
        TorchOpCollector.collect_torchops(
            model_, TorchOpCollector.ops_set, TorchOpCollector.print_output
        )
        return TorchOpCollector.orig_compile_fx(
            model_, example_inputs_, inner_compile, config_patches, decompositions
        )

    def __init__(self, print_output=False, print_graph_module=False):
        TorchOpCollector.print_output = print_output
        TorchOpCollector.print_graph_module = print_graph_module

    def __enter__(self):
        TorchOpCollector.graph_module_idx = 0
        TorchOpCollector.ops_set = set()

        TorchOpCollector.orig_compile_fx = torch._inductor.compile_fx.compile_fx
        torch._inductor.compile_fx.compile_fx = TorchOpCollector._compile_fx

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch._inductor.compile_fx.compile_fx = TorchOpCollector.orig_compile_fx
        TorchOpCollector.orig_compile_fx = None

        self.ops_list = list(TorchOpCollector.ops_set)
        self.ops_list.sort()
        TorchOpCollector.ops_set = set()

        self.test_gen_ops = list(TorchOpCollector.test_gen_ops_set)
        self.test_gen_ops.sort()
        TorchOpCollector.test_gen_ops_set = set()
        return False

    def write_yaml(
        self, model_name, output_dir=".", yaml_defaults=None, supress_spyre=False
    ):
        defaults = {**TorchOpCollector.DEFAULT_YAML_DEFAULTS, **(yaml_defaults or {})}

        def _filter_cases(cases: list[dict[str, Any]]):
            result = []
            for tc in cases:
                try:
                    yaml.dump(tc, sort_keys=False)
                    result.append(tc)
                except (ValueError, RecursionError) as e:
                    print(f"Skipping test case due to yaml error: {e}\n  {tc}")
            return result

        config = {}
        if not USE_OLDFORMAT:
            dtypes = [
                "float16",
                "float32",
                "float64",
                "bfloat16",
                "int8",
                "int16",
                "int32",
                "int64",
                "uint8",
                "uint16",
                "uint32",
                "uint64",
                "complex32",
                "complex64",
                "complex128",
                "bool",
                "half",
            ]
            seed = 123
            config = {
                "test_suite_config": {
                    "global": {
                        "supported_dtypes": [
                            {"name": dt, "precision": {"atol": 0.005, "rtol": 0.005}}
                            for dt in dtypes
                        ],
                        "input_config": {
                            "seed": seed,
                        },
                    },
                    "files": [
                        {
                            "path": "${TORCH_DEVICE_ROOT}/tests/models/test_model_ops_v2.py",
                            "unlisted_test_mode": "skip",
                            "tests": [
                                {
                                    "names": ["TestSpyreModelOps::test_model_ops_db"],
                                    "mode": "mandatory_success",
                                    "tags": ["model__" + model_name],
                                    "edits": {
                                        "ops": {
                                            "include": _filter_cases(
                                                self.test_cases_yaml
                                            ),
                                        }
                                    },
                                }
                            ],
                        }
                    ],
                }
            }
            test_cases_count = len(
                config["test_suite_config"]["files"][0]["tests"][0]["edits"]["ops"][
                    "include"
                ]
            )
            print(f"Total no. of test cases: {test_cases_count}")
        else:
            config = {
                "model": model_name,
                "defaults": defaults,
                "cases": _filter_cases(self.test_cases_yaml),
            }
            print(f"Total no. of test cases: {len(config['cases'])}")

        with open(os.path.join(output_dir, model_name + ".yaml"), "w") as f:
            yaml.dump(config, f, sort_keys=False)

        if not supress_spyre:
            if not USE_OLDFORMAT:
                config["test_suite_config"]["files"][0]["tests"][0]["edits"]["ops"][
                    "include"
                ] = _filter_cases(self.test_cases_norm_yaml)
                config["test_suite_config"]["files"][0]["tests"][0]["tags"][0] = (
                    "model__" + model_name + "_spyre"
                )
            else:
                config["cases"] = _filter_cases(self.test_cases_norm_yaml)
            with open(os.path.join(output_dir, model_name + "_spyre.yaml"), "w") as f:
                yaml.dump(config, f, sort_keys=False)


# for debug purpose
def main():
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForMaskedLM,
        AutoTokenizer,
        StaticCache,
    )

    prompt = "Where is the Thomas J. Watson Research Center located?"

    model_path = "openai/gpt-oss-20b"

    is_encoder = "bert" in model_path

    device = "cuda"
    if is_encoder:
        model = AutoModelForMaskedLM.from_pretrained(model_path, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    encoded_input = tokenizer(prompt, return_tensors="pt").to(device)

    past_key_values = StaticCache(config=model.config, max_cache_len=2048)

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    torch._inductor.config.trace.enabled = True
    torch._inductor.config.trace.debug_dir = None

    if is_encoder:
        model = torch.compile(model)
    else:
        model.forward = torch.compile(model.forward)

    with TorchOpCollector(print_output=True, print_graph_module=True) as ctx:
        # with TorchOpCollector() as ctx:
        with torch.no_grad():
            if is_encoder:
                model(**encoded_input)
            else:
                if "granite-4.0-h-" in model_path:
                    model.generate(**encoded_input, use_cache=True)
                else:
                    model.generate(
                        **encoded_input, past_key_values=past_key_values, use_cache=True
                    )

    for op in ctx.ops_list:
        print(op)


if __name__ == "__main__":
    main()
