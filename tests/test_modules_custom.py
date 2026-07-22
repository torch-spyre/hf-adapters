"""
Custom module tests for torch-spyre.

This file contains additional test methods for modules defined in YAML configs:
- test_eager_vs_compile: Cross-check that Spyre eager and Spyre compiled agree with
  each other and with CPU (all three run in a single pass)
- test_with_cpu: Use CPU as the golden reference and compare device output against it;
  which device mode(s) to run (compile and/or eager) is selectable via env vars
- test_layout_stride: Validate real YAML-specified SpyreTensorLayouts and strides (CPU vs Spyre)
- test_vllm: Standalone-run a vLLM-native module (built from AutoConfig under a VllmConfig +
  TP=1 distributed group, NOT via LLM()) and compare CPU eager vs device compile

All tests use pytree for robust handling of nested input/output structures and test only
real model configurations from YAML without artificial modifications.
"""

import os

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_modules import module_db, modules
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.utils._pytree import tree_map


def _extract_all_tensors(output):
    """Extract all tensors from potentially nested output structure.

    Uses pytree to handle nested structures (tuples, lists, dicts, etc.)
    and returns all tensors found. This ensures complete validation of
    module outputs including hidden states, KV caches, attention weights, etc.

    Args:
        output: Module output (can be tensor, tuple, list, dict, or nested structure)

    Returns:
        List of all tensors found in the output structure, in traversal order.
        Returns empty list if no tensors exist.
    """
    tensors = []

    def collect_tensors(x):
        if isinstance(x, torch.Tensor):
            tensors.append(x)
        return x

    tree_map(collect_tensors, output)
    return tensors


def _construct_module(
    module_info, module_input, *, dtype=None, device=None, training=False
):
    """Construct a module from a module_input's constructor args/kwargs.

    Optionally casts to ``dtype`` and/or moves to ``device``, then sets train/eval
    mode. All of ``dtype``/``device``/``training`` are optional so every test method
    can reuse this regardless of whether it casts dtype or only calls ``.eval()``.
    """
    module = module_info.module_cls(
        *module_input.constructor_input.args,
        **module_input.constructor_input.kwargs,
    )
    if dtype is not None:
        module = module.to(dtype)
    if device is not None:
        module = module.to(device)
    module.train(training)
    return module


def _move_inputs(module_input, *, dtype=None, device=None):
    """Move a module_input's forward args/kwargs to ``dtype`` and/or ``device``.

    Both ``dtype`` and ``device`` are optional so callers that only relocate to a
    device (no dtype cast) can reuse this too. Uses pytree so nested input
    structures (tuples, lists, dicts) are handled. Returns an ``(args, kwargs)`` tuple.
    """

    def is_interesting_dtype(dtype):
        if dtype is None:
            return False
        return str(dtype) in ("torch.float16", "torch.float32", "torch.bfloat16")

    def move(x):
        if not isinstance(x, torch.Tensor):
            return x
        if device is not None and is_interesting_dtype(x.dtype):
            return x.to(device, dtype)
        if device is not None:
            return x.to(device)
        if is_interesting_dtype(x.dtype):
            return x.to(dtype)
        return x

    args = tree_map(move, module_input.forward_input.args)
    kwargs = tree_map(move, module_input.forward_input.kwargs)
    return args, kwargs


def _run_forward(module, args, kwargs):
    """Run a no-grad forward pass and return the flattened list of output tensors."""
    with torch.no_grad():
        output = module(*args, **kwargs)
    return _extract_all_tensors(output)


# ---------------------------------------------------------------------------
# vLLM standalone-module helpers (used by test_vllm)
# ---------------------------------------------------------------------------
#
# test_vllm rebuilds a vLLM-native module WITHOUT constructing an LLM(). It uses
# the same recipe as the standalone reproduction script: set a current VllmConfig,
# init a TP=1 distributed group (required even at TP=1 because vLLM's parallel
# layers query the group), construct the module directly, then tear the group
# down. The module's constructor args come from AutoConfig via per-class adapters.


def _setup_distributed(backend: str = "gloo") -> None:
    """Init a single-rank distributed environment for vLLM parallel layers."""
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method="env://",
        backend=backend,
    )
    initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        backend=backend,
    )


def _teardown_distributed() -> None:
    """Tear down the distributed group so a later init can run in-process.

    Tolerant of being called when nothing was initialized (e.g. a build failed
    before setup completed) so it is safe to call unconditionally from a finally.
    """
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )

    try:
        destroy_model_parallel()
    except Exception:
        pass
    try:
        destroy_distributed_environment()
    except Exception:
        pass


def _resolve_hf_config(constructor_input):
    """Return the HF config for a vLLM module from its YAML constructor arg, or None.

    The vLLM YAML records the model config as a config-type constructor arg.
    Resolution order (any one is enough):

    1. Already a live ``PretrainedConfig`` (the OOT framework resolved it, e.g.
       via ``model_id`` on feat/oot-config-from-pretrained).
    2. A raw dict with ``model_id`` -> ``AutoConfig.from_pretrained(model_id)``
       (+ ``config_overrides``). The faithful, full config.
    3. A raw dict with ``config_path`` (+ optional ``config_kwargs``) -> import
       the config class and build it. This is what the current generator emits
       when no model_id is embedded; ``config_kwargs`` may be empty (library
       defaults), which is acceptable for shape-only standalone runs.

    Returns ``None`` when no config can be resolved -- the caller decides whether
    to skip (e.g. the module needs a config but none was provided).
    """
    import importlib

    from transformers import AutoConfig
    from transformers.configuration_utils import PretrainedConfig

    if not constructor_input.args:
        return None
    arg0 = constructor_input.args[0]

    if isinstance(arg0, PretrainedConfig):
        return arg0

    if not isinstance(arg0, dict):
        return None

    if arg0.get("model_id"):
        config = AutoConfig.from_pretrained(arg0["model_id"])
        for key, value in arg0.get("config_overrides", {}).items():
            setattr(config, key, value)
        return config

    if arg0.get("config_path"):
        module_path, _, cls_name = arg0["config_path"].rpartition(".")
        config_cls = getattr(importlib.import_module(module_path), cls_name)
        return config_cls(**(arg0.get("config_kwargs") or {}))

    return None


# Attributes on the HF config that map to common vLLM constructor kwarg names.
# Used only for COMPOSITE modules (vllm.model_executor.models.*), whose scalar
# constructor args are derived from the model config (Option C). Atomic library
# layers carry their constructor kwargs explicitly in the YAML (captured from the
# live instance at generation time), so they need no config derivation.
_CONFIG_ATTR_FOR_PARAM = {
    "hidden_size": ("hidden_size",),
    "intermediate_size": ("intermediate_size",),
    "hidden_act": ("hidden_act",),
    "eps": ("rms_norm_eps", "layer_norm_eps"),
    "bias": ("mlp_bias", "attention_bias"),
}


class _SkipModuleError(Exception):
    """Raised when a composite module's constructor cannot be resolved from config."""


def build_ctor_kwargs_from_config(module_cls, hf_config) -> dict:
    """Derive a COMPOSITE vLLM module's constructor kwargs from the HF config.

    Generic (not per-class): inspect ``module_cls.__init__`` and fill each
    parameter from the HF config using ``_CONFIG_ATTR_FOR_PARAM`` (same-named
    attribute by default). ``quant_config`` -> None, ``prefix`` -> a label, and a
    ``config`` param receives the config object itself (GraniteAttention /
    GraniteDecoderLayer take a config; GraniteMLP takes scalars derived here).
    A required param that cannot be resolved raises ``_SkipModuleError``.
    """
    import inspect

    sig = inspect.signature(module_cls.__init__)
    kwargs: dict = {}
    unresolved: list[str] = []

    for pname, param in sig.parameters.items():
        if pname == "self":
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue

        if pname == "quant_config":
            kwargs[pname] = None
            continue
        if pname == "prefix":
            kwargs[pname] = module_cls.__name__.lower()
            continue
        if pname == "config":
            kwargs[pname] = hf_config
            continue

        resolved = False
        for attr in _CONFIG_ATTR_FOR_PARAM.get(pname, (pname,)):
            if hasattr(hf_config, attr):
                kwargs[pname] = getattr(hf_config, attr)
                resolved = True
                break

        if not resolved and param.default is inspect.Parameter.empty:
            unresolved.append(pname)

    if unresolved:
        raise _SkipModuleError(
            f"{module_cls.__name__}: cannot resolve required ctor params "
            f"{unresolved} from HF config"
        )
    return kwargs


def randomize_weights_xavier(module: torch.nn.Module, seed: int = 0) -> None:
    """Deterministically initialize a module's parameters with xavier-uniform.

    >=2-D params get xavier_uniform_. For 1-D params (biases, norm weights),
    xavier is undefined, so they get a deterministic uniform ``rand`` init
    instead. Iterating sorted(named_parameters()) with a fixed generator makes
    the result reproducible, so two independently-built copies (CPU ref and
    device) get identical weights.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for _, p in sorted(module.named_parameters()):
            tmp = torch.empty(p.shape, dtype=torch.float32)
            if p.dim() >= 2:
                torch.nn.init.xavier_uniform_(tmp, generator=gen)
            else:
                # xavier is undefined for 1-D params -> deterministic rand.
                tmp.uniform_(0.0, 1.0, generator=gen)
            p.copy_(tmp.to(p.dtype))


# Some vLLM modules take a non-activation object (e.g. a weight-carrying layer)
# as a forward argument that the generator cannot capture as a tensor. These
# builders synthesize that extra argument inside the vLLM context (so layers that
# read the TP group can be constructed) and return the full positional forward
# args. Registered by module class name.


def _fwd_args_logits_processor(
    module, constructor_input, fwd_args, *, device, dtype, seed
):
    """Prepend a constructed ``lm_head`` (ParallelLMHead) to LogitsProcessor's args.

    ``LogitsProcessor.forward(lm_head, hidden_states, embedding_bias=None)`` needs
    a real ParallelLMHead (with weights + quant_method), not a plain tensor. Build
    it from the captured ``vocab_size`` and the hidden_states' feature dim, init it
    with the same deterministic xavier weights as everything else, and place it on
    the same device/dtype as the captured hidden_states.
    """
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    if not fwd_args or not isinstance(fwd_args[0], torch.Tensor):
        raise _SkipModuleError("LogitsProcessor: no captured hidden_states tensor")
    hidden_states = fwd_args[0]
    embedding_dim = int(hidden_states.shape[-1])
    ctor_kwargs = dict(constructor_input.kwargs)
    vocab_size = ctor_kwargs.get("vocab_size")
    if vocab_size is None:
        raise _SkipModuleError("LogitsProcessor: no vocab_size in constructor_inputs")

    lm_head = ParallelLMHead(
        num_embeddings=int(vocab_size),
        embedding_dim=embedding_dim,
        org_num_embeddings=ctor_kwargs.get("org_vocab_size"),
    )
    randomize_weights_xavier(lm_head, seed=seed)
    if dtype is not None:
        lm_head = lm_head.to(dtype)
    lm_head.eval()
    if device is not None:
        lm_head = lm_head.to(device)
    # forward(lm_head, hidden_states, ...) -- lm_head is the first positional arg.
    return [lm_head, *fwd_args]


_FWD_ARGS_BUILDERS = {
    "LogitsProcessor": _fwd_args_logits_processor,
}


def _run_vllm_module(
    module_cls,
    constructor_input,
    fwd_args,
    fwd_kwargs,
    *,
    device,
    compile,
    dtype=None,
    seed=0,
):
    """Build a vLLM module, init its weights, and run one forward -- all inside
    the vLLM context -- and return the flattened output tensors.

    Everything that touches vLLM state (construction, ``torch.compile``, and the
    forward pass itself) runs inside a single ``set_current_vllm_config(...)`` +
    TP=1 distributed group, then the group is torn down. This mirrors the
    init -> build -> run -> destroy structure of the standalone reproduction
    script: the current vLLM config is a thread-local that is restored when the
    ``with`` block exits, and vLLM's compile path reads it, so the forward must
    not run outside the block.

    Constructor args come from the YAML-resolved ``constructor_input``:

    - **Atomic** library layers carry explicit captured kwargs (ints/lists/etc.),
      so the module is built generically as ``module_cls(*args, **kwargs)`` -- no
      HF config, no per-class adapter.
    - **Composite** model modules carry a single config arg (resolved to a live
      ``PretrainedConfig``); their scalar ctor args are derived from it via the
      generic ``build_ctor_kwargs_from_config`` (Option C).

    ``fwd_args``/``fwd_kwargs`` must already be on the target device. Raises
    ``_SkipModuleError`` if a composite's ctor cannot be resolved from config.
    """
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

    args = list(constructor_input.args)
    kwargs = dict(constructor_input.kwargs)
    hf_config = _resolve_hf_config(constructor_input)

    vllm_config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    try:
        with set_current_vllm_config(vllm_config):
            _setup_distributed()
            if hf_config is not None:
                # Composite (Option C): derive scalar ctor args from the config.
                module = module_cls(
                    **build_ctor_kwargs_from_config(module_cls, hf_config)
                )
            else:
                # Atomic: explicit captured args/kwargs from the YAML.
                module = module_cls(*args, **kwargs)

            # Same seed on both the CPU-ref and device runs => identical weights.
            randomize_weights_xavier(module, seed=seed)
            if dtype is not None:
                module = module.to(dtype)
            module.eval()
            if device is not None:
                module = module.to(device)

            # Synthesize any non-tensor forward argument the generator could not
            # capture (e.g. LogitsProcessor's lm_head layer). Built here so it is
            # inside the vLLM context/TP group, with the same device/dtype/seed.
            run_args = fwd_args
            builder = _FWD_ARGS_BUILDERS.get(module_cls.__name__)
            if builder is not None:
                run_args = builder(
                    module,
                    constructor_input,
                    fwd_args,
                    device=device,
                    dtype=dtype,
                    seed=seed,
                )

            runnable = torch.compile(module) if compile else module
            # Run forward INSIDE the context: vLLM's compile/dispatch path reads
            # the current vLLM config, so it must be active during the forward.
            out_tensors = _run_forward(runnable, run_args, fwd_kwargs)
    finally:
        _teardown_distributed()
    return out_tensors


class TestModuleCustom(TestCase):
    """Custom test cases for module validation with different execution modes and layouts."""

    def setUp(self):
        super().setUp()
        torch.manual_seed(0xAFFE)

    @modules(module_db)
    def test_eager_vs_compile(self, device, dtype, module_info, training):
        """Test eager mode vs compile mode, comparing CPU and Spyre outputs.

        This test:
        1. Runs module in eager mode on CPU
        2. Runs module in eager mode on Spyre
        3. Runs module in compile mode on Spyre
        4. Compares outputs between eager CPU, eager Spyre, and compile Spyre
        """
        module_inputs = module_info.module_inputs_func(
            module_info, device=device, dtype=dtype, requires_grad=False, training=False
        )

        for module_input in module_inputs:
            # Create module on CPU (eager)
            module_cpu = module_info.module_cls(
                *module_input.constructor_input.args,
                **module_input.constructor_input.kwargs,
            )
            module_cpu.to(dtype)
            module_cpu.eval()

            # Create module on device (eager)
            module_device_eager = module_info.module_cls(
                *module_input.constructor_input.args,
                **module_input.constructor_input.kwargs,
            )
            module_device_eager.to(device).to(dtype)
            module_device_eager.eval()

            # Copy weights from CPU to device
            module_device_eager.load_state_dict(module_cpu.state_dict())

            # Create compiled version
            module_device_compile_base = module_info.module_cls(
                *module_input.constructor_input.args,
                **module_input.constructor_input.kwargs,
            )
            module_device_compile_base.to(device).to(dtype)
            module_device_compile_base.eval()
            module_device_compile_base.load_state_dict(module_cpu.state_dict())
            module_device_compile = torch.compile(module_device_compile_base)

            # module_input.forward_input tensors are already placed on `device`
            # by the OOT framework's module_inputs_func (it builds on CPU, then
            # relocates to the test device so upstream's single-module
            # test_forward can use them directly -- see _move_to_test_device in
            # oot_test_config_models.py). They are NOT CPU tensors despite the
            # attribute name, so build a genuine CPU copy for the CPU reference
            # module instead of using them as-is.
            args_device = module_input.forward_input.args
            kwargs_device = module_input.forward_input.kwargs

            args_cpu = tree_map(
                lambda x: x.cpu() if isinstance(x, torch.Tensor) else x, args_device
            )
            kwargs_cpu = tree_map(
                lambda x: x.cpu() if isinstance(x, torch.Tensor) else x, kwargs_device
            )

            # Run forward passes
            with torch.no_grad():
                output_cpu = module_cpu(*args_cpu, **kwargs_cpu)
                output_device_eager = module_device_eager(*args_device, **kwargs_device)
                output_device_compile = module_device_compile(
                    *args_device, **kwargs_device
                )

            # Extract all tensors from outputs using pytree
            cpu_tensors = _extract_all_tensors(output_cpu)
            device_eager_tensors = _extract_all_tensors(output_device_eager)
            device_compile_tensors = _extract_all_tensors(output_device_compile)

            # Verify all outputs have the same number of tensors
            if not (
                len(cpu_tensors)
                == len(device_eager_tensors)
                == len(device_compile_tensors)
            ):
                self.fail(
                    f"{module_info.name}: Output tensor count mismatch - "
                    f"CPU: {len(cpu_tensors)}, Spyre eager: {len(device_eager_tensors)}, "
                    f"Spyre compile: {len(device_compile_tensors)}"
                )

            # Compare all tensors (hidden states, KV cache, attention weights, etc.)
            for i, (cpu_t, eager_t, compile_t) in enumerate(
                zip(cpu_tensors, device_eager_tensors, device_compile_tensors)
            ):
                # Compare CPU eager vs Spyre eager
                #
                # atol/rtol are passed explicitly (sourced from self.precision /
                # self.rel_tol, populated by the OOT framework's YAML-driven
                # tolerance overrides). Without explicit values, assertEqual
                # falls back to torch's per-dtype default_tolerances() and only
                # ever widens it via max(default, override) -- so a YAML rtol
                # tighter than the dtype default (e.g. bfloat16's built-in 0.016)
                # would otherwise be silently ignored.
                self.assertEqual(
                    cpu_t,
                    eager_t.cpu(),
                    atol=self.precision,
                    rtol=self.rel_tol,
                    msg=f"{module_info.name}: CPU eager vs Spyre eager mismatch (tensor {i})",
                )

                # Compare Spyre eager vs Spyre compile. torch.isclose (used
                # internally by assertEqual) needs aten::isnan, which has no
                # Spyre kernel, so the comparison must happen on CPU tensors.
                self.assertEqual(
                    eager_t.cpu(),
                    compile_t.cpu(),
                    atol=self.precision,
                    rtol=self.rel_tol,
                    msg=f"{module_info.name}: Spyre eager vs Spyre compile mismatch (tensor {i})",
                )

    @modules(module_db)
    def test_with_cpu(self, device, dtype, module_info, training):
        """Use CPU output as the golden reference and compare device output against it.

        Unlike test_eager_vs_compile (which cross-checks Spyre eager vs Spyre compiled
        in a single pass), this test treats the CPU forward as ground truth and validates
        the device against it. The device mode being validated is selectable, so this test
        can exercise device-compile-vs-CPU and device-eager-vs-CPU independently.

        For every module input this test:
        1. Instantiates the module on CPU and runs a forward pass (the golden reference).
        2. Instantiates the same module on the device with the SAME weights and runs a
           forward pass in the selected device mode(s).
        3. Compares every output tensor (CPU vs device).

        Which device mode(s) run is controlled by environment variables (torch.compile
        defaults to enabled):
        - TEST_COMPILE_WITH_CPU=1 -> run torch.compile on device
        - TEST_EAGER_WITH_CPU=1   -> run eager on device
        """
        run_compile = os.getenv("TEST_COMPILE_WITH_CPU", "1") == "1"
        run_eager = os.getenv("TEST_EAGER_WITH_CPU", "0") == "1"
        module_inputs = module_info.module_inputs_func(
            module_info,
            device=device,
            dtype=dtype,
            requires_grad=False,
            training=training,
        )

        modes = [
            name
            for name, run in [("compiled", run_compile), ("eager", run_eager)]
            if run
        ]
        if not modes:
            raise ValueError("At least one of run_compile or run_eager must be True")

        for mode in modes:
            for module_input in module_inputs:  # iterate over prefill and decode
                if module_input.forward_input is None:
                    continue

                # === Instantiate the module on CPU (eager). ===
                torch._dynamo.reset_code_caches()
                torch._inductor.codecache.FxGraphCache.clear()
                module_cpu = _construct_module(
                    module_info, module_input, dtype=dtype, training=training
                )
                # Capture the CPU module's (randomly-initialized) weights so the
                # device module below can be given the SAME weights. Without this
                # the two modules have different random weights and outputs never match.
                cpu_state_dict = module_cpu.state_dict()

                # === CPU forward pass. ===
                args_cpu, kwargs_cpu = _move_inputs(
                    module_input, dtype=dtype, device="cpu"
                )
                cpu_tensors = _run_forward(module_cpu, args_cpu, kwargs_cpu)

                # === Instantiate the module on device with the same weights. ===
                torch._dynamo.reset_code_caches()
                torch._inductor.codecache.FxGraphCache.clear()
                module_device = _construct_module(
                    module_info,
                    module_input,
                    dtype=dtype,
                    device=device,
                    training=training,
                )
                module_device.load_state_dict(cpu_state_dict)
                if mode == "compiled":
                    module_device = torch.compile(module_device)

                # === Device forward pass. ===
                # Move inputs to device using pytree to handle nested structures.
                args_device, kwargs_device = _move_inputs(
                    module_input, dtype=dtype, device=device
                )
                # Outputs may be a bare tensor, a tuple/list, or a dict (e.g.
                # attention/decoder layers return hidden_states + attn weights +
                # cache). Flatten both sides with pytree and compare every tensor,
                # moving device tensors back to CPU for the comparison.
                device_tensors = _run_forward(module_device, args_device, kwargs_device)

                if len(cpu_tensors) != len(device_tensors):
                    self.fail(
                        f"{module_info.name}: output tensor count mismatch ({mode}) - "
                        f"CPU: {len(cpu_tensors)}, device: {len(device_tensors)}"
                    )

                for i, (cpu_t, device_t) in enumerate(zip(cpu_tensors, device_tensors)):
                    self.assertEqual(
                        cpu_t,
                        device_t.cpu(),
                        msg=f"{module_info.name}: CPU vs device mismatch ({mode}, tensor {i})",
                    )

    @modules(module_db)
    def test_layout_stride(self, device, dtype, module_info, training):
        """Test module with real YAML-specified layouts and strides.

        Validates modules work correctly with actual SpyreTensorLayouts from YAML config.
        Compares CPU vs device outputs for correctness.
        """
        module_inputs = module_info.module_inputs_func(
            module_info, device=device, dtype=dtype, requires_grad=False, training=False
        )

        for module_input in module_inputs:
            # Create module on CPU
            module_cpu = module_info.module_cls(
                *module_input.constructor_input.args,
                **module_input.constructor_input.kwargs,
            )
            module_cpu.to(dtype)
            module_cpu.eval()

            # Create module on device
            module_device = module_info.module_cls(
                *module_input.constructor_input.args,
                **module_input.constructor_input.kwargs,
            )
            module_device.to(device).to(dtype)
            module_device.eval()

            # Copy weights from CPU to device
            module_device.load_state_dict(module_cpu.state_dict())

            # module_input.forward_input tensors are already placed on `device`
            # by the OOT framework's module_inputs_func (it builds on CPU, then
            # relocates to the test device so upstream's single-module
            # test_forward can use them directly -- see _move_to_test_device in
            # oot_test_config_models.py). They are NOT CPU tensors despite the
            # attribute name, so build a genuine CPU copy for the CPU reference
            # module instead of using them as-is.
            args_device = module_input.forward_input.args
            kwargs_device = module_input.forward_input.kwargs

            args_cpu = tree_map(
                lambda x: x.cpu() if isinstance(x, torch.Tensor) else x, args_device
            )
            kwargs_cpu = tree_map(
                lambda x: x.cpu() if isinstance(x, torch.Tensor) else x, kwargs_device
            )

            # Run forward passes
            with torch.no_grad():
                output_cpu = module_cpu(*args_cpu, **kwargs_cpu)
                output_device = module_device(*args_device, **kwargs_device)

            # Extract all tensors from outputs using pytree
            cpu_tensors = _extract_all_tensors(output_cpu)
            device_tensors = _extract_all_tensors(output_device)

            # Verify both outputs have the same number of tensors
            if len(cpu_tensors) != len(device_tensors):
                self.fail(
                    f"{module_info.name}: Output tensor count mismatch - "
                    f"CPU: {len(cpu_tensors)}, Spyre: {len(device_tensors)}"
                )

            # Compare all tensors (hidden states, KV cache, attention weights, etc.)
            for i, (cpu_t, device_t) in enumerate(zip(cpu_tensors, device_tensors)):
                # See comment in test_eager_vs_compile for why atol/rtol are
                # passed explicitly rather than relying on assertEqual's
                # implicit default+override tolerance resolution.
                self.assertEqual(
                    cpu_t,
                    device_t.cpu(),
                    atol=self.precision,
                    rtol=self.rel_tol,
                    msg=f"{module_info.name}: layout/stride mismatch on real inputs (tensor {i})",
                )

    @modules(module_db)
    def test_vllm(self, device, dtype, module_info, training):
        """Standalone-run a vLLM-native module and compare CPU eager vs device compile.

        Unlike the other tests, vLLM modules need a VllmConfig + TP=1 distributed
        group, so ``_run_vllm_module`` builds, initializes, and runs each module
        entirely inside that context (vLLM's compile/dispatch path reads the
        current config, so the forward must run inside it too, not just the build):

        1. Atomic library layers (vllm.model_executor.layers.*) are built from the
           explicit constructor kwargs captured into the YAML at generation time
           -- ``module_cls(*args, **kwargs)``, no HF config, no per-class adapter.
        2. Composite model modules (vllm.model_executor.models.*) carry a config
           arg; their scalar ctor args are derived from that config via the generic
           ``build_ctor_kwargs_from_config`` (Option C).
        3. A CPU-eager reference and a device (spyre) + torch.compile run use the
           SAME deterministic xavier weights (same seed), and their outputs are
           compared.

        Forward-context-dependent modules (Attention / DecoderLayer) are skipped
        this phase -- their forward needs a global vLLM forward context.
        """
        # Skip cleanly where vLLM is unavailable (e.g. CPU-only dev boxes).
        try:
            import vllm  # noqa: F401
        except ImportError as exc:
            self.skipTest(f"vLLM not available: {exc}")

        # Only vLLM-native modules are standalone targets here. @modules(module_db)
        # fans this test out over every registered module, including PyTorch
        # standard modules (nn.GroupNorm, ...) and any transformers.* modules from
        # HF-generated YAMLs -- those are not vLLM standalone targets. Identify a
        # vLLM module by its class's module path (robust, independent of whether
        # the YAML embedded a model_id).
        module_qualpath = getattr(module_info.module_cls, "__module__", "")
        if not module_qualpath.startswith("vllm."):
            self.skipTest(
                f"{module_info.name}: not a vLLM-native module "
                f"({module_qualpath or '?'}); test_vllm targets vllm.* modules only"
            )

        cls_name = module_info.module_cls.__name__
        if cls_name.endswith("Attention") or cls_name.endswith("DecoderLayer"):
            self.skipTest(
                f"{cls_name}: forward-context-dependent module, deferred to a later phase"
            )

        module_inputs = module_info.module_inputs_func(
            module_info,
            device=device,
            dtype=dtype,
            requires_grad=False,
            training=training,
        )

        for module_input in module_inputs:
            if module_input.forward_input is None:
                continue

            # forward_input tensors are already on `device` (see note in
            # test_layout_stride); build a genuine CPU copy for the reference.
            args_device = module_input.forward_input.args
            kwargs_device = module_input.forward_input.kwargs
            args_cpu = tree_map(
                lambda x: x.cpu() if isinstance(x, torch.Tensor) else x, args_device
            )
            kwargs_cpu = tree_map(
                lambda x: x.cpu() if isinstance(x, torch.Tensor) else x, kwargs_device
            )

            # --- Reference: CPU eager ---
            # Build + init + forward all happen inside _run_vllm_module's
            # set_current_vllm_config context (vLLM's compile/dispatch reads the
            # current config, so the forward must run inside it); the distributed
            # group is torn down there too. A _SkipModuleError is raised for a
            # composite whose ctor cannot be resolved from config.
            try:
                cpu_tensors = _run_vllm_module(
                    module_info.module_cls,
                    module_input.constructor_input,
                    args_cpu,
                    kwargs_cpu,
                    device="cpu",
                    compile=False,
                    dtype=dtype,
                    seed=0,
                )
            except _SkipModuleError as exc:
                self.skipTest(str(exc))

            # --- Under test: device (spyre) + torch.compile ---
            torch._dynamo.reset_code_caches()
            torch._inductor.codecache.FxGraphCache.clear()
            device_tensors = _run_vllm_module(
                module_info.module_cls,
                module_input.constructor_input,
                args_device,
                kwargs_device,
                device=device,
                compile=True,
                dtype=dtype,
                seed=0,  # same seed => same weights as the CPU reference
            )

            if len(cpu_tensors) != len(device_tensors):
                self.fail(
                    f"{module_info.name}: Output tensor count mismatch - "
                    f"CPU: {len(cpu_tensors)}, device: {len(device_tensors)}"
                )

            for i, (cpu_t, device_t) in enumerate(zip(cpu_tensors, device_tensors)):
                print(
                    "###CPU in ", args_cpu[0].device, args_cpu[0].cpu().flatten()[:10]
                )
                print(
                    "###DEV in",
                    args_device[0].device,
                    args_device[0].cpu().flatten()[:10],
                )
                print("###CPU out", cpu_t.device, cpu_t.cpu().flatten()[:10])
                print("###DEV out", device_t.device, device_t.cpu().flatten()[:10])
                self.assertEqual(
                    cpu_t,
                    device_t.cpu(),
                    atol=self.precision,
                    rtol=self.rel_tol,
                    msg=f"{module_info.name}: CPU eager vs device compile mismatch (tensor {i})",
                )


# Instantiate tests for all device types
instantiate_device_type_tests(TestModuleCustom, globals())


if __name__ == "__main__":
    run_tests()
