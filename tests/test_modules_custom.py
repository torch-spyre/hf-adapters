"""
Custom module tests for torch-spyre.

This file contains additional test methods for modules defined in YAML configs:
- test_eager_vs_compile: Cross-check that Spyre eager and Spyre compiled agree with
  each other and with CPU (all three run in a single pass)
- test_with_cpu: Use CPU as the golden reference and compare device output against it;
  which device mode(s) to run (compile and/or eager) is selectable via env vars
- test_layout_stride: Validate real YAML-specified SpyreTensorLayouts and strides (CPU vs Spyre)

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


def _granite_decoder_layer_adapter_forward(
    module_cpu, module_device, module_input, dtype, device
):
    """Run GraniteDecoderLayer through the real hf_granite compiled block.

    A bare ``torch.compile(module_device)`` on the stock HF module bypasses
    everything the adapter actually does on Spyre (matmul-form RoPE, the
    pre-allocated KV cache, the fused attn+MLP block) -- that combination is
    what makes attention work on this backend, and it's why raw-module
    compilation fails with runtime errors like "D2H data conversion failed"
    instead of producing comparable output. This builds the same inputs
    ``hf_granite._make_compiled_block`` expects (a rotation-matrix
    ``selected_freqs`` instead of HF's native cos/sin, a prefill causal mask,
    empty KV caches) and calls the adapter's block directly.

    The YAML's ``position_embeddings`` forward-input is a placeholder (random
    values, not valid cos/sin), so it's ignored here in favor of a real
    ``GraniteRotaryEmbedding`` run on both sides from the YAML's
    ``position_ids`` -- CPU gets the native (cos, sin) tuple, device gets the
    matching rotation-matrix form, so the two paths are actually comparable.
    """
    from transformers.models.granite.modeling_granite import GraniteRotaryEmbedding

    from hf_adapters.hf_common import PrecomputedRotaryEmbedding, build_prefill_mask
    from hf_adapters.hf_granite import _make_compiled_block

    config = module_input.constructor_input.args[0]
    hidden_states_cpu = module_input.forward_input.args[0].to("cpu", dtype)
    position_ids_cpu = module_input.forward_input.kwargs["position_ids"].to("cpu")
    hidden_states_device = hidden_states_cpu.to(device, dtype)
    position_ids_device = position_ids_cpu.to(device)

    rotary = GraniteRotaryEmbedding(config)
    cos_cpu, sin_cpu = rotary(hidden_states_cpu, position_ids_cpu)
    precomputed_rope = PrecomputedRotaryEmbedding(rotary)
    # PrecomputedRotaryEmbedding defaults its frequency cache to fp16; without
    # this the matmul-form RoPE below silently promotes bf16 q/k to fp32
    # (mixed bf16 x fp16 multiply), which SDPA then rejects as a dtype
    # mismatch against the bf16 KV cache. prepare_for_spyre's real flow does
    # this via set_rope_dtype(model, dtype).
    precomputed_rope.set_dtype(dtype)
    selected_freqs = precomputed_rope(hidden_states_device, position_ids_device)

    bsz, seq_len, _ = hidden_states_device.shape
    attn_mask = build_prefill_mask(
        bsz, seq_len, seq_len, prompt_offsets=0, dtype=dtype
    ).to(device)

    num_kv_heads = config.num_key_value_heads
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    key_cache = torch.zeros(
        bsz, num_kv_heads, seq_len, head_dim, dtype=dtype, device=device
    )
    value_cache = torch.zeros(
        bsz, num_kv_heads, seq_len, head_dim, dtype=dtype, device=device
    )

    with torch.no_grad():
        cpu_out = module_cpu(
            hidden_states_cpu,
            position_ids=position_ids_cpu,
            position_embeddings=(cos_cpu, sin_cpu),
        )
        compiled_block = _make_compiled_block(module_device)
        device_h, _, _ = compiled_block(
            hidden_states_device,
            selected_freqs,
            attn_mask,
            key_cache,
            value_cache,
            False,  # is_filling: single-shot prefill, not an incremental decode step
            0,  # token_index: unused when is_filling is False
            0,  # cache_position: write the KV cache starting at position 0
        )

    return _extract_all_tensors(cpu_out), [device_h]


# Decoder-layer-granularity modules that have a real hf_adapters compiled
# block. test_with_cpu's "compiled" mode routes these through the adapter's
# actual fused block instead of compiling the stock HF module directly, since
# the stock module's eager/SDPA forward is not what runs on Spyre in
# production. Modules not listed here (MLP, RMSNorm, standalone Attention,
# RotaryEmbedding, ...) keep going through the generic torch.compile path.
#
# Status for GraniteDecoderLayer, verified on Spyre hardware: this closes the
# crash this test used to hit ("D2H data conversion failed" from compiling
# the raw stock module -- see the module docstring above). It still XFAILs
# under the module-test YAML's `mode: xfail`, but now for a real numeric
# reason: ~2-3% of output elements differ from the CPU reference by up to
# ~5x the configured atol=0.005/rtol=0.005. A standalone repro showed the
# divergence spread evenly across every sequence position at a consistent
# magnitude, not concentrated at any boundary -- i.e. not an off-by-one
# masking/RoPE bug, but accumulated bf16 rounding drift between two
# independently-implemented compute paths (CPU eager rotate_half RoPE + CPU
# SDPA vs. Spyre's matmul-form RoPE + Spyre SDPA), compounded across a full
# fused attention+MLP layer. Left as (non-strict) xfail rather than loosening
# the tolerance, since picking a layer-granularity tolerance is a testing-
# policy call, not a bug fix -- worth reconsidering if this becomes a
# priority to get real pass/fail signal from module-level attention tests.
_ADAPTER_DECODER_LAYER_FORWARD = {
    "GraniteDecoderLayer": _granite_decoder_layer_adapter_forward,
}


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

                adapter_forward = _ADAPTER_DECODER_LAYER_FORWARD.get(
                    type(module_cpu).__name__
                )
                if mode == "compiled" and adapter_forward is not None:
                    # Route through the real hf_adapters compiled block
                    # instead of a bare torch.compile of the stock HF module
                    # -- see _ADAPTER_DECODER_LAYER_FORWARD above.
                    cpu_tensors, device_tensors = adapter_forward(
                        module_cpu, module_device, module_input, dtype, device
                    )
                else:
                    # === CPU forward pass. ===
                    args_cpu, kwargs_cpu = _move_inputs(
                        module_input, dtype=dtype, device="cpu"
                    )
                    cpu_tensors = _run_forward(module_cpu, args_cpu, kwargs_cpu)

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
                    device_tensors = _run_forward(
                        module_device, args_device, kwargs_device
                    )

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


# Instantiate tests for all device types
instantiate_device_type_tests(TestModuleCustom, globals())


if __name__ == "__main__":
    run_tests()
