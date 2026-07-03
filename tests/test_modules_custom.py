"""
Custom module tests for torch-spyre.

This file contains additional test methods for modules defined in YAML configs:
- test_eager_vs_compile: Compare eager and compile mode outputs (CPU vs Spyre eager vs Spyre compiled)
- test_with_cpu: Compare CPU (eager) vs device (compile and/or eager), selectable via env vars
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
            module_cpu.eval()

            # Create module on device (eager)
            module_device_eager = module_info.module_cls(
                *module_input.constructor_input.args,
                **module_input.constructor_input.kwargs,
            ).to(device)
            module_device_eager.eval()

            # Copy weights from CPU to device
            module_device_eager.load_state_dict(module_cpu.state_dict())

            # Create compiled version
            module_device_compile_base = module_info.module_cls(
                *module_input.constructor_input.args,
                **module_input.constructor_input.kwargs,
            ).to(device)
            module_device_compile_base.eval()
            module_device_compile_base.load_state_dict(module_cpu.state_dict())
            module_device_compile = torch.compile(module_device_compile_base)

            # Prepare inputs
            args_cpu = module_input.forward_input.args
            kwargs_cpu = module_input.forward_input.kwargs

            # Move inputs to device using pytree to handle nested structures
            args_device = tree_map(
                lambda x: x.to(device) if isinstance(x, torch.Tensor) else x, args_cpu
            )
            kwargs_device = tree_map(
                lambda x: x.to(device) if isinstance(x, torch.Tensor) else x, kwargs_cpu
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
                self.assertEqual(
                    cpu_t,
                    eager_t.cpu(),
                    msg=f"{module_info.name}: CPU eager vs Spyre eager mismatch (tensor {i})",
                )

                # Compare Spyre eager vs Spyre compile
                self.assertEqual(
                    eager_t,
                    compile_t,
                    msg=f"{module_info.name}: Spyre eager vs Spyre compile mismatch (tensor {i})",
                )

    @modules(module_db)
    def test_with_cpu(self, device, dtype, module_info, training):
        """Compare CPU (eager) outputs against device outputs for each module.

        For every module input this test:
        1. Instantiates the module on CPU (eager) and runs a forward pass.
        2. Instantiates the same module on the device with the SAME weights and
           runs a forward pass in compile and/or eager mode.
        3. Compares every output tensor (CPU vs device).

        Which device mode(s) run is controlled by environment variables (torch.compile
        default to enabled):
        - TEST_COMPILE_WITH_CPU=1 -> run torch.compile on device
        - TEST_EAGER_WITH_CPU=1   -> run eager on device
        """
        run_compile = os.getenv("TEST_COMPILE_WITH_CPU", "1") == "1"
        run_eager = os.getenv("TEST_EAGER_WITH_CPU", "0") == "1"
        module_cls = module_info.module_cls
        module_inputs = module_info.module_inputs_func(
            module_info, device=device, dtype=dtype, requires_grad=False, training=training
        )

        modes = [name for name, run in [("compiled", run_compile), ("eager", run_eager)] if run]
        if not modes:
            raise ValueError("At least one of run_compile or run_eager must be True")

        for mode in modes:
            for module_input in module_inputs:  # iterate over prefill and decode
                if module_input.forward_input is None:
                    continue

                # === Instantiate the module on CPU (eager). ===
                torch._dynamo.reset_code_caches()
                torch._inductor.codecache.FxGraphCache.clear()
                args_ctor = module_input.constructor_input.args
                kwargs_ctor = module_input.constructor_input.kwargs
                module_cpu = module_cls(*args_ctor, **kwargs_ctor)
                module_cpu = module_cpu.to(dtype)
                module_cpu.train(training)
                # Capture the CPU module's (randomly-initialized) weights so the
                # device module below can be given the SAME weights. Without this
                # the two modules have different random weights and outputs never match.
                cpu_state_dict = module_cpu.state_dict()

                # === CPU forward pass. ===
                args = module_input.forward_input.args
                kwargs = module_input.forward_input.kwargs
                args_cpu = tree_map(
                    lambda x: x.to(dtype) if isinstance(x, torch.Tensor) else x, args
                )
                kwargs_cpu = tree_map(
                    lambda x: x.to(dtype) if isinstance(x, torch.Tensor) else x, kwargs
                )
                with torch.no_grad():
                    cpu_outputs = module_cpu(*args_cpu, **kwargs_cpu)

                # === Instantiate the module on device with the same weights. ===
                torch._dynamo.reset_code_caches()
                torch._inductor.codecache.FxGraphCache.clear()
                module_device = module_cls(*args_ctor, **kwargs_ctor)
                module_device.load_state_dict(cpu_state_dict)
                module_device = module_device.to(dtype).to(device)
                module_device.train(training)
                if mode == "compiled":
                    module_device = torch.compile(module_device)

                # === Device forward pass. ===
                # Move inputs to device using pytree to handle nested structures.
                args_device = tree_map(
                    lambda x: x.to(device, dtype) if isinstance(x, torch.Tensor) else x, args
                )
                kwargs_device = tree_map(
                    lambda x: x.to(device, dtype) if isinstance(x, torch.Tensor) else x, kwargs
                )
                with torch.no_grad():
                    device_outputs = module_device(*args_device, **kwargs_device)

                # Outputs may be a bare tensor, a tuple/list, or a dict (e.g.
                # attention/decoder layers return hidden_states + attn weights +
                # cache). Flatten both sides with pytree and compare every tensor,
                # moving device tensors back to CPU for the comparison.
                cpu_tensors = _extract_all_tensors(cpu_outputs)
                device_tensors = _extract_all_tensors(device_outputs)

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
            module_cpu.eval()

            # Create module on device
            module_device = module_info.module_cls(
                *module_input.constructor_input.args,
                **module_input.constructor_input.kwargs,
            ).to(device)
            module_device.eval()

            # Copy weights from CPU to device
            module_device.load_state_dict(module_cpu.state_dict())

            # Prepare inputs
            args_cpu = module_input.forward_input.args
            kwargs_cpu = module_input.forward_input.kwargs

            # Move inputs to device using pytree to handle nested structures
            args_device = tree_map(
                lambda x: x.to(device) if isinstance(x, torch.Tensor) else x, args_cpu
            )
            kwargs_device = tree_map(
                lambda x: x.to(device) if isinstance(x, torch.Tensor) else x, kwargs_cpu
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
                self.assertEqual(
                    cpu_t,
                    device_t.cpu(),
                    msg=f"{module_info.name}: layout/stride mismatch on real inputs (tensor {i})",
                )


# Instantiate tests for all device types
instantiate_device_type_tests(TestModuleCustom, globals())


if __name__ == "__main__":
    run_tests()
