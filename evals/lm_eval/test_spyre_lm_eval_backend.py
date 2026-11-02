"""CPU contract tests; run with python -m unittest evals.lm_eval.test_spyre_lm_eval_backend."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from lm_eval.models.huggingface import HFLM
from transformers import Qwen3Config, Qwen3ForCausalLM, StoppingCriteriaList

from evals.lm_eval.run_spyre_lm_eval import SpyreHFLM
from hf_adapters import hf_common, hf_qwen3


class ModelCallTests(unittest.TestCase):
    def setUp(self):
        # These are CPU tests; manual_seed() also initializes attached devices.
        torch.random.default_generator.manual_seed(42)
        config = Qwen3Config(
            vocab_size=97,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=128,
            max_position_embeddings=256,
        )
        self.model = Qwen3ForCausalLM(config).eval()
        self.model.config._attn_implementation = "eager"
        self.backend = object.__new__(SpyreHFLM)
        self.backend.backend = "causal"
        self.backend._model = self.model
        self.backend._spyre_adapter = hf_qwen3
        self.backend.mixed_precision_dtype = None
        self.backend._device = torch.device("cpu")

    def test_matches_hf_logits_and_resets_caches(self):
        inputs = torch.tensor([[3, 0, 7, 2, 8], [1, 4, 6, 0, 0]])
        with torch.no_grad():
            reference = self.model(inputs, use_cache=False).logits
        # Exercise the real adapter math without needing a Spyre compiler.
        with (
            patch.object(hf_common, "DEVICE", "cpu"),
            patch("torch.compile", side_effect=lambda fn, **kwargs: fn),
        ):
            hf_qwen3.prepare_for_spyre(self.model)
            actual = self.backend._model_call(inputs)
            repeat = self.backend._model_call(inputs)
        self.assertEqual(actual.shape, (2, 5, 97))
        self.assertEqual(actual.device.type, "cpu")
        self.assertEqual(actual.dtype, torch.float32)
        # Precomputed RoPE uses lower-precision frequency storage than HF.
        torch.testing.assert_close(actual, reference, rtol=1e-4, atol=3e-5)
        torch.testing.assert_close(repeat, actual)

    def test_forwards_block_boundary_and_fresh_caches(self):
        calls = []

        def forward(model, ids, positions, mask, keys, values, cache_index):
            calls.append((ids, positions, mask, keys, cache_index))
            return torch.zeros(ids.shape[0], ids.shape[1], 128)

        self.backend._spyre_adapter = SimpleNamespace(_run_forward=forward)
        for length in (1, 64, 65):
            result = self.backend._model_call(torch.ones(1, length, dtype=torch.long))
            self.assertEqual(result.shape, (1, length, 97))
            ids, positions, mask, keys, index = calls[-1]
            padded = 64 if length <= 64 else 128
            self.assertEqual(ids.shape, (1, padded))
            self.assertEqual(mask.shape, (1, 1, padded, 512))
            self.assertEqual(index.numel(), padded)
            self.assertEqual(positions[0, -1].item(), padded - 1)
            self.assertTrue(torch.all(mask[0, 0, 0, 1:] < 0))
            self.assertTrue(torch.all(keys[0] == 0))
        self.assertIsNot(calls[0][3][0], calls[1][3][0])

    def test_rejects_unsupported_inputs(self):
        for inputs in (
            torch.empty(1, 0, dtype=torch.long),
            torch.ones(5, dtype=torch.long),
        ):
            with self.assertRaises(ValueError):
                self.backend._model_call(inputs)
        with self.assertRaises(TypeError):
            self.backend._model_call(torch.ones(1, 5))
        with self.assertRaises(ValueError):
            self.backend._model_call(
                torch.ones(1, 5, dtype=torch.long), labels=torch.ones(1, 5)
            )

    def test_model_call_pads_to_adapter_chunk_size_and_crops_logits(self):
        self.model._spyre_prefill_chunk_size = 512
        for length in (1, 512, 513):
            with self.subTest(length=length):
                padded = ((length + 511) // 512) * 512
                inputs = torch.ones(1, length, dtype=torch.long)
                raw = torch.ones(1, padded, 128, dtype=torch.bfloat16)
                with patch(
                    "evals.lm_eval.run_spyre_lm_eval.prefill_logits", return_value=raw
                ) as helper:
                    result = self.backend._model_call(inputs)
                _, _, ids, positions, offsets = helper.call_args.args
                self.assertEqual(ids.shape, (1, padded))
                torch.testing.assert_close(ids[:, :length], inputs)
                self.assertTrue(torch.all(ids[:, length:] == 0))
                torch.testing.assert_close(
                    positions, torch.arange(padded, dtype=torch.int32)[None]
                )
                torch.testing.assert_close(offsets, torch.zeros(1, dtype=torch.long))
                self.assertEqual(result.shape, (1, length, 97))
                self.assertEqual(result.dtype, torch.float32)

    def test_loader_bridge_uses_cpu_scoring_device(self):
        loader = Mock(return_value=self.model)
        self.backend.AUTO_MODEL_CLASS = SimpleNamespace(from_pretrained=loader)
        with patch(
            "evals.lm_eval.run_spyre_lm_eval.resolve_adapter_module",
            return_value=hf_qwen3,
        ):
            self.backend._create_model("test/model", dtype="bfloat16")
        loader.assert_called_once_with(
            "test/model", dtype=torch.bfloat16, tp_plan=None, trust_remote_code=False
        )
        self.assertEqual(self.backend.device.type, "cpu")

    def test_loader_delegates_checkpoint_fp8_metadata_to_adapter(self):
        from transformers import FineGrainedFP8Config

        metadata = FineGrainedFP8Config()
        original_metadata = metadata.to_dict()
        loader = Mock(return_value=self.model)
        self.backend.AUTO_MODEL_CLASS = SimpleNamespace(from_pretrained=loader)
        with (
            patch(
                "evals.lm_eval.run_spyre_lm_eval.dtype_for_model_path",
                return_value=torch.bfloat16,
            ),
            patch(
                "evals.lm_eval.run_spyre_lm_eval.resolve_adapter_module",
                return_value=hf_qwen3,
            ),
        ):
            self.backend._create_model(
                "mistralai/Ministral-3-14B-Instruct-2512",
                dtype="auto",
                quantization_config=metadata,
            )
        # The ordinary loader re-reads the checkpoint and owns dequantization;
        # the evaluation bridge neither overrides nor strips its metadata.
        loader.assert_called_once_with(
            "mistralai/Ministral-3-14B-Instruct-2512",
            dtype=torch.bfloat16,
            tp_plan=None,
            trust_remote_code=False,
        )
        self.assertEqual(metadata.to_dict(), original_metadata)
        self.assertIs(self.backend.model, self.model)

    def test_model_call_delegates_to_shared_helper(self):
        inputs = torch.ones(1, 5, dtype=torch.long)
        raw = torch.ones(1, 64, 128, dtype=torch.bfloat16)
        with patch(
            "evals.lm_eval.run_spyre_lm_eval.prefill_logits",
            return_value=raw,
        ) as helper:
            result = self.backend._model_call(inputs)
        helper.assert_called_once()
        forward, model, ids, positions, offsets = helper.call_args.args
        self.assertIs(forward, hf_qwen3._run_forward)
        self.assertIs(model, self.model)
        torch.testing.assert_close(ids[:, :5], inputs)
        self.assertTrue(torch.all(ids[:, 5:] == 0))
        torch.testing.assert_close(positions, torch.arange(64, dtype=torch.int32)[None])
        torch.testing.assert_close(offsets, torch.zeros(1, dtype=torch.long))
        self.assertEqual(result.shape, (1, 5, 97))
        self.assertEqual(result.dtype, torch.float32)

    def test_shared_helper_without_harness(self):
        inputs = torch.ones(1, 64, dtype=torch.int32)
        positions = torch.arange(64, dtype=torch.int32)[None]
        offsets = torch.tensor([3])
        raw = torch.ones(1, 64, 128, dtype=torch.bfloat16)

        def forward(model, ids, positions, mask, keys, values, cache_index):
            self.assertFalse(torch.is_grad_enabled())
            self.assertIs(ids, inputs)
            self.assertIs(positions, positions_arg)
            self.assertTrue(torch.all(mask[:, :, :, :3] < 0))
            return raw

        positions_arg = positions
        actual = hf_common.prefill_logits(
            forward, self.model, inputs, positions, offsets
        )
        self.assertIs(actual, raw)
        self.model._spyre_prefill_chunk_size = 64
        actual = hf_common.prefill_logits(
            lambda *args, **kwargs: raw, self.model, inputs, positions, offsets
        )
        torch.testing.assert_close(actual, raw)

    def test_shared_cache_setup_installs_metadata_before_allocator(self):
        offsets = torch.tensor([3])
        observations = []

        def allocator(model, batch_size, length, dtype, device):
            observations.append(
                (
                    model._spyre_prompt_offsets,
                    model._spyre_padded_prompt_len,
                    model._spyre_active_prefill_chunk_size,
                    length,
                    device,
                )
            )
            return [torch.zeros(1)], [torch.zeros(1)]

        self.model._spyre_cache_allocator = allocator
        chunked = hf_common._initialize_prefill_caches(
            self.model, 1, 512, offsets, 1, prefill_chunk_size=64, device="cpu"
        )
        self.assertEqual((chunked.max_cache_len, chunked.prefill_kv_len), (1024, 512))
        self.assertIs(observations[-1][0], offsets)
        self.assertEqual(observations[-1][1:], (512, 64, 1024, "cpu"))
        oneshot = hf_common._initialize_prefill_caches(
            self.model, 1, 64, offsets, device="cpu"
        )
        self.assertEqual((oneshot.max_cache_len, oneshot.prefill_kv_len), (512, 512))
        self.assertIsNone(observations[-1][2])

    def test_generation_uses_shared_setup_for_oneshot_chunked_and_custom_prefill(self):
        inputs = torch.tensor([[3, 4, 5]])

        def forward(model, ids, positions, mask, keys, values, cache_index):
            return torch.zeros(1, ids.shape[1], 128)

        def custom_prefill(**kwargs):
            return torch.zeros(1, kwargs["input_ids"].shape[1], 128)

        for options in ({}, {"prefill_chunk_size": 64}, {"prefill_fn": custom_prefill}):
            with (
                self.subTest(options=options),
                patch.object(hf_common, "DEVICE", "cpu"),
                patch.object(
                    hf_common,
                    "_initialize_prefill_caches",
                    wraps=hf_common._initialize_prefill_caches,
                ) as setup,
            ):
                result = hf_common.generate(
                    forward,
                    self.model,
                    inputs,
                    max_new_tokens=2,
                    do_sample=False,
                    eos_token_id=None,
                    **options,
                )
            setup.assert_called_once()
            torch.testing.assert_close(result, torch.tensor([[3, 4, 5, 0, 0]]))
            self.assertEqual(
                self.model._spyre_active_prefill_chunk_size,
                options.get("prefill_chunk_size"),
            )

    def test_generation_stopping_criteria_see_original_prefix_and_decode(self):
        calls = []
        observed = []

        def forward(model, ids, *args, **kwargs):
            calls.append(ids.shape[1])
            logits = torch.zeros(2, ids.shape[1], 128)
            logits[:, :, 7] = 10
            return logits

        def criteria(ids, scores):
            observed.append((ids.clone(), scores))
            # Each row stops independently; row 0 emits padding on step 2.
            return torch.tensor([True, ids.shape[1] >= 5])

        inputs = torch.tensor([[3, 4, 5], [6, 4, 5]])
        with patch.object(hf_common, "DEVICE", "cpu"):
            result = hf_common.generate(
                forward,
                self.model,
                inputs,
                max_new_tokens=5,
                do_sample=False,
                eos_token_id=None,
                pad_token_id=0,
                stopping_criteria=StoppingCriteriaList([criteria]),
            )
        self.assertEqual(calls, [64, 1])
        torch.testing.assert_close(
            result, torch.tensor([[3, 4, 5, 7, 0], [6, 4, 5, 7, 7]])
        )
        self.assertEqual(len(observed), 2)
        torch.testing.assert_close(observed[0][0][:, :3], inputs)
        self.assertIsNone(observed[0][1])

    def test_generation_bridge_preserves_masks_and_builds_stops(self):
        context = torch.tensor([[3, 4, 5]])
        mask = torch.ones_like(context)
        self.backend.tokenizer = SimpleNamespace(pad_token_id=0)
        output = torch.tensor([[3, 4, 5, 7]])
        self.model.generate = Mock(return_value=output)
        with patch(
            "lm_eval.models.huggingface.stop_sequences_criteria",
            return_value="criteria",
        ) as builder:
            actual = self.backend._model_generate(
                context,
                8,
                ["Question:"],
                attention_mask=mask,
                temperature=0.0,
            )
        self.assertIs(actual, output)
        builder.assert_called_once_with(self.backend.tokenizer, ["Question:"], 3, 1)
        self.model.generate.assert_called_once_with(
            input_ids=context,
            max_length=8,
            stopping_criteria="criteria",
            pad_token_id=0,
            attention_mask=mask,
            do_sample=False,
            use_cache=True,
        )

    def test_generation_method_is_inherited(self):
        self.assertIs(SpyreHFLM._model_generate, HFLM._model_generate)

    def test_init_rejects_unsupported_options_before_loading(self):
        for kwargs in (
            {"backend": "seq2seq"},
            {"mixed_precision_dtype": "bfloat16"},
            {"parallelize": True},
            {"tp_plan": "auto"},
            {"revision": "other"},
            {"peft": "adapter"},
            {"device_map": "auto"},
            {"device": "cuda"},
            {"quantization_config": {"quant_method": "fp8"}},
            {"autogptq": True},
            {"gptqmodel": True},
            {"load_in_4bit": True},
            {"load_in_8bit": True},
        ):
            with (
                self.subTest(kwargs=kwargs),
                patch.object(HFLM, "__init__", autospec=True) as parent,
            ):
                with self.assertRaises(ValueError):
                    SpyreHFLM("test/model", **kwargs)
                parent.assert_not_called()

    def test_init_normalizes_cpu_device_and_causal_backend(self):
        with patch.object(HFLM, "__init__", autospec=True) as parent:
            parent.return_value = None
            backend = SpyreHFLM("test/model", device="spyre", dtype="bfloat16")
        parent.assert_called_once_with(
            backend,
            pretrained="test/model",
            device="cpu",
            backend="causal",
            dtype="bfloat16",
        )

    def test_multitoken_stop_criterion_uses_only_generated_text(self):
        class Tokenizer:
            pad_token_id = 0

            def encode(self, text, **kwargs):
                return [7, 8]

            def batch_decode(self, rows):
                return [
                    "".join({7: "STOP", 8: "END"}.get(int(t), "x") for t in row)
                    for row in rows
                ]

        self.backend.tokenizer = Tokenizer()

        def forward(model, ids, *args, **kwargs):
            token = 7 if ids.shape[1] > 1 else 8
            logits = torch.zeros(1, ids.shape[1], 128)
            logits[:, :, token] = 10
            return logits

        self.model.generate = lambda **kwargs: hf_common.generate(
            forward, self.model, eos_token_id=None, **kwargs
        )
        with patch.object(hf_common, "DEVICE", "cpu"):
            result = self.backend._model_generate(
                torch.tensor([[7, 8, 3]]),
                10,
                ["STOPEND"],
            )
        torch.testing.assert_close(result, torch.tensor([[7, 8, 3, 7, 8]]))


if __name__ == "__main__":
    unittest.main()
