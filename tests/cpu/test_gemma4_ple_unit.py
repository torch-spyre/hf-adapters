import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from hf_adapters import hf_gemma4

E2B = "google/gemma-4-E2B-it"


def test_compute_per_layer_inputs_matches_hf():
    model = AutoModelForCausalLM.from_pretrained(E2B, dtype=torch.float32)
    backbone = hf_gemma4._gemma4_backbone(model)
    input_ids = torch.tensor([[2, 100, 200, 300, 400]])

    # Stock HF reference (both PLE components combined).
    inputs_embeds = backbone.embed_tokens(input_ids)
    ref_token = backbone.get_per_layer_inputs(input_ids, inputs_embeds)
    ref = backbone.project_per_layer_inputs(inputs_embeds, ref_token)

    got = hf_gemma4._compute_per_layer_inputs(model, inputs_embeds, input_ids)

    assert got is not None
    assert got.shape == ref.shape  # [1, 5, 35, 256]
    assert torch.allclose(got, ref, atol=1e-5, rtol=1e-4)


def test_block_ple_tail_matches_hf_layer():
    model = AutoModelForCausalLM.from_pretrained(E2B, dtype=torch.float32)
    backbone = hf_gemma4._gemma4_backbone(model)
    layer = backbone.layers[0]  # non-shared layer (idx 0 < first_shared 15)
    hf_gemma4._patch_gemma4_rmsnorm(type(layer.input_layernorm))

    B, S, H = 1, 4, hf_gemma4.text_config(model.config).hidden_size
    input_ids = torch.tensor([[2, 10, 20, 30]])
    inputs_embeds = backbone.embed_tokens(input_ids)
    ple = hf_gemma4._compute_per_layer_inputs(model, inputs_embeds, input_ids)
    per_layer_input = ple[:, :, 0, :]  # layer 0 slice [B,S,ple_dim]

    # Reference: apply just the PLE tail as stock HF does (post-residual state h).
    h = torch.randn(B, S, H)
    ref = h + layer.post_per_layer_input_norm(
        layer.per_layer_projection(
            F.gelu(layer.per_layer_input_gate(h), approximate="tanh") * per_layer_input
        )
    )

    got = hf_gemma4._ple_tail(layer, h, per_layer_input)
    assert torch.allclose(got, ref, atol=1e-5)
