from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
import torch_spyre  # noqa: F401
from transformers import Qwen3VLModel
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision

from hf_adapters import hf_qwen3_vl_embedding as adapter
from hf_adapters.hf_common import apply_rope_matmul, move_model_to_spyre
from tests.cpu.test_qwen3_vl_embedding import _config


def cosine(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return F.cosine_similarity(actual.float(), expected.float(), dim=-1).min().item()


def pad_qk(tensor: torch.Tensor) -> torch.Tensor:
    result = torch.zeros(*tensor.shape[:-1], 128, dtype=tensor.dtype)
    result[..., :32] = tensor[..., :32]
    result[..., 64:96] = tensor[..., 32:]
    return result


def main() -> None:
    torch.manual_seed(7)
    torch.set_num_threads(8)
    model = Qwen3VLModel(_config()).eval().to(torch.bfloat16)
    pixels = torch.randn(8, 24, dtype=torch.bfloat16)
    grid = torch.tensor([[1, 2, 4]])
    reference: list[torch.Tensor] = []
    inputs: list[torch.Tensor] = []

    def hook(module: torch.nn.Module, args: tuple[Any, ...], output: torch.Tensor) -> None:
        inputs.append(args[0].detach().clone())
        reference.append(output.detach().clone())

    handles = [block.register_forward_hook(hook) for block in model.visual.blocks]
    with torch.inference_mode():
        expected = model.visual(pixels, grid, return_dict=True)
        first = model.visual.blocks[0]
        normalized = first.norm1(inputs[0])
        q, k, v = first.attn.qkv(normalized).reshape(8, 3, 2, 64).permute(1, 0, 2, 3).unbind(0)
        from transformers.vision_utils import get_vision_position_ids
        positions = get_vision_position_ids(grid, 2, kwargs={})
        cos, sin = model.visual.rotary_pos_emb(inputs[0], positions)
        rotated_q, rotated_k = apply_rotary_pos_emb_vision(q, k, cos, sin)
        expected_attended = F.scaled_dot_product_attention(rotated_q.transpose(0, 1)[None], rotated_k.transpose(0, 1)[None], v.transpose(0, 1)[None], scale=64 ** -0.5)
        for handle in handles:
            handle.remove()
        move_model_to_spyre(model, adapter, torch.bfloat16)
        q_proj = first.attn._spyre_q_proj
        k_proj = first.attn._spyre_k_proj
        v_proj = first.attn._spyre_v_proj

        def projections(hidden: torch.Tensor, rope: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, ...]:
            norm = first.norm1(hidden)
            query = q_proj(norm).view(1, 64, 2, 128).transpose(1, 2)
            key = k_proj(norm).view(1, 64, 2, 128).transpose(1, 2)
            value = v_proj(norm).view(1, 64, 2, 128).transpose(1, 2)
            rotated_query = apply_rope_matmul(query.float(), rope.float()).to(hidden.dtype)
            rotated_key = apply_rope_matmul(key.float(), rope.float()).to(hidden.dtype)
            attended = F.scaled_dot_product_attention(rotated_query, rotated_key, value, attn_mask=mask, scale=64 ** -0.5)
            contiguous_attended = F.scaled_dot_product_attention(rotated_query.contiguous(), rotated_key.contiguous(), value, attn_mask=mask, scale=64 ** -0.5)
            scores = rotated_query @ rotated_key.transpose(-2, -1) * (64 ** -0.5) + mask
            manual_attended = F.softmax(scores, dim=-1, dtype=torch.float32).to(value.dtype) @ value
            return norm, query, key, value, rotated_query, rotated_key, attended, contiguous_attended, manual_attended

        hidden = F.pad(inputs[0], (0, 0, 0, 56))[None].to("spyre")
        rope = F.pad(adapter._rotation_matrices(cos, sin, 128), (0, 0, 0, 0, 0, 0, 0, 56))[None].to("spyre")
        mask = torch.full((1, 1, 64, 64), -torch.inf, dtype=torch.bfloat16)
        mask[..., :8, :8] = 0
        mask.diagonal(dim1=-2, dim2=-1)[..., 8:] = 0
        actual_components = torch.compile(projections, dynamic=False, fullgraph=True)(hidden, rope, mask.to("spyre"))
        expectations = [normalized[None], pad_qk(q.transpose(0, 1)[None]), pad_qk(k.transpose(0, 1)[None]), F.pad(v.transpose(0, 1)[None], (0, 64)), pad_qk(rotated_q.transpose(0, 1)[None]), pad_qk(rotated_k.transpose(0, 1)[None]), F.pad(expected_attended, (0, 64)), F.pad(expected_attended, (0, 64)), F.pad(expected_attended, (0, 64))]
        for name, actual, expected_component in zip(["norm", "q", "k", "v", "rotated_q", "rotated_k", "attention", "contiguous_attention", "manual_attention"], actual_components, expectations):
            actual = actual.cpu()
            actual = actual[:, :8] if name == "norm" else actual[:, :, :8]
            print(name, "cosine", cosine(actual, expected_component), "max_abs", (actual.float() - expected_component.float()).abs().max().item(), flush=True)

        blocks: list[Callable[..., torch.Tensor]] = model.visual._spyre_compiled_blocks
        for index, compiled in enumerate(blocks):
            def wrap(hidden: torch.Tensor, rope: torch.Tensor, mask: torch.Tensor, compiled: Callable[..., torch.Tensor] = compiled, index: int = index) -> torch.Tensor:
                result = compiled(hidden, rope, mask)
                actual = result.cpu()[0, :8]
                print("vision_block", index, "cosine", cosine(actual, reference[index]), flush=True)
                return result
            blocks[index] = wrap
        actual = model.visual(pixels, grid, return_dict=True)
        print("vision_final", cosine(actual.last_hidden_state, expected.last_hidden_state), flush=True)
        print("vision_merged", cosine(actual.pooler_output, expected.pooler_output), flush=True)
        for index, (actual_feature, expected_feature) in enumerate(zip(actual.deepstack_features, expected.deepstack_features)):
            print("deepstack", index, cosine(actual_feature, expected_feature), flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
