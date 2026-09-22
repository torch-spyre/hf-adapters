import torch

from hf_adapters.hf_common import _apply_standard_gqa_attention_dim_names


def test_kv_sequence_axis_is_untracked_across_for_each_tile_boundary():
    query = torch.empty(4, 32, 256, 128)
    key = torch.empty(4, 8, 512, 128)
    value = torch.empty(4, 8, 512, 128)
    declarations = []
    named_tensors = []

    _apply_standard_gqa_attention_dim_names(
        query,
        key,
        value,
        lambda name, size: declarations.append((name, size)),
        lambda tensor, names: named_tensors.append((tensor, names)),
    )

    assert declarations == [
        ("_b", 4),
        ("num_heads", 32),
        ("num_kvheads", 8),
        ("max_seqlen_q", 256),
        ("_untracked_512", 512),
        ("head_dim", 128),
        ("value_head_dim", 128),
    ]
    assert len(named_tensors) == 3
    assert named_tensors[0][0] is query
    assert named_tensors[0][1] == [
        "_b",
        "num_heads",
        "max_seqlen_q",
        "head_dim",
    ]
    assert named_tensors[1][0] is key
    assert named_tensors[1][1] == [
        "_b",
        "num_kvheads",
        "_untracked_512",
        "head_dim",
    ]
    assert named_tensors[2][0] is value
    assert named_tensors[2][1] == [
        "_b",
        "num_kvheads",
        "_untracked_512",
        "value_head_dim",
    ]
    assert all("max_seqlen_kv" not in names for _, names in named_tensors)
