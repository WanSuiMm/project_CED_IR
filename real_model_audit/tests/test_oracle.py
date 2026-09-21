import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from aoc.oracle import block_targets, entropy_effective_rank, fit_oracle, oracle_metrics


def test_effective_rank_identifies_rank_one():
    a = torch.randn(3, 7, 1)
    b = torch.randn(3, 1, 8)
    rank = entropy_effective_rank(a @ b)
    assert torch.allclose(rank, torch.ones_like(rank), atol=1e-5)


def test_identity_block_metrics_are_zero():
    torch.manual_seed(1)
    q = torch.randn(2, 2, 6, 4)
    k = torch.randn(3, 2, 8, 4)
    v = torch.randn(3, 2, 8, 4)
    z, mu = block_targets(q, k, v, 0.5)
    weight = torch.ones(3, 2, 2, 6)
    bias = torch.zeros(3, 2, 8)
    zr, mr, _ = oracle_metrics(q, k, v, bias, z, mu, weight, 0.5)
    assert zr.max().item() < 1e-6
    assert mr.max().item() < 1e-6


def test_oracle_learns_duplicated_records():
    torch.manual_seed(2)
    q = torch.randn(1, 2, 24, 6)
    base_k = torch.randn(2, 1, 1, 6)
    base_v = torch.randn(2, 1, 1, 6)
    k = base_k.expand(2, 1, 8, 6).contiguous()
    v = base_v.expand(2, 1, 8, 6).contiguous()
    weight = torch.ones(2, 1, 1, 24)
    result = fit_oracle(q, k, v, weight, c=1, scale=6**-0.5, steps=120)
    z, mu = block_targets(q, k, v, 6**-0.5)
    zr, mr, _ = oracle_metrics(
        q, result.keys, result.values, result.log_mass, z, mu, weight, 6**-0.5
    )
    assert zr.median().item() < 1e-2
    assert mr.median().item() < 1e-2


def test_pair_split_is_non_identity_and_can_optimize():
    torch.manual_seed(3)
    q = torch.randn(1, 2, 32, 6)
    k = torch.randn(2, 1, 8, 6)
    v = torch.randn(2, 1, 8, 6)
    weight = torch.ones(2, 1, 1, 32)
    result = fit_oracle(
        q, k, v, weight, c=8, scale=6**-0.5, steps=4,
        restarts=1, init_mode="pair_split",
    )
    assert not torch.allclose(result.keys, k.float())
    assert torch.isfinite(result.fit_loss).all()
