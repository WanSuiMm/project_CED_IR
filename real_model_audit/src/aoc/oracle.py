from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class OracleResult:
    keys: torch.Tensor
    values: torch.Tensor
    log_mass: torch.Tensor
    fit_loss: torch.Tensor


def entropy_effective_rank(scores: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Entropy effective rank over the last two dimensions."""
    singular = torch.linalg.svdvals(scores.float())
    energy = singular.square()
    prob = energy / energy.sum(dim=-1, keepdim=True).clamp_min(eps)
    return torch.exp(-(prob * prob.clamp_min(eps).log()).sum(dim=-1))


def block_targets(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return log-Z and conditional value mean.

    Shapes: q=[H,G,T,D], keys/values=[B,H,R,D].
    Outputs: log_z=[B,H,G,T], mu=[B,H,G,T,D].
    """
    logits = torch.einsum("hgtd,bhrd->bhgtr", q.float(), keys.float()) * scale
    log_z = torch.logsumexp(logits, dim=-1)
    weights = torch.softmax(logits, dim=-1)
    mu = torch.einsum("bhgtr,bhrd->bhgtd", weights, values.float())
    return log_z, mu


def oracle_metrics(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    log_mass: torch.Tensor,
    true_log_z: torch.Tensor,
    true_mu: torch.Tensor,
    query_weight: torch.Tensor,
    scale: float,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = (
        torch.einsum("hgtd,bhcd->bhgtc", q.float(), keys.float()) * scale
        + log_mass.float()[:, :, None, None, :]
    )
    pred_log_z = torch.logsumexp(logits, dim=-1)
    pred_mu = torch.einsum(
        "bhgtc,bhcd->bhgtd", torch.softmax(logits, dim=-1), values.float()
    )
    z_sq = (pred_log_z - true_log_z).square()
    mu_rel = (pred_mu - true_mu).square().sum(dim=-1) / true_mu.square().sum(
        dim=-1
    ).clamp_min(eps)
    w = query_weight.float()
    w = w / w.mean(dim=(-1, -2), keepdim=True).clamp_min(eps)
    z_rmse = (z_sq * w).mean(dim=(-1, -2)).sqrt()
    mu_error = (mu_rel * w).mean(dim=(-1, -2)).sqrt()
    return z_rmse, mu_error, z_rmse.square() + mu_error.square()


def _partition_initialization(
    keys: torch.Tensor, values: torch.Tensor, c: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Contiguous balanced groups preserve token order while giving the optimizer
    # a stable non-random starting point.
    r = keys.shape[-2]
    bounds = torch.linspace(0, r, c + 1, device=keys.device).round().long()
    init_k, init_v, init_b = [], [], []
    for i in range(c):
        lo, hi = int(bounds[i]), int(bounds[i + 1])
        hi = max(hi, lo + 1)
        init_k.append(keys[..., lo:hi, :].mean(dim=-2))
        init_v.append(values[..., lo:hi, :].mean(dim=-2))
        init_b.append(torch.full_like(keys[..., 0, 0], float(hi - lo)).log())
    return (
        torch.stack(init_k, dim=-2),
        torch.stack(init_v, dim=-2),
        torch.stack(init_b, dim=-1),
    )


def _pair_split_initialization(
    keys: torch.Tensor,
    values: torch.Tensor,
    c: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Non-identity c=r start made by duplicating adjacent pair means.

    This deliberately destroys within-pair distinctions while preserving the
    correct latent cardinality and approximate mass. It is used only to test
    whether the optimizer can recover a known-to-exist 8-record solution.
    """
    r = keys.shape[-2]
    if c != r or r % 2:
        raise ValueError("pair_split requires even c == source record count")
    pair_k = keys.reshape(*keys.shape[:-2], r // 2, 2, keys.shape[-1]).mean(dim=-2)
    pair_v = values.reshape(*values.shape[:-2], r // 2, 2, values.shape[-1]).mean(dim=-2)
    init_k = pair_k.repeat_interleave(2, dim=-2)
    init_v = pair_v.repeat_interleave(2, dim=-2)
    init_b = torch.zeros(*keys.shape[:-2], c, device=keys.device, dtype=keys.dtype)
    return init_k, init_v, init_b


def fit_oracle(
    q: torch.Tensor,
    source_keys: torch.Tensor,
    source_values: torch.Tensor,
    query_weight: torch.Tensor,
    c: int,
    scale: float,
    restarts: int = 3,
    steps: int = 250,
    lr: float = 0.05,
    seed: int = 0,
    init_mode: str = "partition",
    noise_scale: float = 0.01,
) -> OracleResult:
    """Fit independent latent records for every [block, KV-head] unit."""
    true_log_z, true_mu = block_targets(q, source_keys, source_values, scale)
    if init_mode == "partition":
        init_k, init_v, init_b = _partition_initialization(source_keys, source_values, c)
    elif init_mode == "pair_split":
        init_k, init_v, init_b = _pair_split_initialization(source_keys, source_values, c)
    else:
        raise ValueError(f"unknown init_mode: {init_mode}")
    # Optimize oracle parameters in fp32 even when the frozen model is bf16.
    init_k, init_v, init_b = init_k.float(), init_v.float(), init_b.float()
    generator = torch.Generator(device=source_keys.device).manual_seed(seed)
    shape_k = (restarts,) + init_k.shape
    k = init_k.unsqueeze(0).expand(shape_k).clone()
    v = init_v.unsqueeze(0).expand(shape_k).clone()
    b = init_b.unsqueeze(0).expand((restarts,) + init_b.shape).clone()
    if restarts > 1:
        k[1:] += noise_scale * torch.randn(k[1:].shape, generator=generator, device=k.device)
        v[1:] += noise_scale * torch.randn(v[1:].shape, generator=generator, device=v.device)
        b[1:] += noise_scale * torch.randn(b[1:].shape, generator=generator, device=b.device)
    k.requires_grad_(True)
    v.requires_grad_(True)
    b.requires_grad_(True)
    optimizer = torch.optim.Adam([k, v, b], lr=lr)
    w = query_weight.float()
    w = w / w.mean(dim=(-1, -2), keepdim=True).clamp_min(1e-8)
    for _ in range(steps):
        logits = (
            torch.einsum("hgtd,rbhcd->rbhgtc", q.float(), k) * scale
            + b[:, :, :, None, None, :]
        )
        pred_z = torch.logsumexp(logits, dim=-1)
        pred_mu = torch.einsum(
            "rbhgtc,rbhcd->rbhgtd", torch.softmax(logits, dim=-1), v
        )
        z_loss = (pred_z - true_log_z.unsqueeze(0)).square()
        mu_loss = (pred_mu - true_mu.unsqueeze(0)).square().sum(dim=-1) / (
            true_mu.square().sum(dim=-1).clamp_min(1e-8).unsqueeze(0)
        )
        per_unit = ((z_loss + mu_loss) * w.unsqueeze(0)).mean(dim=(-1, -2))
        loss = per_unit.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        logits = (
            torch.einsum("hgtd,rbhcd->rbhgtc", q.float(), k) * scale
            + b[:, :, :, None, None, :]
        )
        pred_z = torch.logsumexp(logits, dim=-1)
        pred_mu = torch.einsum(
            "rbhgtc,rbhcd->rbhgtd", torch.softmax(logits, dim=-1), v
        )
        z_loss = (pred_z - true_log_z.unsqueeze(0)).square()
        mu_loss = (pred_mu - true_mu.unsqueeze(0)).square().sum(dim=-1) / (
            true_mu.square().sum(dim=-1).clamp_min(1e-8).unsqueeze(0)
        )
        per_unit = ((z_loss + mu_loss) * w.unsqueeze(0)).mean(dim=(-1, -2))
        best = per_unit.argmin(dim=0)
        block_idx = torch.arange(k.shape[1], device=k.device)[:, None]
        head_idx = torch.arange(k.shape[2], device=k.device)[None, :]
        return OracleResult(
            keys=k[best, block_idx, head_idx].detach(),
            values=v[best, block_idx, head_idx].detach(),
            log_mass=b[best, block_idx, head_idx].detach(),
            fit_loss=per_unit[best, block_idx, head_idx].detach(),
        )
