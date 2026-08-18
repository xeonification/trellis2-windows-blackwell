from typing import *
import torch
from torch.nn.functional import scaled_dot_product_attention as _sdpa

__all__ = [
    'varlen_sdpa',
]


def _as_seqlen_tensor(seqlen: Union[List[int], torch.Tensor], device: torch.device) -> torch.Tensor:
    if torch.is_tensor(seqlen):
        return seqlen.to(device=device, dtype=torch.long)
    return torch.tensor(seqlen, device=device, dtype=torch.long)


def _pack(x: torch.Tensor, batch_idx: torch.Tensor, pos: torch.Tensor, B: int, L: int) -> torch.Tensor:
    """Scatter a varlen tensor [T, ...] into a padded batch [B, L, ...]."""
    out = x.new_zeros((B, L) + x.shape[1:])
    out[batch_idx, pos] = x
    return out


def _indices(seqlen: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batch index and within-sequence position for each token of a varlen layout."""
    device = seqlen.device
    B = seqlen.shape[0]
    offsets = torch.cumsum(seqlen, dim=0) - seqlen
    batch_idx = torch.repeat_interleave(torch.arange(B, device=device), seqlen)
    pos = torch.arange(int(seqlen.sum()), device=device) - torch.repeat_interleave(offsets, seqlen)
    return batch_idx, pos


def varlen_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_seqlen: Union[List[int], torch.Tensor],
    kv_seqlen: Union[List[int], torch.Tensor],
) -> torch.Tensor:
    """
    Variable-length attention via torch's native scaled_dot_product_attention.

    Used as the backend for GPUs without flash-attn / xformers kernels (e.g. Blackwell sm_120).
    Sequences are padded into a batch and masked, so no Python-level loop over sequences is needed.

    Args:
        q: [T_q, H, C] queries, sequences concatenated along dim 0.
        k: [T_kv, H, C] keys, sequences concatenated along dim 0.
        v: [T_kv, H, C_v] values, sequences concatenated along dim 0.
        q_seqlen: per-sequence query lengths.
        kv_seqlen: per-sequence key/value lengths.

    Returns:
        [T_q, H, C_v] output in the same varlen layout as q.
    """
    device = q.device
    q_seqlen = _as_seqlen_tensor(q_seqlen, device)
    kv_seqlen = _as_seqlen_tensor(kv_seqlen, device)
    assert q_seqlen.shape[0] == kv_seqlen.shape[0], \
        f"Sequence count mismatch, got {q_seqlen.shape[0]} and {kv_seqlen.shape[0]}"

    B = q_seqlen.shape[0]
    Lq = int(q_seqlen.max())
    Lk = int(kv_seqlen.max())

    q_batch_idx, q_pos = _indices(q_seqlen)
    kv_batch_idx, kv_pos = _indices(kv_seqlen)

    qp = _pack(q, q_batch_idx, q_pos, B, Lq)    # [B, Lq, H, C]
    kp = _pack(k, kv_batch_idx, kv_pos, B, Lk)  # [B, Lk, H, C]
    vp = _pack(v, kv_batch_idx, kv_pos, B, Lk)  # [B, Lk, H, C_v]

    # [B, H, L, C]
    qp = qp.transpose(1, 2)
    kp = kp.transpose(1, 2)
    vp = vp.transpose(1, 2)

    # Mask out padded key positions.
    if bool((kv_seqlen != Lk).any()):
        key_valid = torch.arange(Lk, device=device).unsqueeze(0) < kv_seqlen.unsqueeze(1)  # [B, Lk]
        attn_mask = key_valid[:, None, None, :]                                            # [B, 1, 1, Lk]
    else:
        attn_mask = None

    out = _sdpa(qp, kp, vp, attn_mask=attn_mask)    # [B, H, Lq, C_v]
    out = out.transpose(1, 2)                       # [B, Lq, H, C_v]

    return out[q_batch_idx, q_pos]                  # [T_q, H, C_v]
