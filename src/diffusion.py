"""
Masked Diffusion Process (STABLE VERSION)
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class DiffusionConfig:
    mask_token_id: int
    schedule: str = "linear"


# ----------------------------
# Mask schedule
# ----------------------------

def get_mask_schedule(t: torch.Tensor, schedule: str = "linear") -> torch.Tensor:
    if schedule == "linear":
        return t
    elif schedule == "cosine":
        return 1 - torch.cos(t * torch.pi / 2)
    elif schedule == "sqrt":
        return torch.sqrt(t)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


# ----------------------------
# Corruption
# ----------------------------

def corrupt_batch(
    tokens: torch.Tensor,
    t: torch.Tensor,
    mask_token_id: int,
    schedule: str = "linear",
) -> Tuple[torch.Tensor, torch.Tensor]:

    batch_size, seq_len = tokens.shape
    device = tokens.device

    mask_prob = get_mask_schedule(t, schedule)

    # 🔥 STABILITY FIX: never mask entire sequence
    max_mask_ratio = 0.8
    mask_prob = torch.clamp(mask_prob, max=max_mask_ratio)

    rand = torch.rand(batch_size, seq_len, device=device)
    mask = rand < mask_prob.unsqueeze(1)

    corrupted = torch.where(mask, mask_token_id, tokens)

    return corrupted, mask


# ----------------------------
# Training loss
# ----------------------------

def compute_loss(
    model,
    tokens: torch.Tensor,
    mask_token_id: int,
    schedule: str = "linear",
) -> torch.Tensor:

    batch_size, seq_len = tokens.shape
    device = tokens.device

    t = torch.rand(batch_size, device=device)

    corrupted, mask = corrupt_batch(tokens, t, mask_token_id, schedule)

    logits = model(corrupted, t)

    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = tokens.view(-1)
    mask_flat = mask.view(-1)

    # 🔥 STABILITY FIX: ignore GPT2 EOS token
    eos_token_id = 50256
    valid_mask = mask_flat & (targets_flat != eos_token_id)

    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    loss = F.cross_entropy(
        logits_flat[valid_mask],
        targets_flat[valid_mask]
    )

    return loss


# ----------------------------
# Sampling
# ----------------------------

@torch.no_grad()
def sample(
    model,
    seq_len: int,
    mask_token_id: int,
    batch_size: int = 1,
    steps: int = 50,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    schedule: str = "linear",
    device: str = "cuda",
    prompt_tokens: Optional[torch.Tensor] = None,
    prompt_mask: Optional[torch.Tensor] = None,
):

    model.eval()

    if prompt_tokens is not None:
        x = prompt_tokens.clone()
        if prompt_mask is None:
            prompt_mask = torch.ones_like(prompt_tokens, dtype=torch.bool)
        x = torch.where(prompt_mask, prompt_tokens, mask_token_id)
    else:
        x = torch.full((batch_size, seq_len), mask_token_id, device=device)
        prompt_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)

    timesteps = torch.linspace(1, 0, steps + 1, device=device)[:-1]

    for step_idx, t in enumerate(timesteps):

        t_batch = torch.full((batch_size,), t.item(), device=device)
        logits = model(x, t_batch)

        # 🔥 STABILITY FIX: block EOS early
        if step_idx < steps // 2:
            logits[..., 50256] = -float("inf")

        logits = logits / temperature

        if top_k is not None:
            top_k_vals, _ = logits.topk(top_k, dim=-1)
            threshold = top_k_vals[..., -1:]
            logits = torch.where(logits < threshold, float('-inf'), logits)

        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            logits = torch.where(indices_to_remove, float('-inf'), logits)

        probs = F.softmax(logits, dim=-1)

        sampled = torch.multinomial(
            probs.view(-1, probs.size(-1)),
            num_samples=1
        ).view(batch_size, seq_len)

        is_masked = (x == mask_token_id) & ~prompt_mask

        if not is_masked.any():
            break

        confidence = probs.max(dim=-1).values
        confidence = torch.where(is_masked, confidence, torch.tensor(-1.0, device=device))

        n_masked = is_masked.sum(dim=-1).float()
        remaining_steps = max(1, steps - step_idx)
        n_to_unmask = (n_masked / remaining_steps).clamp(min=1)

        for b in range(batch_size):
            n_unmask = int(n_to_unmask[b].item())
            masked_idx = is_masked[b].nonzero().squeeze(-1)

            if len(masked_idx) > 0:
                conf = confidence[b, masked_idx]
                k = min(n_unmask, len(masked_idx))
                top_indices = conf.topk(k).indices
                unmask_pos = masked_idx[top_indices]
                x[b, unmask_pos] = sampled[b, unmask_pos]

    remaining_mask = (x == mask_token_id) & ~prompt_mask
    if remaining_mask.any():
        t_final = torch.zeros(batch_size, device=device)
        logits = model(x, t_final)
        final_preds = logits.argmax(dim=-1)
        x = torch.where(remaining_mask, final_preds, x)

    return x