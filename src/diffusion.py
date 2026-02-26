"""
Masked Diffusion Process

Handles:
- Forward process: corrupting clean text with masks
- Reverse process: iterative denoising/unmasking
- Different masking schedules
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class DiffusionConfig:
    mask_token_id: int
    eos_token_id: int = None
    pad_token_id: int = None
    schedule: str = "linear"
    max_mask_ratio: float = 0.95  # configurable now


def get_mask_schedule(t: torch.Tensor, schedule: str = "linear") -> torch.Tensor:
    """
    Convert timestep t to masking probability.
    t=0 → mask_prob=0 (clean)
    t=1 → mask_prob=1 (all masked)
    """
    if schedule == "linear":
        return t
    elif schedule == "cosine":
        return 1 - torch.cos(t * torch.pi / 2)
    elif schedule == "sqrt":
        return torch.sqrt(t)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


def corrupt_batch(
    tokens: torch.Tensor,
    t: torch.Tensor,
    mask_token_id: int,
    schedule: str = "linear",
    max_mask_ratio: float = 0.95,
    pad_token_id: int = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Corrupt a batch of token sequences by masking.
    
    Args:
        tokens: [batch, seq_len] clean token ids
        t: [batch] timesteps in [0, 1]
        mask_token_id: token id for [MASK]
        schedule: masking schedule type
        max_mask_ratio: maximum fraction of tokens to mask
        pad_token_id: don't mask padding tokens
    
    Returns:
        corrupted: [batch, seq_len] tokens with some masked
        mask: [batch, seq_len] bool tensor of which positions are masked
    """
    batch_size, seq_len = tokens.shape
    device = tokens.device
    
    # Get mask probability for each sequence
    mask_prob = get_mask_schedule(t, schedule)
    mask_prob = torch.clamp(mask_prob, max=max_mask_ratio)
    
    # Sample which positions to mask
    rand = torch.rand(batch_size, seq_len, device=device)
    mask = rand < mask_prob.unsqueeze(1)
    
    # Don't mask padding tokens
    if pad_token_id is not None:
        mask = mask & (tokens != pad_token_id)
    
    # Apply masking
    corrupted = torch.where(mask, mask_token_id, tokens)
    
    return corrupted, mask


def compute_loss(
    model,
    tokens: torch.Tensor,
    mask_token_id: int,
    schedule: str = "linear",
    max_mask_ratio: float = 0.95,
    eos_token_id: int = None,
    pad_token_id: int = None,
    attention_mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute diffusion training loss.
    """
    batch_size, seq_len = tokens.shape
    device = tokens.device
    
    # Sample random timesteps
    t = torch.rand(batch_size, device=device)
    
    # Corrupt
    corrupted, mask = corrupt_batch(
        tokens, t, mask_token_id, schedule, max_mask_ratio, pad_token_id
    )
    
    # Forward pass
    logits = model(corrupted, t, attention_mask=attention_mask)
    
    # Flatten for loss computation
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = tokens.view(-1)
    mask_flat = mask.view(-1)
    
    # Build valid mask: masked positions, excluding special tokens
    valid_mask = mask_flat.clone()
    
    if eos_token_id is not None:
        valid_mask = valid_mask & (targets_flat != eos_token_id)
    
    if pad_token_id is not None:
        valid_mask = valid_mask & (targets_flat != pad_token_id)
    
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    loss = F.cross_entropy(
        logits_flat[valid_mask],
        targets_flat[valid_mask]
    )
    
    return loss


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
    eos_token_id: Optional[int] = None,
    block_eos_until_step: Optional[int] = None,
) -> torch.Tensor:
    """
    Generate sequences using iterative demasking.
    
    Args:
        model: DiffusionLLM model
        seq_len: length of sequence to generate
        mask_token_id: [MASK] token id
        batch_size: number of sequences to generate
        steps: number of denoising steps
        temperature: sampling temperature
        top_k: if set, only sample from top k tokens
        top_p: if set, nucleus sampling threshold
        schedule: masking schedule
        device: cuda or cpu
        prompt_tokens: optional prefix tokens to condition on
        prompt_mask: which positions in prompt are fixed (True = don't change)
        eos_token_id: if set, block this token in early steps
        block_eos_until_step: block EOS until this step (default: steps // 2)
    
    Returns:
        [batch_size, seq_len] generated token ids
    """
    model.eval()
    
    if block_eos_until_step is None:
        block_eos_until_step = steps // 2
    
    # Initialize with all masks (or prompt if provided)
    if prompt_tokens is not None:
        x = prompt_tokens.clone()
        if prompt_mask is None:
            prompt_mask = torch.ones_like(prompt_tokens, dtype=torch.bool)
        x = torch.where(prompt_mask, prompt_tokens, mask_token_id)
    else:
        x = torch.full((batch_size, seq_len), mask_token_id, device=device)
        prompt_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
    
    # Denoising loop
    timesteps = torch.linspace(1, 0, steps + 1, device=device)[:-1]
    
    for step_idx, t in enumerate(timesteps):
        t_batch = torch.full((batch_size,), t.item(), device=device)
        
        # Get model predictions
        logits = model(x, t_batch)
        
        # Block EOS in early steps if configured
        if eos_token_id is not None and step_idx < block_eos_until_step:
            logits[..., eos_token_id] = float('-inf')
        
        # Apply temperature
        logits = logits / temperature
        
        # Optional: top-k filtering
        if top_k is not None:
            top_k_vals, _ = logits.topk(top_k, dim=-1)
            threshold = top_k_vals[..., -1:]
            logits = torch.where(logits < threshold, float('-inf'), logits)
        
        # Optional: top-p (nucleus) filtering  
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            
            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            logits = torch.where(indices_to_remove, float('-inf'), logits)
        
        probs = F.softmax(logits, dim=-1)
        
        # Sample tokens
        sampled = torch.multinomial(
            probs.view(-1, probs.size(-1)),
            num_samples=1
        ).view(batch_size, seq_len)
        
        # Only update masked positions (not prompt)
        is_masked = (x == mask_token_id) & ~prompt_mask
        
        if not is_masked.any():
            break
        
        # Get confidence scores for masked positions
        confidence = probs.max(dim=-1).values
        confidence = torch.where(is_masked, confidence, torch.tensor(-1.0, device=device))
        
        # Determine how many to unmask this step
        n_masked = is_masked.sum(dim=-1).float()
        remaining_steps = max(1, steps - step_idx)
        n_to_unmask = (n_masked / remaining_steps).clamp(min=1)
        
        # Unmask highest confidence positions
        for b in range(batch_size):
            n_unmask = int(n_to_unmask[b].item())
            masked_idx = is_masked[b].nonzero().squeeze(-1)
            
            if len(masked_idx) > 0:
                conf = confidence[b, masked_idx]
                k = min(n_unmask, len(masked_idx))
                top_indices = conf.topk(k).indices
                unmask_pos = masked_idx[top_indices]
                x[b, unmask_pos] = sampled[b, unmask_pos]
    
    # Final pass: unmask any remaining
    remaining_mask = (x == mask_token_id) & ~prompt_mask
    if remaining_mask.any():
        t_final = torch.zeros(batch_size, device=device)
        logits = model(x, t_final)
        final_preds = logits.argmax(dim=-1)
        x = torch.where(remaining_mask, final_preds, x)
    
    return x
