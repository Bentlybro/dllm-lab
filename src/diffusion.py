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
    """Configuration for the diffusion process."""
    mask_token_id: int
    schedule: str = "linear"  # linear, cosine, sqrt
    

def get_mask_schedule(t: torch.Tensor, schedule: str = "linear") -> torch.Tensor:
    """
    Convert timestep t to masking probability.
    
    t=0 → mask_prob=0 (clean)
    t=1 → mask_prob=1 (all masked)
    """
    if schedule == "linear":
        return t
    elif schedule == "cosine":
        # Cosine schedule (smoother, often works better)
        return 1 - torch.cos(t * torch.pi / 2)
    elif schedule == "sqrt":
        # Square root (more aggressive early masking)
        return torch.sqrt(t)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


def corrupt_batch(
    tokens: torch.Tensor,
    t: torch.Tensor,
    mask_token_id: int,
    schedule: str = "linear",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Corrupt a batch of token sequences by masking.
    
    Args:
        tokens: [batch, seq_len] clean token ids
        t: [batch] timesteps in [0, 1]
        mask_token_id: token id for [MASK]
        schedule: masking schedule type
    
    Returns:
        corrupted: [batch, seq_len] tokens with some masked
        mask: [batch, seq_len] bool tensor of which positions are masked
    """
    batch_size, seq_len = tokens.shape
    device = tokens.device
    
    # Get mask probability for each sequence
    mask_prob = get_mask_schedule(t, schedule)  # [batch]
    
    # Sample which positions to mask
    rand = torch.rand(batch_size, seq_len, device=device)
    mask = rand < mask_prob.unsqueeze(1)
    
    # Apply masking
    corrupted = torch.where(mask, mask_token_id, tokens)
    
    return corrupted, mask


def compute_loss(
    model,
    tokens: torch.Tensor,
    mask_token_id: int,
    schedule: str = "linear",
    loss_on_unmasked: bool = False,
) -> torch.Tensor:
    """
    Compute diffusion training loss.
    
    1. Sample random timesteps
    2. Corrupt tokens
    3. Predict original tokens
    4. Cross-entropy loss (optionally only on masked positions)
    """
    batch_size, seq_len = tokens.shape
    device = tokens.device
    
    # Sample random timesteps
    t = torch.rand(batch_size, device=device)
    
    # Corrupt
    corrupted, mask = corrupt_batch(tokens, t, mask_token_id, schedule)
    
    # Forward pass
    logits = model(corrupted, t)  # [batch, seq_len, vocab]
    
    # Compute loss
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = tokens.view(-1)
    
    if loss_on_unmasked:
        # Loss on all positions
        loss = F.cross_entropy(logits_flat, targets_flat)
    else:
        # Loss only on masked positions (standard)
        mask_flat = mask.view(-1)
        if mask_flat.sum() == 0:
            # Edge case: no masks (t very close to 0)
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        loss = F.cross_entropy(
            logits_flat[mask_flat],
            targets_flat[mask_flat]
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
        schedule: masking schedule (for timestep to mask ratio)
        device: cuda or cpu
        prompt_tokens: optional prefix tokens to condition on
        prompt_mask: which positions in prompt are fixed (True = don't change)
    
    Returns:
        [batch_size, seq_len] generated token ids
    """
    model.eval()
    
    # Initialize with all masks (or prompt if provided)
    if prompt_tokens is not None:
        x = prompt_tokens.clone()
        # Mask positions where prompt_mask is False
        if prompt_mask is None:
            prompt_mask = torch.ones_like(prompt_tokens, dtype=torch.bool)
        x = torch.where(prompt_mask, prompt_tokens, mask_token_id)
    else:
        x = torch.full((batch_size, seq_len), mask_token_id, device=device)
        prompt_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
    
    # Denoising loop
    timesteps = torch.linspace(1, 0, steps + 1, device=device)[:-1]  # [1, ..., small]
    
    for step_idx, t in enumerate(timesteps):
        t_batch = torch.full((batch_size,), t.item(), device=device)
        
        # Get model predictions
        logits = model(x, t_batch)
        
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
            
            # Remove tokens with cumulative probability above threshold
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
        # More aggressive unmasking early, slower refinement later
        progress = 1 - t.item()
        n_masked = is_masked.sum(dim=-1).float()
        target_unmasked = (progress * seq_len)
        n_to_unmask = (target_unmasked - (seq_len - n_masked)).clamp(min=1)
        
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
