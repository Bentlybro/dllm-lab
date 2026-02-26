"""
SEDD - Score Entropy Discrete Diffusion

Based on "Discrete Diffusion Language Modeling by Estimating the Ratios 
of the Data Distribution" (Lou et al., 2023)

Key differences from simple masked diffusion:
1. Continuous-time absorbing state diffusion
2. Score-based parameterization (log ratios)
3. Proper score matching loss with importance weighting
4. Euler sampling with score-based reverse process
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class SEDDDiffusion:
    """
    Score Entropy Discrete Diffusion for absorbing state (mask) diffusion.
    
    The forward process: tokens transition to [MASK] with rate σ(t)
    q(xt | x0) = (1 - αt) * δ(xt, x0) + αt * δ(xt, [MASK])
    
    where αt = 1 - exp(-∫σ(s)ds) is the probability of being absorbed
    """
    
    def __init__(
        self,
        mask_token_id: int,
        vocab_size: int,
        schedule: str = "log_linear",  # SEDD uses log-linear
        eps: float = 1e-3,  # small offset to avoid numerical issues
    ):
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.schedule = schedule
        self.eps = eps
    
    def alpha_schedule(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute α(t) = probability of being absorbed (masked) at time t.
        t=0 → α=0 (clean)
        t=1 → α≈1 (all masked)
        """
        t = torch.clamp(t, self.eps, 1 - self.eps)
        
        if self.schedule == "log_linear":
            # SEDD's log-linear schedule: α(t) = t
            # Simple but effective
            return t
        elif self.schedule == "cosine":
            # Cosine schedule (smoother)
            return 1 - torch.cos(t * math.pi / 2)
        elif self.schedule == "geometric":
            # Geometric schedule: faster absorption early
            return 1 - (1 - t) ** 2
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")
    
    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the transition rate σ(t).
        For absorbing diffusion: σ(t) = d(log(1-α(t)))/dt
        
        With log-linear α(t) = t:
        σ(t) = 1 / (1 - t)
        """
        t = torch.clamp(t, self.eps, 1 - self.eps)
        
        if self.schedule == "log_linear":
            return 1.0 / (1 - t)
        elif self.schedule == "cosine":
            alpha = self.alpha_schedule(t)
            dalpha_dt = math.pi / 2 * torch.sin(t * math.pi / 2)
            return dalpha_dt / (1 - alpha + self.eps)
        elif self.schedule == "geometric":
            return 2 * (1 - t) / ((1 - t) ** 2 + self.eps)
        else:
            return 1.0 / (1 - t)  # default
    
    def corrupt(
        self,
        x0: torch.Tensor,  # [batch, seq_len]
        t: torch.Tensor,   # [batch]
        pad_mask: Optional[torch.Tensor] = None,  # [batch, seq_len] True = pad
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample xt from q(xt | x0).
        
        Returns:
            xt: corrupted tokens
            is_masked: boolean mask of which positions are [MASK]
        """
        batch_size, seq_len = x0.shape
        device = x0.device
        
        # Get absorption probability
        alpha = self.alpha_schedule(t)  # [batch]
        
        # Sample which positions get absorbed
        rand = torch.rand(batch_size, seq_len, device=device)
        is_masked = rand < alpha.unsqueeze(1)
        
        # Don't mask padding
        if pad_mask is not None:
            is_masked = is_masked & ~pad_mask
        
        # Apply corruption
        xt = torch.where(is_masked, self.mask_token_id, x0)
        
        return xt, is_masked
    
    def compute_loss(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pad_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute SEDD score matching loss.
        
        The loss is: E_t,x0,xt [w(t) * loss(s_θ(xt,t), x0)]
        
        For absorbing state diffusion, this simplifies to cross-entropy
        on masked positions with importance weighting by σ(t).
        """
        batch_size, seq_len = x0.shape
        device = x0.device
        
        # Sample timesteps (avoid t=0 and t=1)
        t = torch.rand(batch_size, device=device) * (1 - 2 * self.eps) + self.eps
        
        # Build pad mask
        pad_mask = None
        if pad_token_id is not None:
            pad_mask = (x0 == pad_token_id)
        
        # Corrupt
        xt, is_masked = self.corrupt(x0, t, pad_mask)
        
        # Get model predictions (scores / logits)
        logits = model(xt, t, attention_mask=attention_mask)  # [batch, seq, vocab]
        
        # Compute importance weight: σ(t)
        # Higher weight for larger t (when more is masked and prediction is harder)
        weight = self.sigma(t)  # [batch]
        
        # Clamp weights to avoid explosion
        weight = torch.clamp(weight, max=10.0)
        
        # Flatten for loss
        logits_flat = logits.view(-1, self.vocab_size)  # [batch*seq, vocab]
        targets_flat = x0.view(-1)  # [batch*seq]
        is_masked_flat = is_masked.view(-1)  # [batch*seq]
        
        # Only compute loss on masked positions
        if not is_masked_flat.any():
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Expand weight to match flattened shape
        weight_expanded = weight.unsqueeze(1).expand(-1, seq_len).reshape(-1)
        
        # Cross-entropy per token (no reduction)
        ce_loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            reduction='none'
        )  # [batch*seq]
        
        # Apply mask and weights
        masked_loss = ce_loss * is_masked_flat.float() * weight_expanded
        
        # Normalize by number of masked tokens
        loss = masked_loss.sum() / (is_masked_flat.sum() + 1e-8)
        
        return loss
    
    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        seq_len: int,
        batch_size: int = 1,
        steps: int = 64,
        temperature: float = 1.0,
        top_p: Optional[float] = 0.9,
        device: str = "cuda",
        prompt_tokens: Optional[torch.Tensor] = None,
        prompt_len: int = 0,
    ) -> torch.Tensor:
        """
        Sample using Euler integration of the reverse process.
        
        The reverse process unmarks tokens with probability proportional
        to the model's confidence, following the score.
        """
        model.eval()
        
        # Initialize with all masks
        if prompt_tokens is not None:
            x = torch.full((batch_size, seq_len), self.mask_token_id, device=device)
            x[:, :prompt_len] = prompt_tokens[:, :prompt_len]
            prompt_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
            prompt_mask[:, :prompt_len] = True
        else:
            x = torch.full((batch_size, seq_len), self.mask_token_id, device=device)
            prompt_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
        
        # Time steps from t=1 to t=0
        dt = 1.0 / steps
        timesteps = torch.linspace(1 - self.eps, self.eps, steps, device=device)
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t.item(), device=device)
            
            # Get scores (logits)
            logits = model(x, t_batch)  # [batch, seq, vocab]
            
            # Apply temperature
            logits = logits / temperature
            
            # Top-p sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                logits = logits.masked_fill(indices_to_remove, float('-inf'))
            
            probs = F.softmax(logits, dim=-1)  # [batch, seq, vocab]
            
            # Find currently masked positions (not prompt)
            is_masked = (x == self.mask_token_id) & ~prompt_mask
            
            if not is_masked.any():
                break
            
            # Sample new tokens for all positions
            sampled = torch.multinomial(
                probs.view(-1, self.vocab_size),
                num_samples=1
            ).view(batch_size, seq_len)
            
            # Compute unmasking probability based on time
            # As t decreases, we unmask more tokens
            # Expected number to unmask this step: (1/steps) * num_currently_masked
            n_masked_per_seq = is_masked.sum(dim=1).float()  # [batch]
            
            # Probability to unmask each masked token this step
            p_unmask = dt / (t.item() + self.eps)  # Rate-based unmasking
            p_unmask = min(p_unmask, 1.0)
            
            # Get confidence for each position (probability of most likely token)
            confidence = probs.max(dim=-1).values  # [batch, seq]
            
            # Unmask tokens with probability proportional to confidence
            unmask_prob = confidence * p_unmask
            unmask_prob = torch.where(is_masked, unmask_prob, torch.zeros_like(unmask_prob))
            
            # Stochastic unmasking
            do_unmask = torch.rand_like(unmask_prob) < unmask_prob
            
            # Alternatively: deterministic top-k unmasking per step
            # This is often more stable
            for b in range(batch_size):
                masked_positions = is_masked[b].nonzero().squeeze(-1)
                if len(masked_positions) == 0:
                    continue
                
                # Number to unmask this step
                n_to_unmask = max(1, int(len(masked_positions) * p_unmask))
                n_to_unmask = min(n_to_unmask, len(masked_positions))
                
                # Select highest confidence positions
                conf_masked = confidence[b, masked_positions]
                _, top_idx = conf_masked.topk(n_to_unmask)
                positions_to_unmask = masked_positions[top_idx]
                
                # Unmask
                x[b, positions_to_unmask] = sampled[b, positions_to_unmask]
        
        # Final pass: unmask any remaining with argmax
        remaining_mask = (x == self.mask_token_id) & ~prompt_mask
        if remaining_mask.any():
            t_final = torch.full((batch_size,), self.eps, device=device)
            logits = model(x, t_final)
            final_preds = logits.argmax(dim=-1)
            x = torch.where(remaining_mask, final_preds, x)
        
        return x


class SEDDLossWithEntropy(nn.Module):
    """
    Full SEDD loss with entropy regularization.
    
    L = score_matching_loss + λ * entropy_loss
    
    The entropy term encourages diversity in the model's predictions.
    """
    
    def __init__(
        self,
        diffusion: SEDDDiffusion,
        entropy_weight: float = 0.01,
    ):
        super().__init__()
        self.diffusion = diffusion
        self.entropy_weight = entropy_weight
    
    def forward(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pad_token_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute loss and return metrics.
        """
        batch_size, seq_len = x0.shape
        device = x0.device
        
        # Sample timesteps
        t = torch.rand(batch_size, device=device) * (1 - 2 * self.diffusion.eps) + self.diffusion.eps
        
        # Build pad mask
        pad_mask = None
        if pad_token_id is not None:
            pad_mask = (x0 == pad_token_id)
        
        # Corrupt
        xt, is_masked = self.diffusion.corrupt(x0, t, pad_mask)
        
        # Get logits
        logits = model(xt, t, attention_mask=attention_mask)
        
        # Score matching loss (cross-entropy on masked positions)
        weight = torch.clamp(self.diffusion.sigma(t), max=10.0)
        
        logits_flat = logits.view(-1, self.diffusion.vocab_size)
        targets_flat = x0.view(-1)
        is_masked_flat = is_masked.view(-1)
        weight_expanded = weight.unsqueeze(1).expand(-1, seq_len).reshape(-1)
        
        ce_loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
        masked_loss = ce_loss * is_masked_flat.float() * weight_expanded
        score_loss = masked_loss.sum() / (is_masked_flat.sum() + 1e-8)
        
        # Entropy regularization (on masked positions only)
        probs = F.softmax(logits_flat[is_masked_flat], dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        
        # Total loss
        total_loss = score_loss - self.entropy_weight * entropy
        
        metrics = {
            'score_loss': score_loss.item(),
            'entropy': entropy.item(),
            'n_masked': is_masked.sum().item(),
            'avg_t': t.mean().item(),
        }
        
        return total_loss, metrics
