"""
Continuous Diffusion Process (Mercury-style)

- Cosine noise schedule
- ε-prediction (predict noise)
- MSE loss
- Classifier-free guidance
- Few-step inference (4-10 steps)
"""

import math
import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class ContinuousDiffusionConfig:
    num_train_timesteps: int = 1000
    num_inference_steps: int = 8
    schedule: str = "cosine"
    schedule_offset: float = 0.008
    cfg_dropout: float = 0.1  # probability of dropping conditioning during training


def cosine_alpha_cumprod(t: torch.Tensor, T: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine schedule for α_cumprod (cumulative product of 1-β).
    
    Returns α̅_t where higher t = more noise.
    """
    f = (t.float() / T) + s
    theta = f * math.pi / 2.0
    alpha_bar = (torch.cos(theta) / math.cos(s * math.pi / 2.0)) ** 2
    return torch.clamp(alpha_bar, min=1e-5, max=1.0)


def get_schedule(num_steps: int, device: str = "cuda", s: float = 0.008):
    """
    Pre-compute the noise schedule.
    
    Returns:
        alphas_cumprod: [num_steps+1] cumulative product of (1-β)
        betas: [num_steps] noise variance at each step
    """
    timesteps = torch.arange(num_steps + 1, device=device)
    alphas_cumprod = cosine_alpha_cumprod(timesteps, num_steps, s)
    
    # Compute betas from alphas_cumprod
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    betas = 1 - (alphas_cumprod / alphas_cumprod_prev)
    betas = torch.clamp(betas, min=1e-5, max=0.999)
    
    return alphas_cumprod, betas


def add_noise(
    x_0: torch.Tensor,
    noise: torch.Tensor,
    t: torch.Tensor,
    alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    """
    Forward diffusion: add noise to clean embeddings.
    
    x_t = sqrt(α̅_t) * x_0 + sqrt(1 - α̅_t) * ε
    """
    alpha_t = alphas_cumprod[t]
    
    # Reshape for broadcasting: [batch, 1, 1]
    while alpha_t.dim() < x_0.dim():
        alpha_t = alpha_t.unsqueeze(-1)
    
    sqrt_alpha = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha = torch.sqrt(1 - alpha_t)
    
    x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
    return x_t


def compute_loss(
    model,
    token_ids: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    num_timesteps: int,
    cfg_dropout: float = 0.1,
    attention_mask: torch.Tensor = None,
    pad_token_id: int = None,
) -> torch.Tensor:
    """
    Compute ε-prediction MSE loss.
    
    1. Get clean embeddings from tokens
    2. Sample random timesteps
    3. Add noise
    4. Predict noise
    5. MSE loss
    """
    batch_size, seq_len = token_ids.shape
    device = token_ids.device
    
    # Get clean embeddings
    x_0 = model.get_embeddings(token_ids)  # [B, L, d]
    
    # Sample random timesteps (1 to T, not 0)
    t = torch.randint(1, num_timesteps + 1, (batch_size,), device=device)
    
    # Sample noise
    noise = torch.randn_like(x_0)
    
    # Add noise to get x_t
    x_t = add_noise(x_0, noise, t, alphas_cumprod)
    
    # Classifier-free guidance: randomly drop conditioning
    # For unconditional training, we just don't pass any conditioning
    # The model learns to denoise without extra context
    cond_emb = None  # Could add prompt conditioning here
    
    # Predict noise
    noise_pred = model(x_t, t, cond_emb=cond_emb, attention_mask=attention_mask)
    
    # MSE loss on noise
    # Optionally mask out padding positions
    if pad_token_id is not None and attention_mask is not None:
        # Expand mask to embedding dimension
        mask = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
        loss = ((noise_pred - noise) ** 2 * mask).sum() / mask.sum() / noise.shape[-1]
    else:
        loss = F.mse_loss(noise_pred, noise)
    
    return loss


@torch.no_grad()
def sample(
    model,
    seq_len: int,
    batch_size: int = 1,
    num_inference_steps: int = 8,
    num_train_timesteps: int = 1000,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    guidance_scale: float = 1.0,
    device: str = "cuda",
    schedule_offset: float = 0.008,
    cond_emb: Optional[torch.Tensor] = None,
    uncond_emb: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reverse diffusion: generate text from noise.
    
    Uses DDPM-style updates with optional classifier-free guidance.
    
    Returns:
        token_ids: [batch_size, seq_len] generated tokens
        logits: [batch_size, seq_len, vocab_size] final logits
    """
    model.eval()
    
    # Pre-compute schedule
    alphas_cumprod, betas = get_schedule(num_train_timesteps, device, schedule_offset)
    
    # Compute inference timesteps (evenly spaced)
    step_ratio = num_train_timesteps // num_inference_steps
    timesteps = torch.arange(num_inference_steps, device=device) * step_ratio
    timesteps = timesteps.flip(0) + 1  # [T, T-step, T-2*step, ..., step]
    
    # Start from pure noise
    x_t = torch.randn(batch_size, seq_len, model.d_model, device=device)
    
    for i, t in enumerate(timesteps):
        t_batch = torch.full((batch_size,), t.item(), device=device, dtype=torch.long)
        
        # Predict noise
        if guidance_scale > 1.0 and uncond_emb is not None:
            # Classifier-free guidance: blend conditional and unconditional
            noise_cond = model(x_t, t_batch, cond_emb=cond_emb)
            noise_uncond = model(x_t, t_batch, cond_emb=uncond_emb)
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        else:
            noise_pred = model(x_t, t_batch, cond_emb=cond_emb)
        
        # Get schedule values
        alpha_t = alphas_cumprod[t]
        alpha_t_prev = alphas_cumprod[t - step_ratio] if t > step_ratio else alphas_cumprod[0]
        beta_t = 1 - alpha_t / alpha_t_prev
        
        # DDPM update: predict x_0, then compute x_{t-1}
        # x_0 = (x_t - sqrt(1-α_t) * ε) / sqrt(α_t)
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
        
        x_0_pred = (x_t - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
        
        # Compute x_{t-1}
        sqrt_alpha_t_prev = torch.sqrt(alpha_t_prev)
        sqrt_one_minus_alpha_t_prev = torch.sqrt(1 - alpha_t_prev)
        
        # Direction pointing to x_t
        pred_dir = sqrt_one_minus_alpha_t_prev * noise_pred
        
        # x_{t-1} = sqrt(α_{t-1}) * x_0_pred + direction
        x_t = sqrt_alpha_t_prev * x_0_pred + pred_dir
        
        # Add noise for all steps except the last
        if i < len(timesteps) - 1:
            noise = torch.randn_like(x_t)
            # Variance for this step
            sigma = torch.sqrt(beta_t)
            x_t = x_t + sigma * noise * 0.5  # Reduced noise for stability
    
    # Final x_0 prediction
    x_0 = x_t
    
    # Decode to logits
    logits = model.decode_embeddings(x_0, temperature=temperature)
    
    # Apply top-k filtering
    if top_k is not None:
        top_k_vals, _ = logits.topk(top_k, dim=-1)
        threshold = top_k_vals[..., -1:]
        logits = torch.where(logits < threshold, float('-inf'), logits)
    
    # Apply top-p (nucleus) filtering
    if top_p is not None:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        logits = torch.where(indices_to_remove, float('-inf'), logits)
    
    # Sample tokens
    probs = F.softmax(logits, dim=-1)
    token_ids = torch.multinomial(
        probs.view(-1, probs.size(-1)),
        num_samples=1
    ).view(batch_size, seq_len)
    
    return token_ids, logits


@torch.no_grad()
def sample_with_prompt(
    model,
    tokenizer,
    prompt: str,
    max_length: int = 128,
    num_inference_steps: int = 8,
    num_train_timesteps: int = 1000,
    temperature: float = 0.8,
    top_k: Optional[int] = 50,
    guidance_scale: float = 2.0,
    device: str = "cuda",
) -> str:
    """
    Generate text with a prompt using classifier-free guidance.
    """
    model.eval()
    
    # Tokenize prompt
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = prompt_ids.shape[1]
    
    # Get prompt embedding (used as conditioning)
    prompt_emb = model.get_embeddings(prompt_ids)  # [1, prompt_len, d]
    cond_emb = prompt_emb.mean(dim=1)  # [1, d] - average over sequence
    
    # Null conditioning for CFG
    uncond_emb = torch.zeros_like(cond_emb)
    
    # Generate
    token_ids, _ = sample(
        model=model,
        seq_len=max_length,
        batch_size=1,
        num_inference_steps=num_inference_steps,
        num_train_timesteps=num_train_timesteps,
        temperature=temperature,
        top_k=top_k,
        guidance_scale=guidance_scale,
        device=device,
        cond_emb=cond_emb,
        uncond_emb=uncond_emb,
    )
    
    # Decode
    text = tokenizer.decode(token_ids[0], skip_special_tokens=True)
    return text
