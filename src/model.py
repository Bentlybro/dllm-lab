"""
Diffusion LLM Model

A bidirectional transformer that takes:
- Corrupted/masked token sequence
- Timestep t ∈ [0,1]

And predicts the original clean tokens.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    """Timestep embeddings (same as image diffusion)."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class TransformerBlock(nn.Module):
    """Standard transformer block with bidirectional attention."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention (bidirectional - no causal mask!)
        # key_padding_mask: True = ignore this position
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward
        x = self.norm2(x + self.ff(x))
        return x


class DiffusionLLM(nn.Module):
    """
    Masked Diffusion Language Model.
    
    Architecture:
    - Token embeddings + positional embeddings
    - Timestep conditioning via addition
    - Stack of bidirectional transformer blocks
    - Output projection to vocab logits
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        mask_token_id: int = None,
        pad_token_id: int = None,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.mask_token_id = mask_token_id if mask_token_id is not None else vocab_size - 1
        self.pad_token_id = pad_token_id
        
        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        # Timestep embedding
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        x: torch.Tensor,                    # [batch, seq_len] corrupted tokens
        t: torch.Tensor,                    # [batch] timesteps in [0, 1]
        attention_mask: torch.Tensor = None, # [batch, seq_len] 1=attend, 0=ignore
    ) -> torch.Tensor:                      # [batch, seq_len, vocab_size] logits
        batch, seq_len = x.shape
        device = x.device
        
        # Token + position embeddings
        pos = torch.arange(seq_len, device=device)
        h = self.token_emb(x) + self.pos_emb(pos)
        
        # Add timestep embedding (broadcast to all positions)
        t_emb = self.time_emb(t)  # [batch, d_model]
        h = h + t_emb.unsqueeze(1)
        
        h = self.dropout(h)
        
        # Create key_padding_mask for attention (True = ignore)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
        elif self.pad_token_id is not None:
            key_padding_mask = (x == self.pad_token_id)
        
        # Transformer blocks
        for block in self.blocks:
            h = block(h, key_padding_mask=key_padding_mask)
        
        h = self.norm(h)
        
        # Project to vocab
        logits = self.out_proj(h)
        
        return logits


def create_model(config: dict) -> DiffusionLLM:
    """Create model from config dict."""
    return DiffusionLLM(
        vocab_size=config.get("vocab_size", 50257),
        d_model=config.get("d_model", 512),
        n_heads=config.get("n_heads", 8),
        n_layers=config.get("n_layers", 6),
        d_ff=config.get("d_ff", 2048),
        max_seq_len=config.get("max_seq_len", 512),
        dropout=config.get("dropout", 0.1),
        mask_token_id=config.get("mask_token_id", None),
        pad_token_id=config.get("pad_token_id", None),
    )
