"""
Continuous Diffusion LLM (Mercury-style)

Works in embedding space with Gaussian noise, not discrete token masking.
Predicts noise (ε-prediction) like image diffusion models.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    """Timestep embeddings."""
    
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
    """Transformer block with bidirectional attention."""
    
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
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.ff(x))
        return x


class ContinuousDiffusionLLM(nn.Module):
    """
    Mercury-style Continuous Diffusion LLM.
    
    Key differences from masked diffusion:
    - Works in continuous embedding space
    - Adds Gaussian noise (not masking)
    - Predicts the noise ε (not tokens)
    - Uses learned projection head for decoding
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
        pad_token_id: int = None,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        
        # Token embedding (used for encoding input AND decoding output)
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        # Timestep embedding
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        
        # Transformer denoiser (predicts noise)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout_layer = nn.Dropout(dropout)
        
        # Output projection: predicts noise in embedding space
        # Shape: (B, L, d_model) -> (B, L, d_model)
        self.noise_proj = nn.Linear(d_model, d_model)
        
        # Decoding head: maps denoised embeddings to vocab logits
        # Shape: (B, L, d_model) -> (B, L, vocab_size)
        self.out_proj = nn.Linear(d_model, vocab_size)
        
        # Initialize
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def get_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Convert token IDs to embeddings."""
        batch, seq_len = token_ids.shape
        device = token_ids.device
        
        pos = torch.arange(seq_len, device=device)
        return self.token_emb(token_ids) + self.pos_emb(pos)
    
    def forward(
        self,
        x_t: torch.Tensor,          # [batch, seq_len, d_model] noisy embeddings
        t: torch.Tensor,            # [batch] timesteps (integer or float)
        cond_emb: torch.Tensor = None,  # [batch, d_model] optional conditioning
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:              # [batch, seq_len, d_model] predicted noise
        """
        Predict the noise ε that was added to create x_t.
        """
        batch, seq_len, d_model = x_t.shape
        device = x_t.device
        
        # Timestep embedding
        t_emb = self.time_emb(t.float())  # [batch, d_model]
        
        # Add timestep to all positions
        h = x_t + t_emb.unsqueeze(1)
        
        # Add conditioning if provided (for CFG)
        if cond_emb is not None:
            h = h + cond_emb.unsqueeze(1)
        
        h = self.dropout_layer(h)
        
        # Attention mask
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
        
        # Transformer blocks
        for block in self.blocks:
            h = block(h, key_padding_mask=key_padding_mask)
        
        h = self.norm(h)
        
        # Project to noise prediction
        noise_pred = self.noise_proj(h)
        
        return noise_pred
    
    def decode_embeddings(self, x_0: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Decode denoised embeddings to token logits.
        
        Args:
            x_0: [batch, seq_len, d_model] denoised embeddings
            temperature: sampling temperature
        
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        logits = self.out_proj(x_0)
        return logits / temperature


def create_continuous_model(config: dict) -> ContinuousDiffusionLLM:
    """Create model from config dict."""
    return ContinuousDiffusionLLM(
        vocab_size=config.get("vocab_size", 50257),
        d_model=config.get("d_model", 512),
        n_heads=config.get("n_heads", 8),
        n_layers=config.get("n_layers", 6),
        d_ff=config.get("d_ff", 2048),
        max_seq_len=config.get("max_seq_len", 512),
        dropout=config.get("dropout", 0.1),
        pad_token_id=config.get("pad_token_id", None),
    )
