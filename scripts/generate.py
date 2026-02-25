#!/usr/bin/env python3
"""
Generate text from a trained DLLM checkpoint.

Usage:
    python scripts/generate.py --checkpoint checkpoints/latest.pt --prompt "The quick brown"
    python scripts/generate.py --checkpoint checkpoints/latest.pt --length 128 --steps 100
    python scripts/generate.py --checkpoint checkpoints/latest.pt --infill "The [MASK] fox jumps"
"""

import sys
sys.path.insert(0, "src")

import argparse
import torch
from model import DiffusionLLM
from diffusion import sample
from data import get_tokenizer


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]
    
    # Get tokenizer to match vocab size
    tokenizer_name = config.get("tokenizer", "gpt2")
    tokenizer, mask_token_id = get_tokenizer(tokenizer_name)
    
    # Build model
    model_config = config.get("model", {})
    model_config["vocab_size"] = len(tokenizer)
    model_config["mask_token_id"] = mask_token_id
    model_config["max_seq_len"] = config.get("data", {}).get("max_length", 512)
    
    model = DiffusionLLM(**model_config)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    return model, tokenizer, mask_token_id, config


def generate_unconditional(
    model,
    tokenizer,
    mask_token_id: int,
    length: int = 128,
    steps: int = 50,
    batch_size: int = 1,
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None,
    device: str = "cuda",
):
    """Generate text from scratch."""
    
    tokens = sample(
        model=model,
        seq_len=length,
        mask_token_id=mask_token_id,
        batch_size=batch_size,
        steps=steps,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        device=device,
    )
    
    texts = []
    for i in range(batch_size):
        text = tokenizer.decode(tokens[i], skip_special_tokens=True)
        texts.append(text)
    
    return texts


def generate_with_prompt(
    model,
    tokenizer,
    mask_token_id: int,
    prompt: str,
    length: int = 128,
    steps: int = 50,
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None,
    device: str = "cuda",
):
    """Generate text conditioned on a prompt."""
    
    # Tokenize prompt
    prompt_tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = prompt_tokens.shape[1]
    
    # Pad/extend to full length
    if prompt_len >= length:
        print(f"Warning: prompt ({prompt_len} tokens) >= length ({length}), truncating")
        prompt_tokens = prompt_tokens[:, :length]
        prompt_len = length
    
    # Create full sequence with prompt + masks
    full_tokens = torch.full((1, length), mask_token_id, device=device)
    full_tokens[:, :prompt_len] = prompt_tokens
    
    # Mark prompt positions as fixed
    prompt_mask = torch.zeros((1, length), dtype=torch.bool, device=device)
    prompt_mask[:, :prompt_len] = True
    
    tokens = sample(
        model=model,
        seq_len=length,
        mask_token_id=mask_token_id,
        batch_size=1,
        steps=steps,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        device=device,
        prompt_tokens=full_tokens,
        prompt_mask=prompt_mask,
    )
    
    return tokenizer.decode(tokens[0], skip_special_tokens=True)


def generate_infill(
    model,
    tokenizer,
    mask_token_id: int,
    text_with_masks: str,
    steps: int = 50,
    temperature: float = 1.0,
    device: str = "cuda",
):
    """
    Fill in [MASK] tokens in the input text.
    
    Example: "The [MASK] fox [MASK] over the lazy dog"
    """
    
    # Replace [MASK] with actual mask token
    mask_str = "[MASK]"
    
    # Tokenize without masks first to get structure
    parts = text_with_masks.split(mask_str)
    
    # Build token sequence with mask tokens inserted
    tokens = []
    for i, part in enumerate(parts):
        if part:
            tokens.extend(tokenizer.encode(part, add_special_tokens=False))
        if i < len(parts) - 1:  # Don't add mask after last part
            tokens.append(mask_token_id)
    
    tokens = torch.tensor([tokens], device=device)
    seq_len = tokens.shape[1]
    
    # Mark which positions are masks
    prompt_mask = (tokens != mask_token_id)
    
    result = sample(
        model=model,
        seq_len=seq_len,
        mask_token_id=mask_token_id,
        batch_size=1,
        steps=steps,
        temperature=temperature,
        device=device,
        prompt_tokens=tokens,
        prompt_mask=prompt_mask,
    )
    
    return tokenizer.decode(result[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Generate text from DLLM")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt text")
    parser.add_argument("--infill", type=str, default=None, help="Text with [MASK] tokens to fill")
    parser.add_argument("--length", type=int, default=128, help="Sequence length")
    parser.add_argument("--steps", type=int, default=50, help="Denoising steps")
    parser.add_argument("--batch", type=int, default=1, help="Batch size (unconditional only)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=None, help="Nucleus sampling threshold")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
    
    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    print(f"Loading model from {args.checkpoint}...")
    model, tokenizer, mask_token_id, config = load_model(args.checkpoint, args.device)
    print(f"Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    print(f"Generating with {args.steps} steps, temperature={args.temperature}...")
    print("-" * 50)
    
    if args.infill:
        # Infilling mode
        text = generate_infill(
            model, tokenizer, mask_token_id,
            args.infill, args.steps, args.temperature, args.device
        )
        print(f"Input:  {args.infill}")
        print(f"Output: {text}")
        
    elif args.prompt:
        # Prompted generation
        text = generate_with_prompt(
            model, tokenizer, mask_token_id,
            args.prompt, args.length, args.steps,
            args.temperature, args.top_k, args.top_p, args.device
        )
        print(text)
        
    else:
        # Unconditional generation
        texts = generate_unconditional(
            model, tokenizer, mask_token_id,
            args.length, args.steps, args.batch,
            args.temperature, args.top_k, args.top_p, args.device
        )
        for i, text in enumerate(texts):
            if args.batch > 1:
                print(f"--- Sample {i+1} ---")
            print(text)
            if args.batch > 1:
                print()


if __name__ == "__main__":
    main()
