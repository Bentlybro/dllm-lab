#!/usr/bin/env python3
"""
Benchmark inference speed for DLLM.

Compares generation speed with different step counts.
"""

import sys
sys.path.insert(0, "src")

import argparse
import time
import torch
from model import DiffusionLLM
from diffusion import sample
from data import get_tokenizer


def benchmark(checkpoint_path: str, device: str = "cuda"):
    """Run inference benchmarks."""
    
    # Load model
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]
    
    tokenizer_name = config.get("tokenizer", "gpt2")
    tokenizer, mask_token_id = get_tokenizer(tokenizer_name)
    
    model_config = config.get("model", {})
    model_config["vocab_size"] = len(tokenizer)
    model_config["mask_token_id"] = mask_token_id
    model_config["max_seq_len"] = config.get("data", {}).get("max_length", 512)
    
    model = DiffusionLLM(**model_config)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters ({n_params/1e6:.1f}M)")
    print(f"Device: {device}")
    print("-" * 60)
    
    # Benchmark configs
    lengths = [64, 128, 256]
    step_counts = [10, 25, 50, 100]
    
    print(f"{'Length':<10} {'Steps':<10} {'Time (s)':<12} {'Tokens/s':<12}")
    print("-" * 60)
    
    for length in lengths:
        for steps in step_counts:
            # Warmup
            if device == "cuda":
                torch.cuda.synchronize()
            
            with torch.no_grad():
                _ = sample(
                    model=model,
                    seq_len=length,
                    mask_token_id=mask_token_id,
                    batch_size=1,
                    steps=steps,
                    device=device,
                )
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            # Timed runs
            n_runs = 5
            start = time.perf_counter()
            
            for _ in range(n_runs):
                with torch.no_grad():
                    _ = sample(
                        model=model,
                        seq_len=length,
                        mask_token_id=mask_token_id,
                        batch_size=1,
                        steps=steps,
                        device=device,
                    )
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            elapsed = (time.perf_counter() - start) / n_runs
            tokens_per_sec = length / elapsed
            
            print(f"{length:<10} {steps:<10} {elapsed:<12.4f} {tokens_per_sec:<12.1f}")
    
    print("-" * 60)
    print("\nNote: Tokens/s is for full sequence generation, not per-token like autoregressive models.")
    print("Lower step counts = faster but potentially lower quality.")


def main():
    parser = argparse.ArgumentParser(description="Benchmark DLLM inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()
    
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    
    benchmark(args.checkpoint, args.device)


if __name__ == "__main__":
    main()
