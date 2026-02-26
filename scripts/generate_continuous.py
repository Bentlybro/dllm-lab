#!/usr/bin/env python3
"""
Generate text from a Continuous Diffusion LLM checkpoint.

Usage:
    python scripts/generate_continuous.py --checkpoint checkpoints/continuous/latest.pt --length 64 --steps 8
    python scripts/generate_continuous.py --checkpoint checkpoints/continuous/latest.pt --prompt "The future of AI" --guidance 2.0
"""

import sys
sys.path.insert(0, "src")

import argparse
import torch
from model_continuous import ContinuousDiffusionLLM
from diffusion_continuous import sample, sample_with_prompt
from data import get_tokenizer


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]
    
    tokenizer_name = config.get("tokenizer", "gpt2")
    tokenizer, _ = get_tokenizer(tokenizer_name, add_mask_token=False)
    
    model_config = config.get("model", {})
    model_config["vocab_size"] = len(tokenizer)
    model_config["max_seq_len"] = config.get("data", {}).get("max_length", 512)
    model_config["pad_token_id"] = tokenizer.pad_token_id
    
    model = ContinuousDiffusionLLM(**model_config)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    diffusion_config = config.get("diffusion", {})
    
    return model, tokenizer, config, diffusion_config


def generate_unconditional(
    model,
    tokenizer,
    diffusion_config,
    length: int = 64,
    steps: int = None,
    batch_size: int = 1,
    temperature: float = 0.8,
    top_k: int = 50,
    guidance_scale: float = 1.0,
    device: str = "cuda",
):
    """Generate text from scratch (unconditional)."""
    
    if steps is None:
        steps = diffusion_config.get("num_inference_steps", 8)
    
    num_train_steps = diffusion_config.get("num_train_timesteps", 1000)
    schedule_offset = float(diffusion_config.get("schedule_offset", 0.008))
    
    token_ids, logits = sample(
        model=model,
        seq_len=length,
        batch_size=batch_size,
        num_inference_steps=steps,
        num_train_timesteps=num_train_steps,
        temperature=temperature,
        top_k=top_k,
        guidance_scale=guidance_scale,
        device=device,
        schedule_offset=schedule_offset,
    )
    
    texts = []
    for i in range(batch_size):
        text = tokenizer.decode(token_ids[i], skip_special_tokens=True)
        texts.append(text)
    
    return texts, token_ids


def generate_with_prompt(
    model,
    tokenizer,
    diffusion_config,
    prompt: str,
    length: int = 128,
    steps: int = None,
    temperature: float = 0.8,
    top_k: int = 50,
    guidance_scale: float = 2.0,
    device: str = "cuda",
):
    """Generate text conditioned on a prompt."""
    
    if steps is None:
        steps = diffusion_config.get("num_inference_steps", 8)
    
    num_train_steps = diffusion_config.get("num_train_timesteps", 1000)
    
    text = sample_with_prompt(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_length=length,
        num_inference_steps=steps,
        num_train_timesteps=num_train_steps,
        temperature=temperature,
        top_k=top_k,
        guidance_scale=guidance_scale,
        device=device,
    )
    
    return text


def main():
    parser = argparse.ArgumentParser(description="Generate from Continuous Diffusion LLM")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--steps", type=int, default=None, help="Inference steps (default: from config)")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=2.0, help="CFG guidance scale")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
    
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    
    print(f"Loading model from {args.checkpoint}...")
    model, tokenizer, config, diffusion_config = load_model(args.checkpoint, args.device)
    print(f"Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    steps = args.steps or diffusion_config.get("num_inference_steps", 8)
    print(f"Generating with {steps} diffusion steps, temperature={args.temperature}, guidance={args.guidance}...")
    print("-" * 50)
    
    if args.prompt:
        text = generate_with_prompt(
            model, tokenizer, diffusion_config,
            args.prompt, args.length, steps,
            args.temperature, args.top_k, args.guidance, args.device
        )
        print(f"Prompt: {args.prompt}")
        print(f"Output: {text}")
    else:
        texts, _ = generate_unconditional(
            model, tokenizer, diffusion_config,
            args.length, steps, args.batch,
            args.temperature, args.top_k, args.guidance, args.device
        )
        for i, text in enumerate(texts):
            if args.batch > 1:
                print(f"--- Sample {i+1} ---")
            print(text)
            if args.batch > 1:
                print()


if __name__ == "__main__":
    main()
