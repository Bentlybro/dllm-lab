"""
SEDD Generation Script

Generate text using a trained SEDD diffusion model.
"""

import sys
import argparse
from pathlib import Path

import torch
from transformers import GPT2TokenizerFast

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import DiffusionLLM
from src.diffusion_sedd import SEDDDiffusion


def main():
    parser = argparse.ArgumentParser(description="Generate text with SEDD")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="", help="Optional prompt to condition on")
    parser.add_argument("--length", type=int, default=64, help="Total sequence length")
    parser.add_argument("--steps", type=int, default=64, help="Number of denoising steps")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling threshold")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of samples")
    parser.add_argument("--schedule", type=str, default="log_linear", help="Diffusion schedule")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else "cpu"
    
    # Load checkpoint
    print(f"Loading model from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt["config"]
    
    # Load tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})
    
    vocab_size = len(tokenizer)
    mask_token_id = tokenizer.mask_token_id
    
    # Create model
    model_config = config["model"]
    model_config["vocab_size"] = vocab_size
    model_config["mask_token_id"] = mask_token_id
    
    model = DiffusionLLM(
        vocab_size=model_config["vocab_size"],
        d_model=model_config.get("d_model", 512),
        n_heads=model_config.get("n_heads", 8),
        n_layers=model_config.get("n_layers", 8),
        d_ff=model_config.get("d_ff", 2048),
        max_seq_len=model_config.get("max_seq_len", 256),
        dropout=0.0,  # No dropout at inference
        mask_token_id=mask_token_id,
    ).to(device)
    
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    print(f"Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create SEDD diffusion
    diffusion = SEDDDiffusion(
        mask_token_id=mask_token_id,
        vocab_size=vocab_size,
        schedule=args.schedule,
    )
    
    print(f"Generating with {args.steps} diffusion steps, temperature={args.temperature}, top_p={args.top_p}...")
    print("-" * 50)
    
    # Handle prompt
    prompt_tokens = None
    prompt_len = 0
    if args.prompt:
        prompt_tokens = tokenizer(args.prompt, return_tensors="pt")["input_ids"].to(device)
        prompt_len = prompt_tokens.shape[1]
        prompt_tokens = torch.nn.functional.pad(
            prompt_tokens, 
            (0, args.length - prompt_len), 
            value=mask_token_id
        )
        prompt_tokens = prompt_tokens.expand(args.batch_size, -1)
        print(f"Prompt: {args.prompt}")
    
    # Generate
    with torch.no_grad():
        generated = diffusion.sample(
            model=model,
            seq_len=args.length,
            batch_size=args.batch_size,
            steps=args.steps,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
            prompt_tokens=prompt_tokens,
            prompt_len=prompt_len,
        )
    
    # Decode and print
    for i in range(args.batch_size):
        text = tokenizer.decode(generated[i], skip_special_tokens=True)
        if args.batch_size > 1:
            print(f"\n[Sample {i+1}]")
        print(text)


if __name__ == "__main__":
    main()
