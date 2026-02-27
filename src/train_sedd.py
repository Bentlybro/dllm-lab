"""
SEDD Training Script

Train a diffusion LLM using Score Entropy Discrete Diffusion.
"""

import os
import sys
import yaml
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Silence noisy HuggingFace/httpx logs BEFORE importing transformers
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# Enable TF32 for Ampere+ GPUs (20-30% speedup)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
from transformers import GPT2TokenizerFast

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import DiffusionLLM, create_model
from src.diffusion_sedd import SEDDDiffusion, SEDDLossWithEntropy
from src.data import create_dataloader


def setup_logging(output_dir: str):
    """Configure logging."""
    os.makedirs(output_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(output_dir, "train.log")),
        ],
    )
    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    step: int,
    loss: float,
    config: dict,
    output_dir: str,
    name: str = "latest",
):
    """Save training checkpoint."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict(),
        "step": step,
        "loss": loss,
        "config": config,
    }
    path = os.path.join(output_dir, f"{name}.pt")
    torch.save(checkpoint, path)


def train(config_path: str, resume: str = None):
    """Main training loop."""
    
    # Load config
    config = load_config(config_path)
    
    # Setup
    output_dir = config.get("output_dir", "checkpoints/sedd")
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(output_dir)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    # Add special tokens if needed
    special_tokens = {}
    if tokenizer.pad_token is None:
        special_tokens["pad_token"] = "[PAD]"
    if tokenizer.mask_token is None:
        special_tokens["mask_token"] = "[MASK]"
    
    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)
    
    vocab_size = len(tokenizer)
    mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    
    logger.info(f"Vocab size: {vocab_size}")
    logger.info(f"Mask token: {mask_token_id}, Pad token: {pad_token_id}, EOS: {eos_token_id}")
    
    # Update config
    config["model"]["vocab_size"] = vocab_size
    config["model"]["mask_token_id"] = mask_token_id
    config["model"]["pad_token_id"] = pad_token_id
    
    # Create model
    model = create_model(config["model"]).to(device)
    logger.info(f"Model parameters: {count_parameters(model):,}")
    
    # NOTE: torch.compile is called AFTER loading checkpoint (see below)
    # This avoids _orig_mod. prefix mismatch when resuming
    should_compile = config["training"].get("compile", True) and hasattr(torch, "compile")
    
    # Create SEDD diffusion
    diffusion = SEDDDiffusion(
        mask_token_id=mask_token_id,
        vocab_size=vocab_size,
        schedule=config.get("diffusion", {}).get("schedule", "log_linear"),
    )
    
    # Loss with optional entropy regularization
    entropy_weight = config.get("diffusion", {}).get("entropy_weight", 0.0)
    if entropy_weight > 0:
        loss_fn = SEDDLossWithEntropy(diffusion, entropy_weight=entropy_weight)
        use_loss_fn = True
    else:
        use_loss_fn = False
    
    # Dataloader
    data_config = config["data"].copy()
    data_config["batch_size"] = config["training"]["batch_size"]
    dataloader = create_dataloader(
        config=data_config,
        tokenizer=tokenizer,
        split=config["data"].get("split", "train"),
    )
    
    logger.info(f"Dataset: {config['data']['dataset']}")
    logger.info(f"Batch size: {config['training']['batch_size']}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"].get("weight_decay", 0.01),
        betas=tuple(config["training"].get("betas", [0.9, 0.98])),
    )
    
    # Scheduler (linear warmup + cosine decay)
    total_steps = config["training"]["max_steps"]
    warmup_steps = config["training"].get("warmup_steps", 1000)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    import math
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Mixed precision
    scaler = GradScaler('cuda', enabled=config["training"].get("mixed_precision", True))
    
    # Resume if specified (BEFORE compiling to avoid key mismatch)
    start_step = 0
    if resume:
        logger.info(f"Resuming from {resume}")
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt["scheduler_state_dict"]:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_step = ckpt["step"]
        logger.info(f"Resumed at step {start_step}")
    
    # Compile model AFTER loading checkpoint (avoids _orig_mod. prefix issues)
    if should_compile:
        logger.info("Compiling model with torch.compile()...")
        model = torch.compile(model, mode="reduce-overhead")
        logger.info("Model compiled!")
    
    # Training config
    grad_accum = config["training"].get("gradient_accumulation_steps", 1)
    log_every = config["training"].get("log_every", 100)
    save_every = config["training"].get("save_every", 5000)
    
    # Check if training is already done
    if start_step >= total_steps:
        logger.info(f"Training already complete (step {start_step} >= {total_steps}). Nothing to do.")
        return
    
    # Training loop
    model.train()
    step = start_step
    epoch = 0
    running_loss = 0.0
    running_metrics = {}
    nan_count = 0
    max_nan_streak = 10  # halt if 10 NaNs in a row
    last_loss = 0.0  # Track last loss for final save
    
    # Stats tracking
    import time
    tokens_per_batch = config["training"]["batch_size"] * config["data"]["max_length"]
    start_time = time.time()
    tokens_seen = 0
    
    pbar = tqdm(total=total_steps, initial=start_step, desc=f"Epoch {epoch+1}")
    
    while step < total_steps:
        epoch += 1
        pbar.set_description(f"Epoch {epoch}")
        
        for batch in dataloader:
            if step >= total_steps:
                break
            
            tokens = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            # Forward pass with mixed precision
            with autocast('cuda', enabled=config["training"].get("mixed_precision", True)):
                if use_loss_fn:
                    loss, metrics = loss_fn(
                        model, tokens,
                        attention_mask=attention_mask,
                        pad_token_id=pad_token_id,
                    )
                    for k, v in metrics.items():
                        running_metrics[k] = running_metrics.get(k, 0) + v
                else:
                    loss = diffusion.compute_loss(
                        model, tokens,
                        attention_mask=attention_mask,
                        pad_token_id=pad_token_id,
                    )
                
                loss = loss / grad_accum
            
            # NaN detection - skip batch and reset gradients
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                # Debug info for first few NaNs
                if nan_count <= 3:
                    non_pad = (tokens != pad_token_id).sum().item() if pad_token_id else tokens.numel()
                    logger.warning(f"Step {step} | NaN/Inf loss! non_pad_tokens={non_pad}, batch_shape={tokens.shape}, streak={nan_count}")
                else:
                    logger.warning(f"Step {step} | NaN/Inf loss detected! Skipping batch. (streak: {nan_count})")
                optimizer.zero_grad()
                # Don't call scaler.update() here - we haven't scaled anything
                
                if nan_count >= max_nan_streak:
                    logger.error(f"Hit {max_nan_streak} NaN losses in a row. Saving emergency checkpoint and halting.")
                    save_checkpoint(
                        model, optimizer, scheduler, scaler,
                        step, 0.0, config,
                        output_dir, f"emergency_step_{step}"
                    )
                    raise RuntimeError(f"Training halted: {max_nan_streak} consecutive NaN losses")
                
                continue  # skip this batch entirely
            
            nan_count = 0  # reset streak on good batch
            last_loss = loss.item() * grad_accum  # track for final save
            
            # Backward
            scaler.scale(loss).backward()
            running_loss += loss.item()
            
            # Optimizer step
            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config["training"].get("max_grad_norm", 1.0)
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
            step += 1
            tokens_seen += tokens_per_batch
            elapsed = time.time() - start_time
            tokens_per_sec = tokens_seen / elapsed if elapsed > 0 else 0
            
            pbar.update(1)
            pbar.set_postfix({
                "loss": f"{running_loss / max(1, step - start_step):.4f}",
                "tok/s": f"{tokens_per_sec/1000:.1f}k",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })
            
            # Logging
            if step % log_every == 0:
                avg_loss = running_loss * grad_accum / log_every
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                tps = tokens_seen / elapsed if elapsed > 0 else 0
                
                log_msg = f"Step {step} | Loss: {avg_loss:.4f} | {tps/1000:.1f}k tok/s | LR: {lr:.2e}"
                if running_metrics:
                    for k, v in running_metrics.items():
                        log_msg += f" | {k}: {v/log_every:.4f}"
                
                logger.info(log_msg)
                running_loss = 0.0
                running_metrics = {}
            
            # Save checkpoint
            if step % save_every == 0:
                save_checkpoint(
                    model, optimizer, scheduler, scaler,
                    step, loss.item() * grad_accum, config,
                    output_dir, f"step_{step}"
                )
                save_checkpoint(
                    model, optimizer, scheduler, scaler,
                    step, loss.item() * grad_accum, config,
                    output_dir, "latest"
                )
                logger.info(f"Saved checkpoint to {output_dir}/step_{step}.pt")
                # Clear CUDA cache after checkpoint to prevent fragmentation
                if device == "cuda":
                    torch.cuda.empty_cache()
            
            # Periodic cache clearing every 1000 steps
            if step % 1000 == 0 and device == "cuda":
                torch.cuda.empty_cache()
    
    pbar.close()
    
    # Final save
    save_checkpoint(
        model, optimizer, scheduler, scaler,
        step, last_loss, config,
        output_dir, "final"
    )
    logger.info(f"Training complete! Final model saved to {output_dir}/final.pt (step {step}, loss {last_loss:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()
    
    train(args.config, args.resume)
