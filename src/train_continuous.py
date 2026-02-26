"""
Training script for Continuous Diffusion LLM (Mercury-style).

Usage:
    python src/train_continuous.py --config configs/continuous_small.yaml
"""

import os
import math
import yaml
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from model_continuous import create_continuous_model
from diffusion_continuous import compute_loss, get_schedule
from data import create_dataloader, get_tokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, optimizer, scheduler, scaler, step, loss, config, path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "step": step,
        "loss": loss,
        "config": config,
    }, path)
    logger.info(f"Saved checkpoint to {path}")


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler and ckpt.get("scaler_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt.get("step", 0), ckpt.get("loss", float("inf"))


def train(config: dict, resume_path: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    output_dir = Path(config.get("output_dir", "checkpoints/continuous"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    tokenizer_name = config.get("tokenizer", "gpt2")
    tokenizer, mask_token_id = get_tokenizer(tokenizer_name, add_mask_token=False)
    vocab_size = len(tokenizer)
    
    logger.info(f"Tokenizer: {tokenizer_name}, vocab_size: {vocab_size}")
    
    # Create model
    model_config = config.get("model", {})
    model_config["vocab_size"] = vocab_size
    model_config["max_seq_len"] = config.get("data", {}).get("max_length", 512)
    model_config["pad_token_id"] = tokenizer.pad_token_id
    
    model = create_continuous_model(model_config)
    model = model.to(device)
    
    n_params = count_parameters(model)
    logger.info(f"Model parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    
    # Create dataloader
    data_config = config.get("data", {})
    train_loader = create_dataloader(data_config, tokenizer, split="train")
    logger.info(f"Training samples: {len(train_loader.dataset):,}")
    
    # Diffusion config
    diffusion_config = config.get("diffusion", {})
    num_timesteps = diffusion_config.get("num_train_timesteps", 1000)
    schedule_offset = float(diffusion_config.get("schedule_offset", 0.008))
    cfg_dropout = float(diffusion_config.get("cfg_dropout", 0.1))
    
    # Pre-compute noise schedule
    alphas_cumprod, _ = get_schedule(num_timesteps, device, schedule_offset)
    
    # Optimizer
    train_config = config.get("training", {})
    base_lr = float(train_config.get("lr", 1e-4))
    weight_decay = float(train_config.get("weight_decay", 0.01))
    
    optimizer = AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.98),
    )
    
    # Scheduler
    total_steps = train_config.get("total_steps", 100000)
    warmup_steps = train_config.get("warmup_steps", 2000)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Mixed precision
    use_amp = train_config.get("use_amp", True) and device.type == "cuda"
    scaler = GradScaler() if use_amp else None
    
    # Resume
    start_step = 0
    if resume_path:
        start_step, _ = load_checkpoint(resume_path, model, optimizer, scheduler, scaler)
        logger.info(f"Resumed from step {start_step}")
    
    # Training settings
    grad_accum = train_config.get("gradient_accumulation_steps", 1)
    log_interval = train_config.get("log_interval", 100)
    save_interval = train_config.get("save_interval", 5000)
    
    logger.info("Starting training...")
    logger.info(f"  Total steps: {total_steps:,}")
    logger.info(f"  Batch size: {data_config.get('batch_size', 8)}")
    logger.info(f"  Gradient accumulation: {grad_accum}")
    logger.info(f"  Learning rate: {base_lr}")
    logger.info(f"  Diffusion timesteps: {num_timesteps}")
    logger.info(f"  Schedule: cosine (s={schedule_offset})")
    logger.info(f"  Mixed precision: {use_amp}")
    
    model.train()
    step = start_step
    running_loss = 0.0
    epoch = 0
    
    while step < total_steps:
        epoch += 1
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            tokens = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            if use_amp:
                with autocast():
                    loss = compute_loss(
                        model, tokens, alphas_cumprod, num_timesteps,
                        cfg_dropout=cfg_dropout,
                        attention_mask=attention_mask,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    loss = loss / grad_accum
                scaler.scale(loss).backward()
            else:
                loss = compute_loss(
                    model, tokens, alphas_cumprod, num_timesteps,
                    cfg_dropout=cfg_dropout,
                    attention_mask=attention_mask,
                    pad_token_id=tokenizer.pad_token_id,
                )
                loss = loss / grad_accum
                loss.backward()
            
            running_loss += loss.item()
            
            if (step + 1) % grad_accum == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                optimizer.zero_grad()
                scheduler.step()
            
            step += 1
            
            if step % log_interval == 0:
                avg_loss = running_loss / log_interval * grad_accum
                lr = optimizer.param_groups[0]["lr"]
                pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")
                logger.info(f"Step {step} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")
                running_loss = 0.0
            
            if step % save_interval == 0:
                save_checkpoint(
                    model, optimizer, scheduler, scaler,
                    step, loss.item() * grad_accum, config,
                    output_dir / f"step_{step}.pt"
                )
                save_checkpoint(
                    model, optimizer, scheduler, scaler,
                    step, loss.item() * grad_accum, config,
                    output_dir / "latest.pt"
                )
            
            if step >= total_steps:
                break
    
    # Final save
    save_checkpoint(
        model, optimizer, scheduler, scaler,
        step, loss.item() * grad_accum, config,
        output_dir / "final.pt"
    )
    logger.info("Training complete!")


def main():
    parser = argparse.ArgumentParser(description="Train Continuous Diffusion LLM")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    
    config = load_config(args.config)
    train(config, args.resume)


if __name__ == "__main__":
    main()
