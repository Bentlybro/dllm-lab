# DLLM Lab

Experiments with Diffusion Language Models. Train your own, run it locally, see what happens.

## What Is This?

A minimal but complete setup for training masked diffusion LLMs (like MDLM/LLaDA) from scratch using HuggingFace datasets.

## Quick Start

```bash
# Install deps
pip install -r requirements.txt

# Train a tiny model on wikitext (good for testing)
python src/train.py --config configs/tiny.yaml

# Train a small model on openwebtext
python src/train.py --config configs/small.yaml

# Generate text
python scripts/generate.py --checkpoint checkpoints/latest.pt --prompt "The quick brown"
```

## Project Structure

```
dllm-lab/
├── src/
│   ├── model.py        # Transformer architecture
│   ├── diffusion.py    # Masked diffusion process
│   ├── train.py        # Training loop
│   └── data.py         # HuggingFace dataset loading
├── scripts/
│   ├── generate.py     # Text generation
│   └── benchmark.py    # Speed benchmarks
├── configs/
│   ├── tiny.yaml       # ~25M params, for testing
│   └── small.yaml      # ~125M params, real experiments
└── checkpoints/        # Saved models
```

## How It Works

1. **Corruption**: Randomly mask tokens based on timestep t ∈ [0,1]
2. **Training**: Model learns to predict original tokens from masked input
3. **Sampling**: Start all-masked, iteratively unmask highest-confidence predictions

## Hardware

- **Tiny (25M)**: Any GPU, even CPU for testing
- **Small (125M)**: Single consumer GPU (3090/4090)
- **Medium (350M)**: 24GB+ VRAM or multi-GPU

## Datasets

Uses HuggingFace datasets. Configured in YAML:

```yaml
data:
  dataset: "openwebtext"  # or "wikitext", "c4", "the_pile"
  subset: null            # e.g., "wikitext-103-raw-v1"
  max_length: 512
```

## Key Differences from GPT-style

| Aspect | Autoregressive (GPT) | Diffusion (This) |
|--------|---------------------|------------------|
| Attention | Causal (left-to-right) | Bidirectional |
| Generation | Sequential tokens | Parallel refinement |
| Training | Next-token prediction | Denoising masked tokens |
| Inference | O(n) forward passes | O(steps) passes, steps << n |
