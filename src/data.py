"""
Data loading utilities using HuggingFace datasets.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """Tokenized text dataset from HuggingFace."""
    
    def __init__(
        self,
        dataset_name: str,
        tokenizer,
        max_length: int = 512,
        split: str = "train",
        subset: Optional[str] = None,
        text_column: str = "text",
        max_samples: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            dataset_name: HuggingFace dataset name (e.g., "openwebtext", "wikitext")
            tokenizer: HuggingFace tokenizer
            max_length: maximum sequence length
            split: dataset split (train/validation/test)
            subset: dataset subset/config (e.g., "wikitext-103-raw-v1")
            text_column: name of text column in dataset
            max_samples: limit number of samples (for quick testing)
            cache_dir: cache directory for datasets
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        logger.info(f"Loading dataset: {dataset_name} (subset={subset}, split={split})")
        
        # Load dataset
        if subset:
            self.dataset = load_dataset(
                dataset_name, 
                subset, 
                split=split,
                cache_dir=cache_dir,
            )
        else:
            self.dataset = load_dataset(
                dataset_name, 
                split=split,
                cache_dir=cache_dir,
            )
        
        # Limit samples if specified
        if max_samples and len(self.dataset) > max_samples:
            self.dataset = self.dataset.select(range(max_samples))
        
        self.text_column = text_column
        
        # Handle different dataset formats
        if text_column not in self.dataset.column_names:
            # Try common alternatives
            for alt in ["content", "document", "passage", "sentence"]:
                if alt in self.dataset.column_names:
                    self.text_column = alt
                    break
            else:
                raise ValueError(
                    f"Text column '{text_column}' not found. "
                    f"Available: {self.dataset.column_names}"
                )
        
        logger.info(f"Loaded {len(self.dataset)} samples")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.dataset[idx][self.text_column]
        
        # Handle empty/None text
        if not text:
            text = " "
        
        # Tokenize with truncation and padding
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }


def create_dataloader(
    config: Dict[str, Any],
    tokenizer,
    split: str = "train",
) -> DataLoader:
    """Create a DataLoader from config."""
    
    dataset = TextDataset(
        dataset_name=config.get("dataset", "wikitext"),
        tokenizer=tokenizer,
        max_length=config.get("max_length", 512),
        split=split,
        subset=config.get("subset", None),
        text_column=config.get("text_column", "text"),
        max_samples=config.get("max_samples", None),
        cache_dir=config.get("cache_dir", None),
    )
    
    return DataLoader(
        dataset,
        batch_size=config.get("batch_size", 8),
        shuffle=(split == "train"),
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )


def get_tokenizer(name: str = "gpt2", add_mask_token: bool = True):
    """
    Load tokenizer and optionally add a [MASK] token.
    
    Returns tokenizer and mask_token_id.
    """
    tokenizer = AutoTokenizer.from_pretrained(name)
    
    # GPT-2 doesn't have a pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Add mask token if not present
    if add_mask_token and "[MASK]" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})
    
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        # Fallback: use vocab_size as mask token id (will resize embedding)
        mask_token_id = len(tokenizer)
    
    return tokenizer, mask_token_id


# Common dataset configs
DATASET_CONFIGS = {
    "wikitext-small": {
        "dataset": "wikitext",
        "subset": "wikitext-2-raw-v1",
        "text_column": "text",
    },
    "wikitext-large": {
        "dataset": "wikitext",
        "subset": "wikitext-103-raw-v1",
        "text_column": "text",
    },
    "openwebtext": {
        "dataset": "openwebtext",
        "subset": None,
        "text_column": "text",
    },
    "c4": {
        "dataset": "c4",
        "subset": "en",
        "text_column": "text",
    },
    "pile-small": {
        "dataset": "monology/pile-uncopyrighted",
        "subset": None,
        "text_column": "text",
    },
    "tiny-shakespeare": {
        "dataset": "tiny_shakespeare",
        "subset": None,
        "text_column": "text",
    },
    "code-python": {
        "dataset": "codeparrot/github-code",
        "subset": "Python-all",
        "text_column": "code",
    },
}
