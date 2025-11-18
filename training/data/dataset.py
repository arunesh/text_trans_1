"""
Dataset loader for translation training.

Loads preprocessed parallel data from Phase 1 and prepares batches for training.
"""

import json
import gzip
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm


class TranslationDataset(Dataset):
    """PyTorch Dataset for parallel translation data."""

    def __init__(
        self,
        data_file: str,
        tokenizer_source: spm.SentencePieceProcessor,
        tokenizer_target: Optional[spm.SentencePieceProcessor] = None,
        max_source_length: int = 128,
        max_target_length: int = 128,
        source_lang: str = "pa",
        target_lang: str = "en"
    ):
        """
        Initialize translation dataset.

        Args:
            data_file: Path to data file (JSON or JSON.gz)
            tokenizer_source: Source language tokenizer
            tokenizer_target: Target language tokenizer (None if shared vocab)
            max_source_length: Maximum source sequence length
            max_target_length: Maximum target sequence length
            source_lang: Source language code
            target_lang: Target language code
        """
        self.data_file = Path(data_file)
        self.tokenizer_source = tokenizer_source
        self.tokenizer_target = tokenizer_target or tokenizer_source
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.source_lang = source_lang
        self.target_lang = target_lang

        # Load data
        self.examples = self._load_data()

    def _load_data(self) -> List[Dict[str, str]]:
        """Load parallel data from file."""
        examples = []

        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_file}")

        # Determine file type and load
        if self.data_file.suffix == '.gz':
            with gzip.open(self.data_file, 'rt', encoding='utf-8') as f:
                for line in f:
                    examples.append(json.loads(line))
        else:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    examples.append(json.loads(line))

        return examples

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single example.

        Returns:
            Dictionary with input_ids, attention_mask, labels
        """
        example = self.examples[idx]
        source_text = example['source']
        target_text = example['target']

        # Tokenize source
        source_ids = self.tokenizer_source.encode(
            source_text,
            out_type=int,
            add_bos=True,
            add_eos=True
        )

        # Tokenize target
        target_ids = self.tokenizer_target.encode(
            target_text,
            out_type=int,
            add_bos=True,
            add_eos=True
        )

        # Truncate if needed
        source_ids = source_ids[:self.max_source_length]
        target_ids = target_ids[:self.max_target_length]

        # Convert to tensors
        source_ids = torch.tensor(source_ids, dtype=torch.long)
        target_ids = torch.tensor(target_ids, dtype=torch.long)

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.ones_like(source_ids)

        return {
            'input_ids': source_ids,
            'attention_mask': attention_mask,
            'labels': target_ids
        }


def collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    pad_token_id: int = 0
) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching with padding.

    Args:
        batch: List of examples from dataset
        pad_token_id: Token ID to use for padding

    Returns:
        Batched and padded tensors
    """
    # Extract sequences
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]

    # Find max lengths in batch
    max_input_len = max(len(ids) for ids in input_ids)
    max_label_len = max(len(ids) for ids in labels)

    # Pad sequences
    padded_input_ids = []
    padded_attention_masks = []
    padded_labels = []

    for ids, mask, label in zip(input_ids, attention_masks, labels):
        # Pad input
        input_padding = max_input_len - len(ids)
        padded_input_ids.append(
            torch.cat([ids, torch.full((input_padding,), pad_token_id, dtype=torch.long)])
        )
        padded_attention_masks.append(
            torch.cat([mask, torch.zeros(input_padding, dtype=torch.long)])
        )

        # Pad labels (use -100 for padding tokens to ignore in loss)
        label_padding = max_label_len - len(label)
        padded_labels.append(
            torch.cat([label, torch.full((label_padding,), -100, dtype=torch.long)])
        )

    # Stack into batches
    return {
        'input_ids': torch.stack(padded_input_ids),
        'attention_mask': torch.stack(padded_attention_masks),
        'labels': torch.stack(padded_labels)
    }


def load_tokenizer(tokenizer_path: str) -> spm.SentencePieceProcessor:
    """
    Load SentencePiece tokenizer.

    Args:
        tokenizer_path: Path to .model file

    Returns:
        Loaded tokenizer
    """
    sp = spm.SentencePieceProcessor()
    sp.load(str(tokenizer_path))
    return sp


def create_dataloaders(
    train_file: str,
    val_file: str,
    tokenizer_source_path: str,
    tokenizer_target_path: Optional[str] = None,
    batch_size: int = 32,
    max_source_length: int = 128,
    max_target_length: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
    source_lang: str = "pa",
    target_lang: str = "en"
) -> Tuple[DataLoader, DataLoader, spm.SentencePieceProcessor]:
    """
    Create train and validation dataloaders.

    Args:
        train_file: Path to training data
        val_file: Path to validation data
        tokenizer_source_path: Path to source tokenizer
        tokenizer_target_path: Path to target tokenizer (None for shared)
        batch_size: Batch size
        max_source_length: Max source sequence length
        max_target_length: Max target sequence length
        num_workers: Number of dataloader workers
        pin_memory: Pin memory for faster GPU transfer
        source_lang: Source language code
        target_lang: Target language code

    Returns:
        Tuple of (train_loader, val_loader, tokenizer)
    """
    # Load tokenizers
    tokenizer_source = load_tokenizer(tokenizer_source_path)
    tokenizer_target = None
    if tokenizer_target_path:
        tokenizer_target = load_tokenizer(tokenizer_target_path)

    # Get pad token ID
    pad_token_id = tokenizer_source.pad_id()

    # Create datasets
    train_dataset = TranslationDataset(
        data_file=train_file,
        tokenizer_source=tokenizer_source,
        tokenizer_target=tokenizer_target,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        source_lang=source_lang,
        target_lang=target_lang
    )

    val_dataset = TranslationDataset(
        data_file=val_file,
        tokenizer_source=tokenizer_source,
        tokenizer_target=tokenizer_target,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        source_lang=source_lang,
        target_lang=target_lang
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=lambda batch: collate_fn(batch, pad_token_id)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=lambda batch: collate_fn(batch, pad_token_id)
    )

    return train_loader, val_loader, tokenizer_source


if __name__ == "__main__":
    # Test data loading
    import sys
    from pathlib import Path

    # Add parent directory to path
    sys.path.append(str(Path(__file__).parent.parent.parent))

    # Test with sample data
    tokenizer_path = "models/tokenizer/tokenizer_shared.model"
    train_file = "data/splits/train.json.gz"

    if Path(tokenizer_path).exists() and Path(train_file).exists():
        print("Testing data loading...")

        train_loader, val_loader, tokenizer = create_dataloaders(
            train_file=train_file,
            val_file="data/splits/val.json.gz",
            tokenizer_source_path=tokenizer_path,
            batch_size=4,
            num_workers=0
        )

        print(f"Train dataset size: {len(train_loader.dataset)}")
        print(f"Val dataset size: {len(val_loader.dataset)}")
        print(f"Vocabulary size: {tokenizer.vocab_size()}")

        # Load one batch
        batch = next(iter(train_loader))
        print(f"\nSample batch:")
        print(f"  Input shape: {batch['input_ids'].shape}")
        print(f"  Labels shape: {batch['labels'].shape}")
        print(f"  Attention mask shape: {batch['attention_mask'].shape}")

        # Decode sample
        print(f"\nSample translation:")
        print(f"  Source: {tokenizer.decode(batch['input_ids'][0].tolist())}")
        print(f"  Target: {tokenizer.decode([t for t in batch['labels'][0].tolist() if t != -100])}")
    else:
        print("Data or tokenizer not found. Run Phase 1 first.")
