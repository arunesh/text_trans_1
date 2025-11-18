"""
Train translation model (Phase 2).

This script trains a Punjabi-English or English-Punjabi translation model.
"""

import sys
import argparse
import random
from pathlib import Path
import numpy as np
import torch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import setup_logger
from utils.config import load_config
from training.data import create_dataloaders
from training.models import TranslationTransformer
from training.trainer import TranslationTrainer


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train translation model")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/train_config.yaml',
        help='Path to training configuration file'
    )
    parser.add_argument(
        '--direction',
        type=str,
        choices=['pa-en', 'en-pa'],
        default='pa-en',
        help='Translation direction'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Override output directory'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup logger
    logger = setup_logger('train', config['output']['logs_dir'])

    logger.info("="*60)
    logger.info("TRANSLATION MODEL TRAINING")
    logger.info("="*60)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Direction: {args.direction}")

    # Set random seed
    seed = config.get('hardware', {}).get('seed', 42)
    set_seed(seed)
    logger.info(f"Random seed: {seed}")

    # Determine translation direction
    if args.direction == 'pa-en':
        source_lang = 'pa'
        target_lang = 'en'
    else:
        source_lang = 'en'
        target_lang = 'pa'

    logger.info(f"Source language: {source_lang}")
    logger.info(f"Target language: {target_lang}")

    # Setup device
    device_name = config.get('hardware', {}).get('device', 'cuda')
    if device_name == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device_name = 'cpu'

    device = torch.device(device_name)
    logger.info(f"Device: {device}")

    if device_name == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Prepare data paths
    data_dir = Path(config['data']['data_dir'])
    tokenizer_dir = Path(config['data']['tokenizer_dir'])

    train_file = data_dir / "train.json.gz"
    val_file = data_dir / "val.json.gz"

    # Determine tokenizer paths
    shared_vocab = config['data'].get('shared_vocab', True)
    if shared_vocab:
        tokenizer_source_path = tokenizer_dir / "tokenizer_shared.model"
        tokenizer_target_path = None
    else:
        tokenizer_source_path = tokenizer_dir / f"tokenizer_{source_lang}.model"
        tokenizer_target_path = tokenizer_dir / f"tokenizer_{target_lang}.model"

    # Check files exist
    if not train_file.exists():
        raise FileNotFoundError(f"Training data not found: {train_file}")
    if not val_file.exists():
        raise FileNotFoundError(f"Validation data not found: {val_file}")
    if not tokenizer_source_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_source_path}")

    logger.info(f"Training data: {train_file}")
    logger.info(f"Validation data: {val_file}")
    logger.info(f"Tokenizer: {tokenizer_source_path}")

    # Create data loaders
    logger.info("Creating data loaders...")

    train_loader, val_loader, tokenizer = create_dataloaders(
        train_file=str(train_file),
        val_file=str(val_file),
        tokenizer_source_path=str(tokenizer_source_path),
        tokenizer_target_path=str(tokenizer_target_path) if tokenizer_target_path else None,
        batch_size=config['training']['batch_size'],
        max_source_length=config['model']['max_source_length'],
        max_target_length=config['model']['max_target_length'],
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=config['data'].get('pin_memory', True),
        source_lang=source_lang,
        target_lang=target_lang
    )

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    logger.info(f"Vocabulary size: {tokenizer.vocab_size()}")

    # Create model
    logger.info("Creating model...")

    model_config = config['model']
    model = TranslationTransformer(
        vocab_size=tokenizer.vocab_size(),
        d_model=model_config['hidden_size'],
        num_encoder_layers=model_config['encoder_layers'],
        num_decoder_layers=model_config['decoder_layers'],
        num_heads=model_config['num_attention_heads'],
        d_ff=model_config['intermediate_size'],
        dropout=model_config['dropout'],
        max_seq_length=model_config['max_position_embeddings'],
        pad_token_id=tokenizer.pad_id(),
        bos_token_id=tokenizer.bos_id(),
        eos_token_id=tokenizer.eos_id()
    )

    logger.info(f"Model architecture: {model_config['type']}")
    logger.info(f"Model parameters: {model.get_num_parameters():,}")
    logger.info(f"Model size (fp32): ~{model.get_num_parameters() * 4 / 1024 / 1024:.1f} MB")

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config['output']['output_dir']) / f"{args.direction}-{seed}"

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Create trainer
    logger.info("Creating trainer...")

    trainer = TranslationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        config=config,
        output_dir=str(output_dir),
        device=device
    )

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Start training
    logger.info("\nStarting training...")
    logger.info("="*60)

    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        trainer.save_checkpoint("interrupted")
    except Exception as e:
        logger.error(f"\nTraining failed with error: {e}", exc_info=True)
        raise

    logger.info("="*60)
    logger.info("Training completed!")
    logger.info(f"Best model saved in: {output_dir}")
    logger.info("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
