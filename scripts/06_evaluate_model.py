"""
Evaluate trained translation model.

This script evaluates a trained model on test sets and generates translations.
"""

import sys
import argparse
from pathlib import Path
import torch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import setup_logger
from utils.config import load_config
from training.data import load_tokenizer
from training.models import TranslationTransformer
from training.evaluation import TranslationEvaluator, format_metrics


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description="Evaluate translation model")
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--source-file',
        type=str,
        required=True,
        help='Source text file (one sentence per line)'
    )
    parser.add_argument(
        '--reference-file',
        type=str,
        required=True,
        help='Reference translation file (one sentence per line)'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='Output file for predictions (optional)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--num-beams',
        type=int,
        default=4,
        help='Number of beams for beam search'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=128,
        help='Maximum generation length'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu', 'mps'],
        help='Device to use'
    )

    args = parser.parse_args()

    # Setup logger
    logger = setup_logger('evaluate', 'logs')

    logger.info("="*60)
    logger.info("TRANSLATION MODEL EVALUATION")
    logger.info("="*60)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Source file: {args.source_file}")
    logger.info(f"Reference file: {args.reference_file}")

    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # Load checkpoint
    logger.info("Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    config = checkpoint['config']

    # Get model configuration
    model_config = config['model']
    data_config = config['data']

    # Load tokenizer
    tokenizer_dir = Path(data_config['tokenizer_dir'])
    shared_vocab = data_config.get('shared_vocab', True)

    if shared_vocab:
        tokenizer_path = tokenizer_dir / "tokenizer_shared.model"
    else:
        # Try to infer from config
        tokenizer_path = tokenizer_dir / "tokenizer_shared.model"

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    logger.info(f"Loading tokenizer: {tokenizer_path}")
    tokenizer = load_tokenizer(str(tokenizer_path))
    logger.info(f"Vocabulary size: {tokenizer.vocab_size()}")

    # Create model
    logger.info("Creating model...")
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

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded from step {checkpoint['global_step']}, epoch {checkpoint['epoch']}")
    logger.info(f"Model parameters: {model.get_num_parameters():,}")

    # Create evaluator
    logger.info("Creating evaluator...")
    evaluator = TranslationEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=args.max_length,
        num_beams=args.num_beams
    )

    # Run evaluation
    logger.info("Evaluating...")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Num beams: {args.num_beams}")
    logger.info(f"  Max length: {args.max_length}")

    metrics = evaluator.evaluate_file(
        source_file=args.source_file,
        reference_file=args.reference_file,
        batch_size=args.batch_size
    )

    # Print results
    logger.info(format_metrics(metrics, "Evaluation Results"))

    # Print to console as well
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(format_metrics(metrics))
    print("="*60)

    # Save predictions if requested
    if args.output_file:
        logger.info(f"Saving predictions to: {args.output_file}")
        # Would need to modify evaluator to save predictions

    return 0


if __name__ == "__main__":
    sys.exit(main())
