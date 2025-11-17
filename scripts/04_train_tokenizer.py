"""
Train SentencePiece tokenizer for Punjabi-English translation.

This script:
1. Loads training data (source and target)
2. Trains SentencePiece BPE tokenizer
3. Supports both shared and separate vocabularies
4. Saves tokenizer models
"""

import sys
import sentencepiece as spm
from pathlib import Path
from typing import List, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import setup_logger
from utils.config import load_config


class TokenizerTrainer:
    """Train SentencePiece tokenizer for translation."""

    def __init__(self, config_path: str = "configs/data_config.yaml"):
        self.config = load_config(config_path)
        self.logger = setup_logger('tokenizer_training', self.config['paths']['logs_dir'])

        self.splits_dir = Path(self.config['paths']['splits_dir'])
        self.tokenizer_dir = Path(self.config['paths']['tokenizer_dir'])
        self.tokenizer_dir.mkdir(parents=True, exist_ok=True)

        # Tokenizer config
        self.tokenizer_config = self.config['tokenizer']

    def prepare_training_data(self, language: str) -> Path:
        """
        Prepare training data for tokenizer.

        Args:
            language: 'pa' for Punjabi or 'en' for English

        Returns:
            Path to training data file
        """
        # Use training split for tokenizer training
        train_file = self.splits_dir / f"train.{language}"

        if not train_file.exists():
            raise FileNotFoundError(f"Training data not found: {train_file}")

        self.logger.info(f"Using training data: {train_file}")
        return train_file

    def prepare_combined_training_data(self) -> Path:
        """
        Combine source and target data for shared vocabulary.

        Returns:
            Path to combined training data
        """
        pa_file = self.splits_dir / "train.pa"
        en_file = self.splits_dir / "train.en"

        if not pa_file.exists() or not en_file.exists():
            raise FileNotFoundError("Training data files not found")

        # Combine both files
        combined_file = self.tokenizer_dir / "combined_train.txt"

        self.logger.info("Combining source and target data for shared vocabulary")

        with open(combined_file, 'w', encoding='utf-8') as out_f:
            # Add Punjabi data
            with open(pa_file, 'r', encoding='utf-8') as pa_f:
                for line in pa_f:
                    out_f.write(line)

            # Add English data
            with open(en_file, 'r', encoding='utf-8') as en_f:
                for line in en_f:
                    out_f.write(line)

        self.logger.info(f"Combined training data saved to: {combined_file}")
        return combined_file

    def train_sentencepiece(self,
                           input_file: Path,
                           model_prefix: str,
                           vocab_size: Optional[int] = None) -> bool:
        """
        Train SentencePiece tokenizer.

        Args:
            input_file: Path to training data
            model_prefix: Prefix for output model files
            vocab_size: Vocabulary size (overrides config if provided)

        Returns:
            True if successful
        """
        vocab_size = vocab_size or self.tokenizer_config['vocab_size']
        model_type = self.tokenizer_config['model_type']
        char_coverage = self.tokenizer_config['character_coverage']

        # Special tokens
        pad_token = self.tokenizer_config['pad_token']
        unk_token = self.tokenizer_config['unk_token']
        bos_token = self.tokenizer_config['bos_token']
        eos_token = self.tokenizer_config['eos_token']

        # Additional parameters
        normalization_rule = self.tokenizer_config.get('normalization_rule_name', 'nfkc')
        split_by_whitespace = self.tokenizer_config.get('split_by_whitespace', True)
        split_by_unicode = self.tokenizer_config.get('split_by_unicode_script', True)
        byte_fallback = self.tokenizer_config.get('byte_fallback', True)

        # Build training command
        train_args = {
            'input': str(input_file),
            'model_prefix': str(self.tokenizer_dir / model_prefix),
            'vocab_size': vocab_size,
            'model_type': model_type,
            'character_coverage': char_coverage,
            'pad_id': 0,
            'unk_id': 1,
            'bos_id': 2,
            'eos_id': 3,
            'pad_piece': pad_token,
            'unk_piece': unk_token,
            'bos_piece': bos_token,
            'eos_piece': eos_token,
            'normalization_rule_name': normalization_rule,
            'split_by_whitespace': split_by_whitespace,
            'split_by_unicode_script': split_by_unicode,
            'byte_fallback': byte_fallback,
            'train_extremely_large_corpus': False,
            'num_threads': 16,
        }

        self.logger.info(f"Training SentencePiece tokenizer: {model_prefix}")
        self.logger.info(f"  Vocabulary size: {vocab_size}")
        self.logger.info(f"  Model type: {model_type}")
        self.logger.info(f"  Character coverage: {char_coverage}")

        try:
            spm.SentencePieceTrainer.train(**train_args)
            self.logger.info(f"Tokenizer trained successfully: {model_prefix}")
            return True

        except Exception as e:
            self.logger.error(f"Error training tokenizer: {e}")
            return False

    def test_tokenizer(self, model_path: Path, test_sentences: List[str]):
        """
        Test the trained tokenizer on sample sentences.

        Args:
            model_path: Path to tokenizer model
            test_sentences: List of test sentences
        """
        self.logger.info(f"Testing tokenizer: {model_path}")

        sp = spm.SentencePieceProcessor()
        sp.load(str(model_path))

        print(f"\n{'='*60}")
        print(f"Tokenizer Test: {model_path.name}")
        print(f"{'='*60}")
        print(f"Vocabulary size: {sp.vocab_size()}")
        print(f"BOS ID: {sp.bos_id()}, EOS ID: {sp.eos_id()}")
        print(f"PAD ID: {sp.pad_id()}, UNK ID: {sp.unk_id()}")
        print(f"\nSample tokenizations:")

        for sentence in test_sentences:
            tokens = sp.encode(sentence, out_type=str)
            ids = sp.encode(sentence, out_type=int)
            reconstructed = sp.decode(ids)

            print(f"\nOriginal: {sentence}")
            print(f"Tokens: {tokens}")
            print(f"IDs: {ids[:20]}{'...' if len(ids) > 20 else ''}")
            print(f"Reconstructed: {reconstructed}")

        print(f"{'='*60}\n")

    def train_shared_vocabulary(self) -> bool:
        """Train a shared vocabulary for both languages."""
        self.logger.info("Training shared vocabulary tokenizer")

        # Combine training data
        combined_file = self.prepare_combined_training_data()

        # Train tokenizer
        success = self.train_sentencepiece(
            input_file=combined_file,
            model_prefix="tokenizer_shared"
        )

        if success:
            model_path = self.tokenizer_dir / "tokenizer_shared.model"

            # Test with sample sentences
            test_sentences = [
                "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ",  # Punjabi
                "Hello, how are you?",  # English
                "ਮੈਂ ਠੀਕ ਹਾਂ, ਤੁਹਾਡਾ ਧੰਨਵਾਦ।",  # Punjabi
            ]
            self.test_tokenizer(model_path, test_sentences)

        return success

    def train_separate_vocabularies(self) -> bool:
        """Train separate vocabularies for source and target languages."""
        self.logger.info("Training separate vocabulary tokenizers")

        # Each gets half the vocab size
        half_vocab = self.tokenizer_config['vocab_size'] // 2

        # Train Punjabi tokenizer
        pa_file = self.prepare_training_data('pa')
        pa_success = self.train_sentencepiece(
            input_file=pa_file,
            model_prefix="tokenizer_pa",
            vocab_size=half_vocab
        )

        # Train English tokenizer
        en_file = self.prepare_training_data('en')
        en_success = self.train_sentencepiece(
            input_file=en_file,
            model_prefix="tokenizer_en",
            vocab_size=half_vocab
        )

        if pa_success and en_success:
            # Test both tokenizers
            pa_model = self.tokenizer_dir / "tokenizer_pa.model"
            en_model = self.tokenizer_dir / "tokenizer_en.model"

            self.test_tokenizer(pa_model, [
                "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ",
                "ਮੈਂ ਠੀਕ ਹਾਂ, ਤੁਹਾਡਾ ਧੰਨਵਾਦ।",
                "ਪੰਜਾਬੀ ਇੱਕ ਸੁੰਦਰ ਭਾਸ਼ਾ ਹੈ।"
            ])

            self.test_tokenizer(en_model, [
                "Hello, how are you?",
                "I am fine, thank you.",
                "Machine translation is fascinating."
            ])

        return pa_success and en_success

    def run(self) -> bool:
        """Run the complete tokenizer training process."""
        self.logger.info("Starting tokenizer training process")

        shared_vocab = self.tokenizer_config.get('shared_vocab', True)

        if shared_vocab:
            success = self.train_shared_vocabulary()
        else:
            success = self.train_separate_vocabularies()

        if success:
            self.logger.info("Tokenizer training complete")
            self.logger.info(f"Tokenizer models saved to: {self.tokenizer_dir}")
        else:
            self.logger.error("Tokenizer training failed")

        return success


def main():
    """Main entry point."""
    trainer = TokenizerTrainer()
    success = trainer.run()

    if success:
        print("\n✓ Tokenizer training completed successfully")
        print(f"  Models saved to: {trainer.tokenizer_dir}")
        return 0
    else:
        print("\n✗ Tokenizer training encountered errors")
        return 1


if __name__ == "__main__":
    sys.exit(main())
