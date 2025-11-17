"""
Split cleaned parallel data into train/validation/test sets.

This script:
1. Loads cleaned parallel data
2. Splits data into train/val/test according to config
3. Optionally stratifies by length or domain
4. Saves splits in separate files
"""

import sys
import json
import gzip
import random
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import setup_logger
from utils.config import load_config


class DataSplitter:
    """Split parallel data into train/validation/test sets."""

    def __init__(self, config_path: str = "configs/data_config.yaml"):
        self.config = load_config(config_path)
        self.logger = setup_logger('data_splitting', self.config['paths']['logs_dir'])

        self.processed_data_dir = Path(self.config['paths']['processed_data_dir'])
        self.splits_dir = Path(self.config['paths']['splits_dir'])
        self.splits_dir.mkdir(parents=True, exist_ok=True)

        # Split ratios
        self.split_config = self.config['splits']
        self.train_ratio = self.split_config['train']
        self.val_ratio = self.split_config['validation']
        self.test_ratio = self.split_config['test']

        # Validate ratios
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total_ratio, 1.0):
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

        # Set random seed for reproducibility
        self.seed = self.split_config.get('seed', 42)
        random.seed(self.seed)
        np.random.seed(self.seed)

    def load_cleaned_data(self) -> List[Dict[str, str]]:
        """Load cleaned parallel data."""
        # Try different file formats
        possible_files = [
            self.processed_data_dir / "cleaned_parallel_data.json.gz",
            self.processed_data_dir / "cleaned_parallel_data.json",
            self.processed_data_dir / "cleaned_parallel_data.tsv",
        ]

        data = []

        for filepath in possible_files:
            if filepath.exists():
                self.logger.info(f"Loading data from {filepath}")

                if filepath.suffix == '.gz':
                    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                        for line in f:
                            data.append(json.loads(line))
                elif filepath.suffix == '.json':
                    with open(filepath, 'r', encoding='utf-8') as f:
                        for line in f:
                            data.append(json.loads(line))
                elif filepath.suffix == '.tsv':
                    with open(filepath, 'r', encoding='utf-8') as f:
                        for line in f:
                            parts = line.strip().split('\t')
                            if len(parts) == 2:
                                data.append({
                                    'source': parts[0],
                                    'target': parts[1],
                                    'source_name': 'unknown'
                                })

                self.logger.info(f"Loaded {len(data)} pairs")
                return data

        raise FileNotFoundError("No cleaned data file found!")

    def assign_length_bin(self, text: str, num_bins: int = 5) -> int:
        """Assign a text to a length bin for stratification."""
        length = len(text.split())

        # Define bin boundaries (you can adjust these)
        if length <= 10:
            return 0
        elif length <= 20:
            return 1
        elif length <= 40:
            return 2
        elif length <= 80:
            return 3
        else:
            return 4

    def stratified_split(self, data: List[Dict[str, str]]) -> Tuple[List, List, List]:
        """
        Split data with stratification by length.

        Args:
            data: List of parallel pairs

        Returns:
            Tuple of (train, val, test) splits
        """
        if not self.split_config.get('stratify_by_length', False):
            # Simple random split
            return self.random_split(data)

        self.logger.info("Performing stratified split by length")

        # Group by length bins
        num_bins = self.split_config.get('length_bins', 5)
        bins = defaultdict(list)

        for item in data:
            bin_id = self.assign_length_bin(item['source'], num_bins)
            bins[bin_id].append(item)

        # Split each bin
        train_data = []
        val_data = []
        test_data = []

        for bin_id, bin_items in bins.items():
            self.logger.info(f"Bin {bin_id}: {len(bin_items)} items")

            # Shuffle bin
            random.shuffle(bin_items)

            # Calculate split points
            n = len(bin_items)
            train_end = int(n * self.train_ratio)
            val_end = train_end + int(n * self.val_ratio)

            # Split
            train_data.extend(bin_items[:train_end])
            val_data.extend(bin_items[train_end:val_end])
            test_data.extend(bin_items[val_end:])

        # Shuffle final splits
        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)

        return train_data, val_data, test_data

    def random_split(self, data: List[Dict[str, str]]) -> Tuple[List, List, List]:
        """
        Randomly split data into train/val/test.

        Args:
            data: List of parallel pairs

        Returns:
            Tuple of (train, val, test) splits
        """
        self.logger.info("Performing random split")

        # Shuffle data
        random.shuffle(data)

        # Calculate split points
        n = len(data)
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)

        # Split
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]

        return train_data, val_data, test_data

    def validate_splits(self, train: List, val: List, test: List) -> bool:
        """Validate that splits meet minimum size requirements."""
        min_train = self.split_config.get('min_train_samples', 100000)
        min_val = self.split_config.get('min_val_samples', 5000)
        min_test = self.split_config.get('min_test_samples', 5000)

        if len(train) < min_train:
            self.logger.warning(f"Train set ({len(train)}) smaller than minimum ({min_train})")

        if len(val) < min_val:
            self.logger.warning(f"Validation set ({len(val)}) smaller than minimum ({min_val})")

        if len(test) < min_test:
            self.logger.warning(f"Test set ({len(test)}) smaller than minimum ({min_test})")

        return len(train) > 0 and len(val) > 0 and len(test) > 0

    def save_split(self, data: List[Dict[str, str]], split_name: str):
        """Save a data split to file."""
        output_format = self.config['output']['format']
        compression = self.config['output']['compression']

        # Save in JSON format (one per line)
        if compression == 'gzip':
            output_path = self.splits_dir / f"{split_name}.json.gz"
            with gzip.open(output_path, 'wt', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        else:
            output_path = self.splits_dir / f"{split_name}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

        # Also save separate source and target files for tokenizer training
        src_path = self.splits_dir / f"{split_name}.pa"
        tgt_path = self.splits_dir / f"{split_name}.en"

        with open(src_path, 'w', encoding='utf-8') as sf, \
             open(tgt_path, 'w', encoding='utf-8') as tf:

            for item in data:
                sf.write(item['source'] + '\n')
                tf.write(item['target'] + '\n')

        self.logger.info(f"Saved {split_name} split: {len(data)} pairs")
        self.logger.info(f"  JSON: {output_path}")
        self.logger.info(f"  Source: {src_path}")
        self.logger.info(f"  Target: {tgt_path}")

    def print_split_statistics(self, train: List, val: List, test: List):
        """Print statistics about the splits."""
        total = len(train) + len(val) + len(test)

        print("\n" + "="*60)
        print("DATA SPLIT STATISTICS")
        print("="*60)
        print(f"Total pairs: {total:,}")
        print(f"\nTrain: {len(train):,} ({len(train)/total*100:.1f}%)")
        print(f"Validation: {len(val):,} ({len(val)/total*100:.1f}%)")
        print(f"Test: {len(test):,} ({len(test)/total*100:.1f}%)")

        # Length statistics
        for split_name, split_data in [('Train', train), ('Val', val), ('Test', test)]:
            src_lengths = [len(item['source'].split()) for item in split_data]
            tgt_lengths = [len(item['target'].split()) for item in split_data]

            print(f"\n{split_name} - Source length:")
            print(f"  Mean: {np.mean(src_lengths):.1f} words")
            print(f"  Median: {np.median(src_lengths):.1f} words")
            print(f"  Min: {np.min(src_lengths)} words")
            print(f"  Max: {np.max(src_lengths)} words")

            print(f"{split_name} - Target length:")
            print(f"  Mean: {np.mean(tgt_lengths):.1f} words")
            print(f"  Median: {np.median(tgt_lengths):.1f} words")
            print(f"  Min: {np.min(tgt_lengths)} words")
            print(f"  Max: {np.max(tgt_lengths)} words")

        print("="*60 + "\n")

    def run(self) -> bool:
        """Run the complete splitting process."""
        self.logger.info("Starting data splitting process")

        # Load cleaned data
        data = self.load_cleaned_data()

        if not data:
            self.logger.error("No data to split!")
            return False

        # Split data
        train, val, test = self.stratified_split(data)

        # Validate splits
        if not self.validate_splits(train, val, test):
            self.logger.error("Split validation failed!")
            return False

        # Save splits
        self.save_split(train, 'train')
        self.save_split(val, 'val')
        self.save_split(test, 'test')

        # Print statistics
        self.print_split_statistics(train, val, test)

        # Save split metadata
        metadata = {
            'total_pairs': len(data),
            'train_pairs': len(train),
            'val_pairs': len(val),
            'test_pairs': len(test),
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'seed': self.seed,
            'stratified': self.split_config.get('stratify_by_length', False),
        }

        metadata_path = self.splits_dir / 'split_metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Split metadata saved to {metadata_path}")
        self.logger.info("Data splitting process complete")

        return True


def main():
    """Main entry point."""
    splitter = DataSplitter()
    success = splitter.run()

    if success:
        print("\n✓ Data splitting completed successfully")
        print(f"  Splits saved to: {splitter.splits_dir}")
        return 0
    else:
        print("\n✗ Data splitting encountered errors")
        return 1


if __name__ == "__main__":
    sys.exit(main())
