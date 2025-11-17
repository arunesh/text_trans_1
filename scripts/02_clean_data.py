"""
Clean and preprocess parallel corpora for Punjabi-English translation.

This script:
1. Loads raw parallel data from multiple sources
2. Applies cleaning and filtering rules
3. Deduplicates data
4. Saves cleaned data in a unified format
"""

import sys
import re
import json
import gzip
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from collections import defaultdict
import unicodedata
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import setup_logger
from utils.config import load_config
from utils.stats import DataStatistics


class DataCleaner:
    """Clean and preprocess parallel text data."""

    def __init__(self, config_path: str = "configs/data_config.yaml"):
        self.config = load_config(config_path)
        self.logger = setup_logger('data_cleaning', self.config['paths']['logs_dir'])
        self.stats = DataStatistics()

        # Get cleaning parameters
        self.cleaning_params = self.config['cleaning']
        self.raw_data_dir = Path(self.config['paths']['raw_data_dir'])
        self.processed_data_dir = Path(self.config['paths']['processed_data_dir'])
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

        # Punjabi (Gurmukhi) Unicode range: \u0A00-\u0A7F
        self.punjabi_pattern = re.compile(r'[\u0A00-\u0A7F]')
        # English (Latin) pattern
        self.english_pattern = re.compile(r'[a-zA-Z]')

    def normalize_text(self, text: str) -> str:
        """Normalize Unicode text."""
        # Apply Unicode normalization
        norm_form = self.cleaning_params.get('unicode_norm', 'NFKC')
        text = unicodedata.normalize(norm_form, text)

        # Remove zero-width characters
        text = re.sub(r'[\u200b-\u200f\u202a-\u202e\ufeff]', '', text)

        # Normalize whitespace
        text = ' '.join(text.split())

        return text.strip()

    def remove_html_tags(self, text: str) -> str:
        """Remove HTML tags from text."""
        return re.sub(r'<[^>]+>', '', text)

    def remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(url_pattern, '', text)

    def remove_emails(self, text: str) -> str:
        """Remove email addresses from text."""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.sub(email_pattern, '', text)

    def check_script_validity(self, text: str, is_punjabi: bool) -> bool:
        """
        Check if text contains the expected script.

        Args:
            text: Text to check
            is_punjabi: True if text should be Punjabi, False if English

        Returns:
            True if text contains expected script
        """
        if is_punjabi:
            # Should contain Gurmukhi characters
            return bool(self.punjabi_pattern.search(text))
        else:
            # Should contain Latin characters
            return bool(self.english_pattern.search(text))

    def check_length(self, text: str) -> bool:
        """Check if text length is within acceptable range."""
        words = text.split()
        word_count = len(words)

        min_len = self.cleaning_params['min_length']
        max_len = self.cleaning_params['max_length']

        return min_len <= word_count <= max_len

    def check_length_ratio(self, source: str, target: str) -> bool:
        """Check if source/target length ratio is acceptable."""
        src_len = len(source.split())
        tgt_len = len(target.split())

        if tgt_len == 0:
            return False

        ratio = src_len / tgt_len
        min_ratio = self.cleaning_params['min_length_ratio']
        max_ratio = self.cleaning_params['max_length_ratio']

        return min_ratio <= ratio <= max_ratio

    def check_digit_ratio(self, text: str) -> bool:
        """Check if text has acceptable digit ratio."""
        if not text:
            return False

        digit_count = sum(c.isdigit() for c in text)
        ratio = digit_count / len(text)

        return ratio <= self.cleaning_params['max_digit_ratio']

    def check_punct_ratio(self, text: str) -> bool:
        """Check if text has acceptable punctuation ratio."""
        if not text:
            return False

        punct_count = sum(c in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~' for c in text)
        ratio = punct_count / len(text)

        return ratio <= self.cleaning_params['max_punct_ratio']

    def is_valid_pair(self, source: str, target: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a source-target pair is valid.

        Args:
            source: Source language text (Punjabi)
            target: Target language text (English)

        Returns:
            Tuple of (is_valid, rejection_reason)
        """
        # Check if either is empty
        if not source or not target:
            return False, "empty_text"

        # Normalize
        source = self.normalize_text(source)
        target = self.normalize_text(target)

        # Remove unwanted content
        if self.cleaning_params['remove_html']:
            source = self.remove_html_tags(source)
            target = self.remove_html_tags(target)

        if self.cleaning_params['remove_urls']:
            source = self.remove_urls(source)
            target = self.remove_urls(target)

        if self.cleaning_params['remove_emails']:
            source = self.remove_emails(source)
            target = self.remove_emails(target)

        # Re-check after cleaning
        if not source.strip() or not target.strip():
            return False, "empty_after_cleaning"

        # Check script validity
        if self.cleaning_params.get('validate_punjabi_script', True):
            if not self.check_script_validity(source, is_punjabi=True):
                return False, "invalid_punjabi_script"

        if self.cleaning_params.get('validate_english_script', True):
            if not self.check_script_validity(target, is_punjabi=False):
                return False, "invalid_english_script"

        # Check lengths
        if not self.check_length(source):
            return False, "source_length_invalid"

        if not self.check_length(target):
            return False, "target_length_invalid"

        # Check length ratio
        if not self.check_length_ratio(source, target):
            return False, "length_ratio_invalid"

        # Check digit ratio
        if not self.check_digit_ratio(source) or not self.check_digit_ratio(target):
            return False, "high_digit_ratio"

        # Check punctuation ratio
        if not self.check_punct_ratio(source) or not self.check_punct_ratio(target):
            return False, "high_punct_ratio"

        # Check if identical (both should be different languages)
        if source == target:
            return False, "identical_text"

        return True, None

    def load_tsv_file(self, filepath: Path) -> List[Tuple[str, str]]:
        """Load parallel data from TSV file."""
        pairs = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        pairs.append((parts[0], parts[1]))
        except Exception as e:
            self.logger.error(f"Error loading {filepath}: {e}")

        return pairs

    def load_parallel_files(self, source_file: Path, target_file: Path) -> List[Tuple[str, str]]:
        """Load parallel data from separate source and target files."""
        pairs = []
        try:
            with open(source_file, 'r', encoding='utf-8') as sf, \
                 open(target_file, 'r', encoding='utf-8') as tf:

                for src_line, tgt_line in zip(sf, tf):
                    src_line = src_line.strip()
                    tgt_line = tgt_line.strip()
                    if src_line and tgt_line:
                        pairs.append((src_line, tgt_line))

        except Exception as e:
            self.logger.error(f"Error loading parallel files: {e}")

        return pairs

    def load_samanantar_data(self) -> List[Tuple[str, str, str]]:
        """Load Samanantar dataset."""
        pairs = []
        samanantar_dir = self.raw_data_dir / "samanantar"

        if not samanantar_dir.exists():
            self.logger.warning(f"Samanantar directory not found: {samanantar_dir}")
            return pairs

        # Look for TSV files
        for tsv_file in samanantar_dir.rglob("*.tsv"):
            self.logger.info(f"Loading Samanantar data from {tsv_file}")
            file_pairs = self.load_tsv_file(tsv_file)
            pairs.extend([(src, tgt, "samanantar") for src, tgt in file_pairs])

        self.logger.info(f"Loaded {len(pairs)} pairs from Samanantar")
        return pairs

    def load_opus_data(self) -> List[Tuple[str, str, str]]:
        """Load OPUS datasets."""
        pairs = []
        opus_dir = self.raw_data_dir / "opus"

        if not opus_dir.exists():
            self.logger.warning(f"OPUS directory not found: {opus_dir}")
            return pairs

        # Iterate through corpus directories
        for corpus_dir in opus_dir.iterdir():
            if not corpus_dir.is_dir():
                continue

            corpus_name = corpus_dir.name
            self.logger.info(f"Loading OPUS {corpus_name}")

            # Look for parallel files
            pa_files = list(corpus_dir.glob("*.pa"))
            en_files = list(corpus_dir.glob("*.en"))

            if pa_files and en_files:
                file_pairs = self.load_parallel_files(pa_files[0], en_files[0])
                pairs.extend([(src, tgt, f"opus_{corpus_name}") for src, tgt in file_pairs])

        self.logger.info(f"Loaded {len(pairs)} pairs from OPUS")
        return pairs

    def deduplicate(self, pairs: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        """Remove duplicate pairs."""
        if not self.cleaning_params['remove_duplicates']:
            return pairs

        self.logger.info("Deduplicating data...")
        seen = set()
        unique_pairs = []

        for src, tgt, source in pairs:
            pair_hash = hash((src, tgt))
            if pair_hash not in seen:
                seen.add(pair_hash)
                unique_pairs.append((src, tgt, source))

        duplicates_removed = len(pairs) - len(unique_pairs)
        self.logger.info(f"Removed {duplicates_removed} duplicate pairs")

        return unique_pairs

    def clean_and_filter(self, pairs: List[Tuple[str, str, str]]) -> List[Dict[str, str]]:
        """Clean and filter all pairs."""
        self.logger.info(f"Cleaning and filtering {len(pairs)} pairs...")

        cleaned_pairs = []

        for src, tgt, source in tqdm(pairs, desc="Cleaning"):
            self.stats.increment_total()

            is_valid, reason = self.is_valid_pair(src, tgt)

            if is_valid:
                # Normalize one final time
                src = self.normalize_text(src)
                tgt = self.normalize_text(tgt)

                cleaned_pairs.append({
                    'source': src,
                    'target': tgt,
                    'source_name': source
                })
                self.stats.add_pair(src, tgt, source)
            else:
                self.stats.add_removal(reason)

        return cleaned_pairs

    def save_cleaned_data(self, pairs: List[Dict[str, str]], output_path: Path):
        """Save cleaned data to file."""
        self.logger.info(f"Saving {len(pairs)} cleaned pairs to {output_path}")

        output_format = self.config['output']['format']
        compression = self.config['output']['compression']

        if output_format == 'json':
            if compression == 'gzip':
                with gzip.open(output_path.with_suffix('.json.gz'), 'wt', encoding='utf-8') as f:
                    for pair in pairs:
                        f.write(json.dumps(pair, ensure_ascii=False) + '\n')
            else:
                with open(output_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
                    for pair in pairs:
                        f.write(json.dumps(pair, ensure_ascii=False) + '\n')

        elif output_format == 'tsv':
            with open(output_path.with_suffix('.tsv'), 'w', encoding='utf-8') as f:
                for pair in pairs:
                    f.write(f"{pair['source']}\t{pair['target']}\n")

        self.logger.info(f"Saved cleaned data to {output_path}")

    def run(self) -> bool:
        """Run the complete cleaning process."""
        self.logger.info("Starting data cleaning process")

        # Load all data sources
        all_pairs = []

        # Load Samanantar
        samanantar_pairs = self.load_samanantar_data()
        all_pairs.extend(samanantar_pairs)

        # Load OPUS
        opus_pairs = self.load_opus_data()
        all_pairs.extend(opus_pairs)

        if not all_pairs:
            self.logger.error("No data loaded from any source!")
            return False

        self.logger.info(f"Total pairs loaded: {len(all_pairs)}")

        # Deduplicate
        all_pairs = self.deduplicate(all_pairs)

        # Clean and filter
        cleaned_pairs = self.clean_and_filter(all_pairs)

        if not cleaned_pairs:
            self.logger.error("No valid pairs after cleaning!")
            return False

        # Save cleaned data
        output_path = self.processed_data_dir / "cleaned_parallel_data"
        self.save_cleaned_data(cleaned_pairs, output_path)

        # Save statistics
        stats_path = self.processed_data_dir / "cleaning_statistics.json"
        self.stats.save(stats_path)
        self.stats.print_summary()

        self.logger.info("Data cleaning process complete")
        return True


def main():
    """Main entry point."""
    cleaner = DataCleaner()
    success = cleaner.run()

    if success:
        print("\n✓ Data cleaning completed successfully")
        print(f"  Cleaned data saved to: {cleaner.processed_data_dir}")
        return 0
    else:
        print("\n✗ Data cleaning encountered errors")
        return 1


if __name__ == "__main__":
    sys.exit(main())
