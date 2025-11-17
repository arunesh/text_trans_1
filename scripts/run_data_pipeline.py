"""
Main script to run the complete data preparation pipeline.

This script orchestrates all data preparation steps:
1. Download datasets
2. Clean and filter data
3. Split into train/val/test
4. Train tokenizer

Usage:
    python scripts/run_data_pipeline.py [--skip-download] [--skip-clean] [--skip-split] [--skip-tokenizer]
"""

import sys
import argparse
from pathlib import Path
import time

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import setup_logger
from utils.config import load_config

# Import individual pipeline components by their numbered names
import importlib.util

def load_script_module(script_name):
    """Dynamically load a script module."""
    script_path = Path(__file__).parent / script_name
    spec = importlib.util.spec_from_file_location(script_name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

download_data = load_script_module('01_download_data.py')
clean_data = load_script_module('02_clean_data.py')
split_data = load_script_module('03_split_data.py')
train_tokenizer = load_script_module('04_train_tokenizer.py')


class DataPipeline:
    """Orchestrate the complete data preparation pipeline."""

    def __init__(self, config_path: str = "configs/data_config.yaml"):
        self.config = load_config(config_path)
        self.logger = setup_logger('data_pipeline', self.config['paths']['logs_dir'])

    def run(self,
            skip_download: bool = False,
            skip_clean: bool = False,
            skip_split: bool = False,
            skip_tokenizer: bool = False) -> bool:
        """
        Run the complete pipeline.

        Args:
            skip_download: Skip data download step
            skip_clean: Skip data cleaning step
            skip_split: Skip data splitting step
            skip_tokenizer: Skip tokenizer training step

        Returns:
            True if all steps succeed
        """
        start_time = time.time()

        self.logger.info("="*60)
        self.logger.info("STARTING DATA PREPARATION PIPELINE")
        self.logger.info("="*60)

        # Step 1: Download data
        if not skip_download:
            self.logger.info("\n[STEP 1/4] Downloading datasets...")
            downloader = download_data.DatasetDownloader()
            if not downloader.run():
                self.logger.error("Download step failed!")
                return False
        else:
            self.logger.info("\n[STEP 1/4] Skipping download (--skip-download)")

        # Step 2: Clean data
        if not skip_clean:
            self.logger.info("\n[STEP 2/4] Cleaning and filtering data...")
            cleaner = clean_data.DataCleaner()
            if not cleaner.run():
                self.logger.error("Cleaning step failed!")
                return False
        else:
            self.logger.info("\n[STEP 2/4] Skipping cleaning (--skip-clean)")

        # Step 3: Split data
        if not skip_split:
            self.logger.info("\n[STEP 3/4] Splitting data into train/val/test...")
            splitter = split_data.DataSplitter()
            if not splitter.run():
                self.logger.error("Splitting step failed!")
                return False
        else:
            self.logger.info("\n[STEP 3/4] Skipping split (--skip-split)")

        # Step 4: Train tokenizer
        if not skip_tokenizer:
            self.logger.info("\n[STEP 4/4] Training tokenizer...")
            trainer = train_tokenizer.TokenizerTrainer()
            if not trainer.run():
                self.logger.error("Tokenizer training failed!")
                return False
        else:
            self.logger.info("\n[STEP 4/4] Skipping tokenizer (--skip-tokenizer)")

        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)

        self.logger.info("="*60)
        self.logger.info("DATA PREPARATION PIPELINE COMPLETED SUCCESSFULLY")
        self.logger.info(f"Total time: {minutes}m {seconds}s")
        self.logger.info("="*60)

        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run the complete data preparation pipeline for Punjabi-English translation"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/data_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip data download step'
    )
    parser.add_argument(
        '--skip-clean',
        action='store_true',
        help='Skip data cleaning step'
    )
    parser.add_argument(
        '--skip-split',
        action='store_true',
        help='Skip data splitting step'
    )
    parser.add_argument(
        '--skip-tokenizer',
        action='store_true',
        help='Skip tokenizer training step'
    )

    args = parser.parse_args()

    # Create pipeline
    pipeline = DataPipeline(config_path=args.config)

    # Run pipeline
    success = pipeline.run(
        skip_download=args.skip_download,
        skip_clean=args.skip_clean,
        skip_split=args.skip_split,
        skip_tokenizer=args.skip_tokenizer
    )

    if success:
        print("\n" + "="*60)
        print("✓ DATA PREPARATION COMPLETE!")
        print("="*60)
        print("\nNext steps:")
        print("  1. Review the data statistics in data/processed/")
        print("  2. Check the data splits in data/splits/")
        print("  3. Verify the tokenizer in models/tokenizer/")
        print("  4. Proceed to model training (Phase 2)")
        print("="*60 + "\n")
        return 0
    else:
        print("\n✗ Data preparation pipeline failed!")
        print("  Check the logs in logs/ directory for details")
        return 1


if __name__ == "__main__":
    sys.exit(main())
