"""
Download parallel corpora for Punjabi-English translation.

This script downloads data from multiple sources:
1. Samanantar (AI4Bharat) - Main source
2. OPUS corpora (Ubuntu, GNOME, KDE, Tatoeba)
3. FLORES-200 (evaluation benchmark)
"""

import os
import sys
import requests
import zipfile
import gzip
import shutil
from pathlib import Path
from typing import Optional
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import setup_logger
from utils.config import load_config


class DatasetDownloader:
    """Handle downloading of parallel corpora."""

    def __init__(self, config_path: str = "configs/data_config.yaml"):
        self.config = load_config(config_path)
        self.logger = setup_logger('data_download', self.config['paths']['logs_dir'])
        self.raw_data_dir = Path(self.config['paths']['raw_data_dir'])
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

    def download_file(self, url: str, output_path: Path, chunk_size: int = 8192) -> bool:
        """
        Download a file from URL with progress bar.

        Args:
            url: URL to download from
            output_path: Path to save the downloaded file
            chunk_size: Download chunk size in bytes

        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Downloading {url}")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(output_path, 'wb') as f, tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                desc=output_path.name
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            self.logger.info(f"Downloaded successfully: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error downloading {url}: {e}")
            return False

    def extract_zip(self, zip_path: Path, extract_to: Path) -> bool:
        """Extract a zip file."""
        try:
            self.logger.info(f"Extracting {zip_path}")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            self.logger.info(f"Extracted to {extract_to}")
            return True
        except Exception as e:
            self.logger.error(f"Error extracting {zip_path}: {e}")
            return False

    def extract_gzip(self, gzip_path: Path, output_path: Path) -> bool:
        """Extract a gzip file."""
        try:
            self.logger.info(f"Extracting {gzip_path}")
            with gzip.open(gzip_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            self.logger.info(f"Extracted to {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error extracting {gzip_path}: {e}")
            return False

    def download_samanantar(self) -> bool:
        """Download Samanantar Punjabi-English corpus."""
        if not self.config['data_sources']['samanantar']['enabled']:
            self.logger.info("Samanantar download disabled in config")
            return True

        samanantar_dir = self.raw_data_dir / "samanantar"
        samanantar_dir.mkdir(parents=True, exist_ok=True)

        url = self.config['data_sources']['samanantar']['url']
        zip_path = samanantar_dir / "samanantar-pa.zip"

        if zip_path.exists():
            self.logger.info(f"Samanantar already downloaded: {zip_path}")
        else:
            if not self.download_file(url, zip_path):
                return False

        # Extract
        if not self.extract_zip(zip_path, samanantar_dir):
            return False

        self.logger.info("Samanantar download complete")
        return True

    def download_opus_corpus(self, corpus_name: str) -> bool:
        """
        Download a specific corpus from OPUS.

        Args:
            corpus_name: Name of the corpus (e.g., 'Ubuntu', 'GNOME')

        Returns:
            True if successful
        """
        opus_dir = self.raw_data_dir / "opus" / corpus_name.lower()
        opus_dir.mkdir(parents=True, exist_ok=True)

        # OPUS URLs follow a pattern
        base_url = f"https://object.pouta.csc.fi/OPUS-{corpus_name}/v1/moses"
        lang_pair = "en-pa"  # OPUS uses alphabetical order

        # Download source and target files
        for lang in ['en', 'pa']:
            url = f"{base_url}/{lang_pair}.txt.zip"
            zip_path = opus_dir / f"{corpus_name}.{lang}.txt.zip"

            if zip_path.exists():
                self.logger.info(f"OPUS {corpus_name} ({lang}) already downloaded")
                continue

            self.logger.info(f"Attempting to download OPUS {corpus_name} ({lang})")
            if not self.download_file(url, zip_path):
                self.logger.warning(f"Could not download OPUS {corpus_name} - may not be available for Punjabi")
                return False

            # Extract
            if not self.extract_zip(zip_path, opus_dir):
                return False

        return True

    def download_opus_corpora(self) -> bool:
        """Download all enabled OPUS corpora."""
        if not self.config['data_sources']['opus']['enabled']:
            self.logger.info("OPUS download disabled in config")
            return True

        corpora = self.config['data_sources']['opus']['corpora']
        success_count = 0

        for corpus_name in corpora:
            if self.download_opus_corpus(corpus_name):
                success_count += 1

        self.logger.info(f"Downloaded {success_count}/{len(corpora)} OPUS corpora")
        return success_count > 0

    def download_flores(self) -> bool:
        """Download FLORES-200 dataset."""
        if not self.config['data_sources']['flores']['enabled']:
            self.logger.info("FLORES download disabled in config")
            return True

        flores_dir = self.raw_data_dir / "flores"
        flores_dir.mkdir(parents=True, exist_ok=True)

        # FLORES-200 dev and devtest sets
        base_url = "https://github.com/facebookresearch/flores/raw/main/flores200"

        for split in ['dev', 'devtest']:
            for lang in ['eng_Latn', 'pan_Guru']:  # English and Punjabi (Gurmukhi)
                url = f"{base_url}/{split}/{lang}.{split}"
                output_path = flores_dir / f"{lang}.{split}"

                if output_path.exists():
                    self.logger.info(f"FLORES {split} ({lang}) already downloaded")
                    continue

                self.logger.info(f"Downloading FLORES {split} ({lang})")
                if not self.download_file(url, output_path):
                    self.logger.warning(f"Could not download FLORES {split} ({lang})")

        return True

    def check_custom_data(self) -> bool:
        """Check if custom data exists."""
        if not self.config['data_sources']['custom']['enabled']:
            return True

        custom_path = Path(self.config['data_sources']['custom']['path'])
        if custom_path.exists():
            self.logger.info(f"Custom data found: {custom_path}")
            return True
        else:
            self.logger.warning(f"Custom data path does not exist: {custom_path}")
            return False

    def run(self) -> bool:
        """Run the complete download process."""
        self.logger.info("Starting data download process")

        # Download Samanantar (primary source)
        self.download_samanantar()

        # Download OPUS corpora (supplementary)
        self.download_opus_corpora()

        # Download FLORES (evaluation)
        self.download_flores()

        # Check custom data
        self.check_custom_data()

        self.logger.info("Data download process complete")
        return True


def main():
    """Main entry point."""
    downloader = DatasetDownloader()
    success = downloader.run()

    if success:
        print("\n✓ Data download completed successfully")
        print(f"  Downloaded data saved to: {downloader.raw_data_dir}")
        return 0
    else:
        print("\n✗ Data download encountered errors")
        return 1


if __name__ == "__main__":
    sys.exit(main())
