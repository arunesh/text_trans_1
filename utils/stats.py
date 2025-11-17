"""Statistics and reporting utilities for data processing."""

import json
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter
import numpy as np


class DataStatistics:
    """Track and report statistics during data processing."""

    def __init__(self):
        self.stats = {
            'total_pairs': 0,
            'valid_pairs': 0,
            'removed_pairs': 0,
            'removal_reasons': Counter(),
            'source_lengths': [],
            'target_lengths': [],
            'length_ratios': [],
            'sources': Counter(),
        }

    def add_pair(self, source: str, target: str, source_name: str = 'unknown') -> None:
        """Add a valid pair to statistics."""
        self.stats['valid_pairs'] += 1
        self.stats['source_lengths'].append(len(source.split()))
        self.stats['target_lengths'].append(len(target.split()))

        ratio = len(source.split()) / max(len(target.split()), 1)
        self.stats['length_ratios'].append(ratio)
        self.stats['sources'][source_name] += 1

    def add_removal(self, reason: str) -> None:
        """Record a removed pair with reason."""
        self.stats['removed_pairs'] += 1
        self.stats['removal_reasons'][reason] += 1

    def increment_total(self, count: int = 1) -> None:
        """Increment total pairs processed."""
        self.stats['total_pairs'] += count

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        summary = {
            'total_pairs_processed': self.stats['total_pairs'],
            'valid_pairs': self.stats['valid_pairs'],
            'removed_pairs': self.stats['removed_pairs'],
            'retention_rate': self.stats['valid_pairs'] / max(self.stats['total_pairs'], 1),
        }

        if self.stats['source_lengths']:
            summary['source_length_stats'] = {
                'mean': float(np.mean(self.stats['source_lengths'])),
                'median': float(np.median(self.stats['source_lengths'])),
                'std': float(np.std(self.stats['source_lengths'])),
                'min': int(np.min(self.stats['source_lengths'])),
                'max': int(np.max(self.stats['source_lengths'])),
            }

        if self.stats['target_lengths']:
            summary['target_length_stats'] = {
                'mean': float(np.mean(self.stats['target_lengths'])),
                'median': float(np.median(self.stats['target_lengths'])),
                'std': float(np.std(self.stats['target_lengths'])),
                'min': int(np.min(self.stats['target_lengths'])),
                'max': int(np.max(self.stats['target_lengths'])),
            }

        if self.stats['length_ratios']:
            summary['length_ratio_stats'] = {
                'mean': float(np.mean(self.stats['length_ratios'])),
                'median': float(np.median(self.stats['length_ratios'])),
                'std': float(np.std(self.stats['length_ratios'])),
            }

        summary['removal_reasons'] = dict(self.stats['removal_reasons'])
        summary['data_sources'] = dict(self.stats['sources'])

        return summary

    def save(self, output_path: str) -> None:
        """Save statistics to JSON file."""
        summary = self.get_summary()
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    def print_summary(self) -> None:
        """Print summary statistics to console."""
        summary = self.get_summary()

        print("\n" + "="*60)
        print("DATA PROCESSING SUMMARY")
        print("="*60)
        print(f"Total pairs processed: {summary['total_pairs_processed']:,}")
        print(f"Valid pairs: {summary['valid_pairs']:,}")
        print(f"Removed pairs: {summary['removed_pairs']:,}")
        print(f"Retention rate: {summary['retention_rate']:.2%}")

        if 'source_length_stats' in summary:
            print("\nSource Length Statistics:")
            for key, value in summary['source_length_stats'].items():
                print(f"  {key}: {value:.2f}")

        if 'target_length_stats' in summary:
            print("\nTarget Length Statistics:")
            for key, value in summary['target_length_stats'].items():
                print(f"  {key}: {value:.2f}")

        if summary['removal_reasons']:
            print("\nRemoval Reasons:")
            for reason, count in sorted(summary['removal_reasons'].items(),
                                       key=lambda x: x[1], reverse=True):
                print(f"  {reason}: {count:,}")

        if summary['data_sources']:
            print("\nData Sources:")
            for source, count in sorted(summary['data_sources'].items(),
                                       key=lambda x: x[1], reverse=True):
                print(f"  {source}: {count:,}")

        print("="*60 + "\n")
