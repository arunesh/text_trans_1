"""
Model evaluator for translation.

Handles generation and evaluation on validation/test sets.
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sentencepiece as spm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from training.evaluation.metrics import compute_all_metrics, format_metrics


class TranslationEvaluator:
    """Evaluate translation model."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: spm.SentencePieceProcessor,
        device: torch.device,
        max_length: int = 128,
        num_beams: int = 4
    ):
        """
        Initialize evaluator.

        Args:
            model: Translation model
            tokenizer: Tokenizer
            device: Device to use
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.num_beams = num_beams

    @torch.no_grad()
    def generate_translations(
        self,
        dataloader: DataLoader,
        max_samples: Optional[int] = None
    ) -> tuple[List[str], List[str]]:
        """
        Generate translations for a dataset.

        Args:
            dataloader: DataLoader with evaluation data
            max_samples: Maximum number of samples to evaluate (None for all)

        Returns:
            Tuple of (predictions, references)
        """
        self.model.eval()

        predictions = []
        references = []

        num_batches = len(dataloader)
        if max_samples:
            num_batches = min(num_batches, (max_samples + dataloader.batch_size - 1) // dataloader.batch_size)

        for i, batch in enumerate(tqdm(dataloader, total=num_batches, desc="Generating")):
            if max_samples and len(predictions) >= max_samples:
                break

            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels']

            # Generate translations
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.max_length,
                num_beams=self.num_beams
            )

            # Decode predictions
            for pred_ids in outputs:
                # Remove BOS token and everything after EOS
                pred_ids = pred_ids.tolist()

                # Find EOS token
                if self.model.eos_token_id in pred_ids:
                    eos_idx = pred_ids.index(self.model.eos_token_id)
                    pred_ids = pred_ids[1:eos_idx]  # Skip BOS, stop at EOS
                else:
                    pred_ids = pred_ids[1:]  # Skip BOS

                pred_text = self.tokenizer.decode(pred_ids)
                predictions.append(pred_text)

            # Decode references
            for label_ids in labels:
                # Remove padding (-100) and special tokens
                label_ids = [id for id in label_ids.tolist() if id not in [-100, self.model.pad_token_id]]

                # Remove BOS and EOS if present
                if label_ids[0] == self.model.bos_token_id:
                    label_ids = label_ids[1:]
                if label_ids and label_ids[-1] == self.model.eos_token_id:
                    label_ids = label_ids[:-1]

                ref_text = self.tokenizer.decode(label_ids)
                references.append(ref_text)

        return predictions, references

    def evaluate(
        self,
        dataloader: DataLoader,
        max_samples: Optional[int] = None,
        save_predictions: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset.

        Args:
            dataloader: DataLoader with evaluation data
            max_samples: Maximum number of samples to evaluate
            save_predictions: Path to save predictions (optional)

        Returns:
            Dictionary of metrics
        """
        # Generate translations
        predictions, references = self.generate_translations(dataloader, max_samples)

        # Compute metrics
        metrics = compute_all_metrics(predictions, references)

        # Save predictions if requested
        if save_predictions:
            self._save_predictions(predictions, references, save_predictions)

        return metrics

    def _save_predictions(
        self,
        predictions: List[str],
        references: List[str],
        output_path: str
    ):
        """Save predictions to file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Reference\tPrediction\n")
            for ref, pred in zip(references, predictions):
                f.write(f"{ref}\t{pred}\n")

    def evaluate_file(
        self,
        source_file: str,
        reference_file: str,
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Evaluate on parallel text files.

        Args:
            source_file: Source text file (one sentence per line)
            reference_file: Reference text file (one sentence per line)
            batch_size: Batch size for processing

        Returns:
            Dictionary of metrics
        """
        # Load texts
        with open(source_file, 'r', encoding='utf-8') as f:
            sources = [line.strip() for line in f]

        with open(reference_file, 'r', encoding='utf-8') as f:
            references = [line.strip() for line in f]

        assert len(sources) == len(references), "Source and reference files must have same length"

        # Tokenize and batch
        predictions = []

        for i in tqdm(range(0, len(sources), batch_size), desc="Translating"):
            batch_sources = sources[i:i + batch_size]

            # Tokenize
            input_ids = []
            for src in batch_sources:
                ids = self.tokenizer.encode(src, out_type=int, add_bos=True, add_eos=True)
                ids = ids[:self.max_length]
                input_ids.append(torch.tensor(ids, dtype=torch.long))

            # Pad batch
            max_len = max(len(ids) for ids in input_ids)
            padded_ids = []
            attention_masks = []

            for ids in input_ids:
                padding = max_len - len(ids)
                padded_ids.append(
                    torch.cat([ids, torch.full((padding,), self.model.pad_token_id, dtype=torch.long)])
                )
                attention_masks.append(
                    torch.cat([torch.ones(len(ids), dtype=torch.long), torch.zeros(padding, dtype=torch.long)])
                )

            input_ids_batch = torch.stack(padded_ids).to(self.device)
            attention_mask_batch = torch.stack(attention_masks).to(self.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids_batch,
                    attention_mask=attention_mask_batch,
                    max_length=self.max_length,
                    num_beams=self.num_beams
                )

            # Decode
            for pred_ids in outputs:
                pred_ids = pred_ids.tolist()

                # Remove special tokens
                if self.model.eos_token_id in pred_ids:
                    eos_idx = pred_ids.index(self.model.eos_token_id)
                    pred_ids = pred_ids[1:eos_idx]
                else:
                    pred_ids = pred_ids[1:]

                pred_text = self.tokenizer.decode(pred_ids)
                predictions.append(pred_text)

        # Compute metrics
        metrics = compute_all_metrics(predictions, references)

        return metrics
