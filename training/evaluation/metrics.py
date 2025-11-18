"""
Evaluation metrics for translation models.

Implements BLEU, chrF, and TER metrics.
"""

from typing import List, Dict
import sacrebleu
from collections import Counter


def compute_bleu(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute BLEU score.

    Args:
        predictions: List of predicted translations
        references: List of reference translations

    Returns:
        Dictionary with BLEU score and components
    """
    # Wrap references in list (sacrebleu expects list of reference lists)
    refs = [[ref] for ref in references]

    # Compute BLEU
    bleu = sacrebleu.corpus_bleu(predictions, list(zip(*refs)))

    return {
        'bleu': bleu.score,
        'bleu_1': bleu.precisions[0],
        'bleu_2': bleu.precisions[1],
        'bleu_3': bleu.precisions[2],
        'bleu_4': bleu.precisions[3],
        'bp': bleu.bp,
        'sys_len': bleu.sys_len,
        'ref_len': bleu.ref_len
    }


def compute_chrf(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute chrF score (character n-gram F-score).

    Args:
        predictions: List of predicted translations
        references: List of reference translations

    Returns:
        Dictionary with chrF score
    """
    # Wrap references in list
    refs = [[ref] for ref in references]

    # Compute chrF
    chrf = sacrebleu.corpus_chrf(predictions, list(zip(*refs)))

    return {
        'chrf': chrf.score
    }


def compute_ter(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute TER (Translation Error Rate).

    Args:
        predictions: List of predicted translations
        references: List of reference translations

    Returns:
        Dictionary with TER score
    """
    # Wrap references in list
    refs = [[ref] for ref in references]

    # Compute TER
    ter = sacrebleu.corpus_ter(predictions, list(zip(*refs)))

    return {
        'ter': ter.score
    }


def compute_all_metrics(
    predictions: List[str],
    references: List[str]
) -> Dict[str, float]:
    """
    Compute all translation metrics.

    Args:
        predictions: List of predicted translations
        references: List of reference translations

    Returns:
        Dictionary with all metrics
    """
    metrics = {}

    # Compute BLEU
    bleu_scores = compute_bleu(predictions, references)
    metrics.update(bleu_scores)

    # Compute chrF
    chrf_scores = compute_chrf(predictions, references)
    metrics.update(chrf_scores)

    # Compute TER
    ter_scores = compute_ter(predictions, references)
    metrics.update(ter_scores)

    return metrics


def format_metrics(metrics: Dict[str, float], prefix: str = "") -> str:
    """
    Format metrics for printing.

    Args:
        metrics: Dictionary of metrics
        prefix: Prefix for metric names

    Returns:
        Formatted string
    """
    lines = []

    if prefix:
        lines.append(f"\n{prefix}:")
    else:
        lines.append("\nMetrics:")

    # Main metrics
    if 'bleu' in metrics:
        lines.append(f"  BLEU: {metrics['bleu']:.2f}")

    if 'chrf' in metrics:
        lines.append(f"  chrF: {metrics['chrf']:.2f}")

    if 'ter' in metrics:
        lines.append(f"  TER: {metrics['ter']:.2f}")

    # BLEU components
    if 'bleu_1' in metrics:
        lines.append(f"  BLEU-1/2/3/4: {metrics['bleu_1']:.2f} / "
                    f"{metrics['bleu_2']:.2f} / "
                    f"{metrics['bleu_3']:.2f} / "
                    f"{metrics['bleu_4']:.2f}")

    return "\n".join(lines)


if __name__ == "__main__":
    # Test metrics
    predictions = [
        "Hello world",
        "This is a test",
        "Machine translation is great"
    ]

    references = [
        "Hello world",
        "This is a test sentence",
        "Machine translation is awesome"
    ]

    print("Testing translation metrics...")
    metrics = compute_all_metrics(predictions, references)
    print(format_metrics(metrics))
