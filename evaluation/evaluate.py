"""
evaluate.py
===========
Evaluation pipeline for the NL → HTML/CSS generation project.

Metrics:
  - BLEU       : n-gram overlap between generated and reference HTML
  - ChrF       : character-level F-score (better for code)
  - Validity   : % of outputs that pass html_utils validation
  - Length     : average character count of generated HTML
  - Components : how many UI components were correctly generated

Usage:
  # Evaluate a dataset against itself (sanity check)
  python evaluation/evaluate.py --dataset data/dataset.json

  # Evaluate predictions against references
  python evaluation/evaluate.py --predictions data/predictions.json \
                                 --references  data/references.json

  # Save results to a file
  python evaluation/evaluate.py --dataset data/dataset.json \
                                 --output  evaluation/results.json
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
from collections import Counter

from sacrebleu.metrics import BLEU, CHRF

from utils.html_utils import is_valid_output, validate_with_report, extract_components, summarize

# ── Metric instances ───────────────────────────────────────────────────────────
bleu_metric = BLEU(effective_order=True)
chrf_metric = CHRF()


# ── Per-pair scoring ───────────────────────────────────────────────────────────

def score_pair(hypothesis: str, reference: str) -> dict:
    """
    Score a single (hypothesis, reference) HTML pair.

    Args:
        hypothesis: Generated HTML string
        reference:  Ground truth HTML string

    Returns:
        Dict with bleu, chrf, valid, char_count, components
    """
    bleu  = bleu_metric.sentence_score(hypothesis, [reference]).score
    chrf  = chrf_metric.sentence_score(hypothesis, [reference]).score
    valid = is_valid_output(hypothesis)
    meta  = summarize(hypothesis)

    return {
        "bleu":       round(bleu, 2),
        "chrf":       round(chrf, 2),
        "valid":      valid,
        "char_count": meta["char_count"],
        "components": meta["components"],
        "colors":     meta["colors"],
        "fonts":      meta["fonts"],
    }


# ── Dataset evaluation ─────────────────────────────────────────────────────────

def evaluate_dataset(dataset: list[dict]) -> dict:
    """
    Evaluate a dataset where each entry has both 'prompt' and 'html'.
    Scores each entry against itself as a self-consistency check —
    useful for checking validity, length, and component coverage.

    Args:
        dataset: List of dicts with 'prompt' and 'html' keys

    Returns:
        Dict with per-entry results and aggregate stats
    """
    results = []
    for entry in dataset:
        html  = entry.get("html", "")
        valid = is_valid_output(html)
        meta  = summarize(html)
        results.append({
            "prompt":     entry.get("prompt", ""),
            "valid":      valid,
            "char_count": meta["char_count"],
            "components": meta["components"],
            "colors":     meta["colors"],
            "fonts":      meta["fonts"],
            "validation": meta["validation"],
        })

    return {
        "results":    results,
        "summary":    aggregate_stats(results),
    }


def evaluate_predictions(predictions: list[dict],
                          references:  list[dict]) -> dict:
    """
    Evaluate generated predictions against reference outputs.
    Both lists must be the same length and in the same order.

    Args:
        predictions: List of dicts with 'prompt' and 'html'
        references:  List of dicts with 'prompt' and 'html'

    Returns:
        Dict with per-pair scores and aggregate stats
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs {len(references)} references"
        )

    results = []
    for pred, ref in zip(predictions, references):
        scores = score_pair(pred["html"], ref["html"])
        scores["prompt"] = pred.get("prompt", "")
        results.append(scores)

    return {
        "results":  results,
        "summary":  aggregate_stats(results),
    }


# ── Aggregate stats ────────────────────────────────────────────────────────────

def aggregate_stats(results: list[dict]) -> dict:
    """
    Compute aggregate statistics across all scored results.

    Args:
        results: List of per-entry score dicts

    Returns:
        Dict with averages, validity rate, component frequency
    """
    n = len(results)
    if n == 0:
        return {}

    valid_count  = sum(1 for r in results if r.get("valid", False))
    total_chars  = sum(r.get("char_count", 0) for r in results)

    # BLEU / ChrF averages (only present in prediction mode)
    bleu_scores  = [r["bleu"] for r in results if "bleu" in r]
    chrf_scores  = [r["chrf"] for r in results if "chrf" in r]

    # Component frequency across dataset
    all_components: list[str] = []
    for r in results:
        all_components.extend(r.get("components", []))
    component_freq = dict(Counter(all_components).most_common())

    # Font frequency
    all_fonts: list[str] = []
    for r in results:
        all_fonts.extend(r.get("fonts", []))
    font_freq = dict(Counter(all_fonts).most_common())

    stats: dict = {
        "total":          n,
        "valid_count":    valid_count,
        "validity_rate":  round(valid_count / n * 100, 1),
        "avg_char_count": round(total_chars / n),
        "component_freq": component_freq,
        "font_freq":      font_freq,
    }

    if bleu_scores:
        stats["avg_bleu"] = round(sum(bleu_scores) / len(bleu_scores), 2)
        stats["avg_chrf"] = round(sum(chrf_scores) / len(chrf_scores), 2)

    return stats


# ── Pretty print ───────────────────────────────────────────────────────────────

def print_summary(summary: dict) -> None:
    """Print a formatted summary to the terminal."""
    print(f"\n{'═' * 50}")
    print(f"  EVALUATION SUMMARY")
    print(f"{'═' * 50}")
    print(f"  Total entries     : {summary.get('total', 0)}")
    print(f"  Valid HTML        : {summary.get('valid_count', 0)} "
          f"({summary.get('validity_rate', 0)}%)")
    print(f"  Avg char count    : {summary.get('avg_char_count', 0)}")

    if "avg_bleu" in summary:
        print(f"  Avg BLEU          : {summary['avg_bleu']}")
        print(f"  Avg ChrF          : {summary['avg_chrf']}")

    print(f"\n  Top components:")
    for comp, count in list(summary.get("component_freq", {}).items())[:8]:
        bar = "█" * min(count, 30)
        print(f"    {comp:<14} {bar} {count}")

    print(f"\n  Top fonts:")
    for font, count in list(summary.get("font_freq", {}).items())[:5]:
        print(f"    {font:<20} {count}")

    print(f"{'═' * 50}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NL → HTML dataset or predictions")

    parser.add_argument("--dataset",     type=str, default=None,
                        help="Path to dataset JSON (evaluate validity + stats)")
    parser.add_argument("--predictions", type=str, default=None,
                        help="Path to predictions JSON (requires --references)")
    parser.add_argument("--references",  type=str, default=None,
                        help="Path to references JSON (requires --predictions)")
    parser.add_argument("--output",      type=str, default=None,
                        help="Save full results to this JSON file")
    parser.add_argument("--verbose",     action="store_true",
                        help="Print per-entry scores")

    args = parser.parse_args()

    # ── Mode 1: dataset validity check ────────────────────────────────────────
    if args.dataset:
        print(f"📂 Loading dataset: {args.dataset}")
        with open(args.dataset) as f:
            dataset = json.load(f)
        print(f"   {len(dataset)} entries found\n")

        report = evaluate_dataset(dataset)
        print_summary(report["summary"])

        if args.verbose:
            print("Per-entry results:")
            for r in report["results"]:
                status = "✓" if r["valid"] else "✗"
                print(f"  {status} [{r['char_count']:>5} chars] "
                      f"{r['prompt'][:60]}")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(report, f, indent=2)
            print(f"💾 Results saved to {args.output}")

    # ── Mode 2: prediction vs reference scoring ────────────────────────────────
    elif args.predictions and args.references:
        print(f"📂 Loading predictions : {args.predictions}")
        print(f"📂 Loading references  : {args.references}")

        with open(args.predictions) as f:
            predictions = json.load(f)
        with open(args.references) as f:
            references = json.load(f)

        report = evaluate_predictions(predictions, references)
        print_summary(report["summary"])

        if args.verbose:
            print("Per-pair scores:")
            for r in report["results"]:
                status = "✓" if r["valid"] else "✗"
                print(f"  {status} BLEU:{r['bleu']:>6.2f}  "
                      f"ChrF:{r['chrf']:>6.2f}  "
                      f"{r['prompt'][:50]}")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(report, f, indent=2)
            print(f"💾 Results saved to {args.output}")

    else:
        parser.print_help()