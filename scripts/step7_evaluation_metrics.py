"""
STEP 7: Evaluation Metrics

Implements the metrics ScanQA and similar 3D-QA papers report, so your
results are directly comparable to published baselines (3D-VisTA, LL3DA, etc.):
  - EM@1 (Exact Match @ 1): does the generated answer exactly match ground truth
  - BLEU-4: n-gram overlap score, standard in QA/captioning literature

HONESTY NOTE: EM@1 and BLEU-4 computation themselves have NO GPU dependency -
these are pure string/text metrics. This script CAN be fully tested in this
sandbox using dummy predictions. What CANNOT be tested here is generating
those predictions in the first place, since that requires the full model
(Steps 2-5) running on Kaggle with real LLaMA.

Normalization matters a lot for EM@1 - "The chair is blue." vs "the chair is
blue" vs "blue" can all be "correct" depending on how strict you are. ScanQA's
official eval script has specific normalization rules (lowercase, strip
punctuation, etc.) - replicate theirs exactly rather than inventing your own,
or your numbers won't be comparable to the baselines you're citing.
"""

import re
import string
from collections import Counter

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


def normalize_answer(text):
    """
    Standard QA normalization: lowercase, remove punctuation, remove
    articles (a/an/the), collapse whitespace.
    This mirrors the normalization style used in ScanQA/SQuAD-style eval -
    VERIFY against ScanQA's actual eval script before trusting these numbers
    for a paper submission, since exact normalization rules affect EM@1
    significantly and reviewers may check this.
    """
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = " ".join(text.split())
    return text


def exact_match(prediction, ground_truth):
    """EM@1: 1 if normalized prediction == normalized ground truth, else 0."""
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def bleu4(prediction, ground_truth):
    """
    BLEU-4 score for a single prediction/reference pair.
    Uses NLTK's smoothing to avoid zero scores on short sentences, which
    is standard practice for short QA answers (often <10 words).
    """
    if not NLTK_AVAILABLE:
        raise ImportError("nltk not installed. Run: pip install nltk")

    reference = [normalize_answer(ground_truth).split()]
    hypothesis = normalize_answer(prediction).split()

    if len(hypothesis) == 0:
        return 0.0

    smoothie = SmoothingFunction().method4
    score = sentence_bleu(reference, hypothesis, smoothing_function=smoothie)
    return score


def evaluate_predictions(predictions, ground_truths):
    """
    Args:
        predictions: list[str] - model-generated answers
        ground_truths: list[str] - ground truth answers (same length, same order)

    Returns:
        dict with 'em@1' and 'bleu4' averaged over all examples
    """
    assert len(predictions) == len(ground_truths), \
        f"Length mismatch: {len(predictions)} predictions vs {len(ground_truths)} ground truths"

    em_scores = []
    bleu_scores = []

    for pred, gt in zip(predictions, ground_truths):
        em_scores.append(exact_match(pred, gt))
        bleu_scores.append(bleu4(pred, gt))

    return {
        "em@1": sum(em_scores) / len(em_scores),
        "bleu4": sum(bleu_scores) / len(bleu_scores),
        "num_examples": len(predictions),
    }


if __name__ == "__main__":
    if not NLTK_AVAILABLE:
        print("Installing nltk punkt tokenizer data...")
        import nltk
        nltk.download("punkt", quiet=True)

    # Dummy test cases simulating model outputs vs ground truth, including
    # cases that test normalization (articles, punctuation, capitalization)
    dummy_predictions = [
        "The chair is blue.",       # exact match after normalization
        "there are two windows",    # exact match after normalization (no period)
        "A red table.",             # wrong - testing non-match
        "Yes, near the wall.",      # partial match - testing BLEU on partial overlap
    ]
    dummy_ground_truths = [
        "the chair is blue",
        "There are two windows.",
        "The table is red.",
        "Yes, it is near the wall.",
    ]

    print("Per-example results:")
    for pred, gt in zip(dummy_predictions, dummy_ground_truths):
        em = exact_match(pred, gt)
        bl = bleu4(pred, gt)
        print(f"  Pred: {pred!r:35} GT: {gt!r:30} EM={em} BLEU4={bl:.3f}")

    results = evaluate_predictions(dummy_predictions, dummy_ground_truths)
    print(f"\nAggregate results: {results}")

    print("\nStep 7 complete. Metric computation verified on dummy text pairs.")
    print("REMINDER: Cross-check normalize_answer() against ScanQA's official")
    print("eval script before reporting numbers in your paper - exact")
    print("normalization rules affect EM@1 and must match for fair comparison.")
