"""
Hebrew OCR Benchmark
====================
Runs the OCR engine against ground-truth annotated images and computes
detection/recognition metrics. Use this to measure the impact of
parameter changes, preprocessing tweaks, or strategy modifications.

Usage:
    conda activate hebrew-ocr
    cd backend
    python test_benchmark.py                     # run all annotated images
    python test_benchmark.py --filter poster     # only images matching 'poster'
    python test_benchmark.py --filter technai    # single image by name
    python test_benchmark.py --verbose           # per-character breakdown
"""

import json
import os
import re
import sys
import time
import argparse
from collections import Counter
from ocr_engine import HebrewOCREngine, HEBREW_REGEX


# ──────────────────────────────────────────────────
# TUNEABLE PARAMETERS
# ──────────────────────────────────────────────────

GROUND_TRUTH_PATH = os.path.join('..', 'assets', 'test_images', 'ground_truth.json')
RESULTS_OUTPUT_PATH = 'benchmark_results.json'
TEST_IMAGES_BASE = os.path.join('..', 'assets', 'test_images')

# Only count base Hebrew letters (alef–tav, U+05D0–U+05EA).
# Excludes niqqud, cantillation, maqaf, geresh — things the OCR never segments individually.
HEBREW_BASE_LETTER_REGEX = re.compile('[\u05D0-\u05EA]')


# ──────────────────────────────────────────────────
# Ground-truth loading
# ──────────────────────────────────────────────────

def load_ground_truth(path):
    """Load ground-truth annotations from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ──────────────────────────────────────────────────
# Metrics computation
# ──────────────────────────────────────────────────

def build_char_inventory(text):
    """Count occurrences of each base Hebrew letter (alef–tav) in a text string."""
    return Counter(c for c in text if HEBREW_BASE_LETTER_REGEX.search(c))


def compute_metrics(expected_text, detected_symbols):
    """Compare detected OCR symbols against expected ground-truth text, return full metrics dict."""
    expected_counts = build_char_inventory(expected_text)
    expected_total = sum(expected_counts.values())

    detected_counts = Counter(
        s['text'] for s in detected_symbols if HEBREW_BASE_LETTER_REGEX.search(s['text'])
    )
    detected_total = sum(detected_counts.values())

    expected_unique = set(expected_counts.keys())
    detected_unique = set(detected_counts.keys())

    # Inventory-level: which unique chars were found?
    matched_chars = expected_unique & detected_unique
    inv_precision = len(matched_chars) / len(detected_unique) if detected_unique else 0.0
    inv_recall = len(matched_chars) / len(expected_unique) if expected_unique else 0.0
    inv_f1 = (2 * inv_precision * inv_recall / (inv_precision + inv_recall)
              if (inv_precision + inv_recall) > 0 else 0.0)

    # Count-level: detection completeness
    detection_rate = detected_total / expected_total if expected_total > 0 else 0.0
    over_detection = (max(0, detected_total - expected_total) / expected_total
                      if expected_total > 0 else 0.0)

    # Per-character count accuracy
    per_char = {}
    for ch in expected_unique | detected_unique:
        exp = expected_counts.get(ch, 0)
        det = detected_counts.get(ch, 0)
        max_count = max(exp, det)
        per_char[ch] = {
            'expected': exp,
            'detected': det,
            'accuracy': round(min(exp, det) / max_count, 3) if max_count > 0 else 0.0,
            'delta': det - exp
        }

    # Average confidence of detected symbols (ignore unscored / negative)
    confidences = [s['confidence'] for s in detected_symbols if s['confidence'] > 0]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    return {
        'expected_total': expected_total,
        'detected_total': detected_total,
        'detection_rate': round(detection_rate, 4),
        'over_detection_rate': round(over_detection, 4),
        'inventory_precision': round(inv_precision, 4),
        'inventory_recall': round(inv_recall, 4),
        'inventory_f1': round(inv_f1, 4),
        'unique_expected': len(expected_unique),
        'unique_detected': len(detected_unique),
        'unique_matched': len(matched_chars),
        'missing_chars': sorted(expected_unique - detected_unique),
        'extra_chars': sorted(detected_unique - expected_unique),
        'avg_confidence': round(avg_confidence, 1),
        'per_char': per_char
    }


# ──────────────────────────────────────────────────
# Report printing
# ──────────────────────────────────────────────────

def print_image_report(metrics, verbose=False):
    """Print benchmark results for a single image."""
    det_rate = metrics['detection_rate']
    f1 = metrics['inventory_f1']

    if det_rate >= 0.9:
        rate_label = 'GOOD'
    elif det_rate >= 0.6:
        rate_label = 'FAIR'
    else:
        rate_label = 'POOR'

    print(f"  [{rate_label}] Detection: {metrics['detected_total']}/{metrics['expected_total']} "
          f"({det_rate:.0%})")
    print(f"  Inventory F1: {f1:.2f}  "
          f"(P={metrics['inventory_precision']:.2f}  R={metrics['inventory_recall']:.2f})")
    print(f"  Avg confidence: {metrics['avg_confidence']:.0f}")
    print(f"  Strategy: {metrics.get('strategy', '?')}  |  Time: {metrics.get('elapsed_s', 0):.1f}s")

    if metrics['missing_chars']:
        print(f"  Missing chars: {' '.join(metrics['missing_chars'])}")
    if metrics['extra_chars']:
        print(f"  Extra chars:   {' '.join(metrics['extra_chars'])}")

    if verbose and metrics['per_char']:
        print(f"  -- Per-character breakdown --")
        for ch in sorted(metrics['per_char'].keys()):
            info = metrics['per_char'][ch]
            delta_str = f"+{info['delta']}" if info['delta'] > 0 else str(info['delta'])
            bar = '#' * min(info['detected'], 30)
            print(f"    {ch}  exp={info['expected']:2d}  det={info['detected']:2d}  "
                  f"d={delta_str:>3s}  {bar}")


def print_aggregate_report(all_metrics):
    """Print summary across all benchmarked images."""
    n = len(all_metrics)
    if n == 0:
        print("\n  No images were benchmarked.")
        return

    avg_det = sum(m['detection_rate'] for m in all_metrics) / n
    avg_f1 = sum(m['inventory_f1'] for m in all_metrics) / n
    avg_conf = sum(m['avg_confidence'] for m in all_metrics) / n
    total_expected = sum(m['expected_total'] for m in all_metrics)
    total_detected = sum(m['detected_total'] for m in all_metrics)
    total_time = sum(m.get('elapsed_s', 0) for m in all_metrics)

    strategies = Counter(m.get('strategy', '?') for m in all_metrics)

    print(f"\n{'='*70}")
    print(f" AGGREGATE  ({n} images)")
    print(f"{'='*70}")
    print(f"  Total chars:          {total_detected}/{total_expected} detected")
    print(f"  Avg detection rate:   {avg_det:.1%}")
    print(f"  Avg inventory F1:     {avg_f1:.2f}")
    print(f"  Avg confidence:       {avg_conf:.0f}")
    print(f"  Total time:           {total_time:.1f}s")
    print(f"  Strategy usage:       {dict(strategies)}")

    # Per-collection breakdown
    collections = {}
    for m in all_metrics:
        col = m.get('collection', 'unknown')
        collections.setdefault(col, []).append(m)

    if len(collections) > 1:
        for col, col_metrics in sorted(collections.items()):
            cn = len(col_metrics)
            col_det = sum(m['detection_rate'] for m in col_metrics) / cn
            col_f1 = sum(m['inventory_f1'] for m in col_metrics) / cn
            print(f"\n  [{col}] ({cn} images): avg det={col_det:.0%}, avg F1={col_f1:.2f}")


# ──────────────────────────────────────────────────
# Main runner
# ──────────────────────────────────────────────────

def make_serializable(obj):
    """Recursively convert sets and non-string dict keys for JSON."""
    if isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_serializable(i) for i in obj]
    if isinstance(obj, set):
        return sorted(make_serializable(i) for i in obj)
    return obj


def run_benchmark(ground_truth_path=GROUND_TRUTH_PATH, image_filter=None, verbose=False):
    """Run the full benchmark suite against ground-truth annotations."""
    gt = load_ground_truth(ground_truth_path)
    engine = HebrewOCREngine(use_easyocr=False)

    all_metrics = []
    skipped = 0

    for entry in gt['images']:
        if entry.get('skip', False):
            skipped += 1
            continue

        if image_filter and image_filter.lower() not in entry['path'].lower():
            continue

        expected_text = entry.get('expected_text', '')
        if not expected_text or expected_text.startswith('TODO'):
            print(f"  SKIP (no annotation): {entry['path']}")
            skipped += 1
            continue

        image_path = os.path.join(TEST_IMAGES_BASE, entry['path'])
        if not os.path.exists(image_path):
            print(f"  MISSING: {image_path}")
            skipped += 1
            continue

        print(f"\n{'~'*70}")
        print(f"  {entry['path']}")
        print(f"  {entry.get('notes', '')}")
        print(f"{'~'*70}")

        start = time.time()
        try:
            ocr_result = engine.process_image(image_path, only_hebrew=True)
        except Exception as e:
            print(f"  ERROR: {e}")
            all_metrics.append({
                'path': entry['path'],
                'error': str(e),
                'detection_rate': 0, 'inventory_f1': 0,
                'avg_confidence': 0, 'expected_total': 0, 'detected_total': 0,
                'collection': entry.get('collection', 'unknown')
            })
            continue
        elapsed = time.time() - start

        metrics = compute_metrics(expected_text, ocr_result['symbols'])
        metrics['path'] = entry['path']
        metrics['collection'] = entry.get('collection', 'unknown')
        metrics['difficulty'] = entry.get('difficulty', 'unknown')
        metrics['strategy'] = ocr_result.get('strategy', 'unknown')
        metrics['elapsed_s'] = round(elapsed, 2)
        all_metrics.append(metrics)

        print_image_report(metrics, verbose=verbose)

    # Aggregate summary
    print_aggregate_report(all_metrics)

    # Persist results to JSON for tracking over time
    output = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_benchmarked': len(all_metrics),
        'skipped': skipped,
        'results': all_metrics
    }
    with open(RESULTS_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(make_serializable(output), f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved to {RESULTS_OUTPUT_PATH}")

    return all_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hebrew OCR Benchmark')
    parser.add_argument('--filter', type=str, default=None,
                        help='Only run images whose path contains this string')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show per-character breakdown for each image')
    parser.add_argument('--gt', type=str, default=GROUND_TRUTH_PATH,
                        help='Path to ground_truth.json')
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f" HEBREW OCR BENCHMARK")
    print(f"{'='*70}")
    print(f"  Ground truth: {args.gt}")
    if args.filter:
        print(f"  Filter: '{args.filter}'")

    run_benchmark(
        ground_truth_path=args.gt,
        image_filter=args.filter,
        verbose=args.verbose
    )
