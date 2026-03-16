"""
Export OCR Crops as Training Data
=================================
Runs the OCR engine on images and exports normalized, binarized, and inverted
crop variants organized by character — ready for training a custom classifier.

For each detected glyph, four image variants are saved:
  - *_raw.png    — original grayscale crop (variable size)
  - *_gray.png   — normalized grayscale on fixed canvas (centered, padded)
  - *_bin.png    — binarized (black text on white background)
  - *_inv.png    — inverted (white text on black background)

Usage:
    conda activate hebrew-ocr
    cd backend
    python export_training_data.py --images ../assets/test_images/specimen-collection/
    python export_training_data.py --images ../assets/test_images/poster-collection/high-res/ --canvas-size 128
    python export_training_data.py --images path/to/single_image.jpg --min-confidence 40
"""

import cv2
import numpy as np
import os
import re
import json
import csv
import time
import argparse
from pathlib import Path
from ocr_engine import HebrewOCREngine, HEBREW_REGEX

# Only base Hebrew letters (alef–tav) — excludes niqqud, cantillation, maqaf, geresh
HEBREW_BASE_LETTER_REGEX = re.compile('[\u05D0-\u05EA]')


# ──────────────────────────────────────────────────
# TUNEABLE PARAMETERS
# ──────────────────────────────────────────────────

DEFAULT_CANVAS_SIZE = 64          # crops are centered inside a square canvas of this size
DEFAULT_MIN_CONFIDENCE = 20.0     # skip crops below this OCR confidence
DEFAULT_OUTPUT_DIR = '../training_data'
PADDING_FRACTION = 0.1            # fraction of canvas reserved as margin around glyph
MANIFEST_FILENAME = 'manifest.csv'
SUMMARY_FILENAME = 'export_summary.json'

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}

# Hebrew character → filesystem-safe directory name
CHAR_DIR_NAMES = {
    'א': 'alef',     'ב': 'bet',       'ג': 'gimel',     'ד': 'dalet',
    'ה': 'he',       'ו': 'vav',       'ז': 'zayin',     'ח': 'chet',
    'ט': 'tet',      'י': 'yod',       'כ': 'kaf',       'ך': 'kaf-sofit',
    'ל': 'lamed',    'מ': 'mem',       'ם': 'mem-sofit',  'נ': 'nun',
    'ן': 'nun-sofit','ס': 'samekh',    'ע': 'ayin',      'פ': 'pe',
    'ף': 'pe-sofit', 'צ': 'tsadi',     'ץ': 'tsadi-sofit','ק': 'qof',
    'ר': 'resh',     'ש': 'shin',      'ת': 'tav'
}


# ──────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────

def get_char_dir_name(char):
    """Map a Hebrew character to a filesystem-safe directory name."""
    return CHAR_DIR_NAMES.get(char, f'u{ord(char):04x}')


def compute_ink_coverage(gray_crop):
    """Fraction of dark (ink) pixels in a grayscale crop via Otsu binarization."""
    if gray_crop.size == 0:
        return 0.0
    _, binary = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Normalize polarity: ink should be foreground (white in mask)
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)
    return float(np.count_nonzero(binary)) / binary.size


def normalize_crop(gray_crop, canvas_size, padding_fraction=PADDING_FRACTION):
    """Resize crop to fit centered inside a square canvas, preserving aspect ratio."""
    h, w = gray_crop.shape[:2]
    if h == 0 or w == 0:
        return np.full((canvas_size, canvas_size), 255, dtype=np.uint8)

    # Usable area after reserving padding on each side
    usable = int(canvas_size * (1.0 - 2 * padding_fraction))
    scale = min(usable / w, usable / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))

    resized = cv2.resize(gray_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Place centered on a white canvas
    canvas = np.full((canvas_size, canvas_size), 255, dtype=np.uint8)
    y_off = (canvas_size - new_h) // 2
    x_off = (canvas_size - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    return canvas


def binarize_crop(gray_crop):
    """Otsu binarization → black text on white background."""
    _, binary = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Ensure dark-on-light polarity
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)
    return binary


def invert_crop(binarized_crop):
    """Invert binarized crop → white text on black background."""
    return cv2.bitwise_not(binarized_crop)


def perceptual_hash(gray_img, hash_size=8):
    """Simple average-hash for near-duplicate detection."""
    resized = cv2.resize(gray_img, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    avg = resized.mean()
    bits = (resized > avg).flatten()
    return ''.join('1' if b else '0' for b in bits)


def collect_image_paths(input_path):
    """Gather image file paths from a single file or a directory (recurses into subdirectories)."""
    path = Path(input_path)

    if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
        return [str(path)]

    if path.is_dir():
        return sorted(
            str(f) for f in path.rglob('*')
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS and not f.name.startswith('.')
        )

    return []


# ──────────────────────────────────────────────────
# Main export pipeline
# ──────────────────────────────────────────────────

def export_training_data(input_path, output_dir=DEFAULT_OUTPUT_DIR,
                         canvas_size=DEFAULT_CANVAS_SIZE,
                         min_confidence=DEFAULT_MIN_CONFIDENCE,
                         deduplicate=True):
    """Run OCR on images and export normalized training crops with all variants."""
    image_paths = collect_image_paths(input_path)
    if not image_paths:
        print(f"No supported images found in: {input_path}")
        return

    print(f"\n{'='*60}")
    print(f" EXPORT TRAINING DATA")
    print(f"{'='*60}")
    print(f"  Input:          {input_path} ({len(image_paths)} images)")
    print(f"  Output:         {output_dir}")
    print(f"  Canvas size:    {canvas_size}x{canvas_size}")
    print(f"  Min confidence: {min_confidence}")
    print(f"  Deduplication:  {'on' if deduplicate else 'off'}")

    engine = HebrewOCREngine(use_easyocr=False)
    os.makedirs(output_dir, exist_ok=True)

    manifest_rows = []
    seen_hashes = {}
    stats = {
        'total_crops': 0,
        'skipped_low_conf': 0,
        'skipped_duplicate': 0,
        'skipped_non_hebrew': 0,
        'skipped_tiny': 0,
        'exported': 0,
        'by_char': {}
    }

    for img_path in image_paths:
        print(f"\n  Processing: {os.path.basename(img_path)}")

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"    Could not load: {img_path}")
            continue

        img_gray = (cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                    if len(img_bgr.shape) == 3 else img_bgr)

        try:
            result = engine.process_image(img_path, only_hebrew=True)
        except Exception as e:
            print(f"    OCR failed: {e}")
            continue

        source_name = Path(img_path).stem

        for i, sym in enumerate(result['symbols']):
            stats['total_crops'] += 1
            char = sym['text']

            if not HEBREW_BASE_LETTER_REGEX.search(char):
                stats['skipped_non_hebrew'] += 1
                continue

            conf = sym['confidence']
            if conf > 0 and conf < min_confidence:
                stats['skipped_low_conf'] += 1
                continue

            # Extract raw crop from grayscale image
            x0, y0, x1, y1 = [int(v) for v in sym['bbox']]
            h_img, w_img = img_gray.shape[:2]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w_img, x1), min(h_img, y1)

            raw_crop = img_gray[y0:y1, x0:x1]
            if raw_crop.size == 0 or raw_crop.shape[0] < 3 or raw_crop.shape[1] < 3:
                stats['skipped_tiny'] += 1
                continue

            # Deduplication via perceptual hash
            if deduplicate:
                phash = perceptual_hash(raw_crop)
                hash_key = f"{char}_{phash}"
                if hash_key in seen_hashes:
                    stats['skipped_duplicate'] += 1
                    continue
                seen_hashes[hash_key] = True

            # Generate all four crop variants
            normalized = normalize_crop(raw_crop, canvas_size)
            binarized = binarize_crop(normalized)
            inverted = invert_crop(binarized)

            # Compute metadata
            orig_h, orig_w = raw_crop.shape[:2]
            ink = compute_ink_coverage(raw_crop)
            aspect_ratio = orig_w / orig_h if orig_h > 0 else 1.0

            # Write to character subdirectory
            char_dir_name = get_char_dir_name(char)
            char_dir = os.path.join(output_dir, char_dir_name)
            os.makedirs(char_dir, exist_ok=True)

            seq = stats['by_char'].get(char, 0)
            base_name = f"{source_name}_{seq:04d}_{char_dir_name}"

            raw_path = os.path.join(char_dir, f"{base_name}_raw.png")
            gray_path = os.path.join(char_dir, f"{base_name}_gray.png")
            bin_path = os.path.join(char_dir, f"{base_name}_bin.png")
            inv_path = os.path.join(char_dir, f"{base_name}_inv.png")

            cv2.imwrite(raw_path, raw_crop)
            cv2.imwrite(gray_path, normalized)
            cv2.imwrite(bin_path, binarized)
            cv2.imwrite(inv_path, inverted)

            stats['by_char'][char] = seq + 1
            stats['exported'] += 1

            manifest_rows.append({
                'gray_path': os.path.relpath(gray_path, output_dir),
                'raw_path': os.path.relpath(raw_path, output_dir),
                'bin_path': os.path.relpath(bin_path, output_dir),
                'inv_path': os.path.relpath(inv_path, output_dir),
                'label': char,
                'label_name': char_dir_name,
                'unicode': f'U+{ord(char):04X}',
                'confidence': round(conf, 1),
                'source_image': os.path.basename(img_path),
                'bbox': f"{x0},{y0},{x1},{y1}",
                'original_size': f"{orig_w}x{orig_h}",
                'canvas_size': canvas_size,
                'aspect_ratio': round(aspect_ratio, 3),
                'ink_coverage': round(ink, 4),
                'method': sym.get('method', 'unknown')
            })

    # ── Write manifest CSV ──
    manifest_path = os.path.join(output_dir, MANIFEST_FILENAME)
    if manifest_rows:
        fieldnames = list(manifest_rows[0].keys())
        with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_rows)

    # ── Write summary JSON ──
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'input': str(input_path),
        'output': str(output_dir),
        'canvas_size': canvas_size,
        'min_confidence': min_confidence,
        'deduplicate': deduplicate,
        'total_source_images': len(image_paths),
        'stats': {
            'total_crops_found': stats['total_crops'],
            'exported': stats['exported'],
            'skipped_low_confidence': stats['skipped_low_conf'],
            'skipped_duplicate': stats['skipped_duplicate'],
            'skipped_non_hebrew': stats['skipped_non_hebrew'],
            'skipped_tiny': stats['skipped_tiny']
        },
        'class_distribution': {
            f"{ch} ({get_char_dir_name(ch)})": count
            for ch, count in sorted(stats['by_char'].items())
        }
    }

    summary_path = os.path.join(output_dir, SUMMARY_FILENAME)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ── Print report ──
    print(f"\n{'='*60}")
    print(f" EXPORT COMPLETE")
    print(f"{'='*60}")
    print(f"  Exported:           {stats['exported']} crops")
    print(f"  Skipped (low conf): {stats['skipped_low_conf']}")
    print(f"  Skipped (dups):     {stats['skipped_duplicate']}")
    print(f"  Skipped (non-Heb):  {stats['skipped_non_hebrew']}")
    print(f"  Skipped (tiny):     {stats['skipped_tiny']}")

    print(f"\n  Class distribution:")
    for ch in sorted(stats['by_char'].keys()):
        count = stats['by_char'][ch]
        bar = '#' * min(count, 40)
        print(f"    {ch} ({get_char_dir_name(ch):>12s}): {count:4d}  {bar}")

    print(f"\n  Output dir: {output_dir}")
    print(f"  Manifest:   {manifest_path}")
    print(f"  Summary:    {summary_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export OCR crops as training data')
    parser.add_argument('--images', type=str, required=True,
                        help='Path to image file or directory of images')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--canvas-size', type=int, default=DEFAULT_CANVAS_SIZE,
                        help=f'Normalized canvas size in px (default: {DEFAULT_CANVAS_SIZE})')
    parser.add_argument('--min-confidence', type=float, default=DEFAULT_MIN_CONFIDENCE,
                        help=f'Minimum OCR confidence to include (default: {DEFAULT_MIN_CONFIDENCE})')
    parser.add_argument('--no-dedup', action='store_true',
                        help='Disable perceptual hash deduplication')
    args = parser.parse_args()

    export_training_data(
        input_path=args.images,
        output_dir=args.output,
        canvas_size=args.canvas_size,
        min_confidence=args.min_confidence,
        deduplicate=not args.no_dedup
    )
