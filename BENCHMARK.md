# Hebrew OCR Benchmark

Formalized testing framework for measuring OCR accuracy against ground-truth annotations.

## Why a Benchmark?

Without objective measurements, every parameter change is a coin flip. The benchmark lets us:
- **Quantify** detection accuracy (not just "looks about right")
- **Compare** before/after when tuning parameters (CROP_BOX_MARGIN, MIN_CHAR_CONFIDENCE, etc.)
- **Track regressions** — a change that helps posters might hurt specimens
- **Identify weak spots** — which characters, collections, or difficulty levels fail most?

## Architecture

```
assets/test_images/ground_truth.json   ← human-annotated expected text per image
backend/test_benchmark.py              ← runs OCR + computes metrics
backend/benchmark_results.json         ← output (auto-generated, gitignored)
```

The benchmark imports `HebrewOCREngine` directly — no API server needed.

## Ground Truth Format

Each image entry in `ground_truth.json` contains:

| Field | Description |
|-------|-------------|
| `path` | Relative path from `assets/test_images/` |
| `collection` | Group name for per-collection reporting |
| `difficulty` | `easy` / `medium` / `hard` — subjective rating |
| `expected_text` | Every Hebrew character visible in the image, in reading order |
| `total_expected` | Count of Hebrew chars (auto-verified against expected_text length) |
| `notes` | Free-text description |
| `skip` | (optional) Set `true` to exclude from benchmark |

### How to Annotate

1. Open the image
2. Type every Hebrew character you see into `expected_text` — spaces don't matter, only Hebrew chars are counted
3. Set `total_expected` to the number of Hebrew characters
4. Leave non-Hebrew characters (Latin, numbers, punctuation) out

**Key insight**: We're measuring character-level detection, not reading order or word segmentation. The benchmark counts "did the OCR find the right characters in the right quantities?" — not "did it read them left-to-right correctly."

### Why Not Bounding-Box Ground Truth?

Full bbox annotation (x,y coordinates per glyph) would enable IoU-based spatial accuracy, but:
- It's extremely labor-intensive (10+ minutes per image vs 1 minute for text inventory)
- Our primary question is "did the OCR find all the letters?" not "are the boxes pixel-perfect?"
- Character-inventory ground truth is fast to create and already reveals 90% of the useful signal

We can add bbox-level ground truth later for specific problem images.

## Metrics

### Per-Image Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Detection Rate** | `detected / expected` | Did we find all the letters? >1.0 = over-detection |
| **Over-detection Rate** | `max(0, detected - expected) / expected` | How much noise/false positives? |
| **Inventory Precision** | `matched_unique / detected_unique` | Of unique chars detected, how many are real? |
| **Inventory Recall** | `matched_unique / expected_unique` | Of unique chars expected, how many were found? |
| **Inventory F1** | Harmonic mean of P and R | Best single number for "how well did we do?" |
| **Avg Confidence** | Mean Tesseract confidence of detected symbols | Higher = OCR is more certain |

### Per-Character Breakdown (--verbose)

For each character: expected count vs detected count, delta, and a visual bar. This reveals:
- Which letters the OCR consistently misses (e.g., ן/ך sofit forms)
- Which letters get over-detected (false positives)
- Confusion patterns (e.g., ו detected as ן)

### Aggregate Metrics

- Average detection rate and F1 across all images
- Per-collection breakdown (poster-high-res vs poster-low-res vs specimen)
- Strategy distribution (which strategies fired for which images)

## Usage

```bash
conda activate hebrew-ocr
cd backend

# Run all annotated images
python test_benchmark.py

# Run only poster images
python test_benchmark.py --filter poster

# Run a specific image with per-character detail
python test_benchmark.py --filter technai --verbose

# Use a custom ground-truth file
python test_benchmark.py --gt /path/to/custom_ground_truth.json
```

## Output

Results are saved to `backend/benchmark_results.json` with full metrics per image plus a timestamp. Compare two runs by diffing or loading both JSON files.

### Interpreting Results

| Detection Rate | Meaning |
|---------------|---------|
| 90–110% | Excellent — nearly all chars found with minimal noise |
| 70–90% | Good — most chars found, some missed |
| 50–70% | Fair — significant misses, investigate specific chars |
| <50% | Poor — strategy or preprocessing likely failing |

| Inventory F1 | Meaning |
|--------------|---------|
| >0.85 | All expected character types are being found |
| 0.6–0.85 | Some character types missing |
| <0.6 | Major gaps in character coverage |

## Test Image Selection

The current ground-truth set covers three difficulty tiers:

### Poster Collection — High-Res (easy)
Clean, high-resolution scans. OCR should perform well here. If it doesn't, there's a fundamental issue.

### Poster Collection — Low-Res (medium)
Typical real-world quality. JPEG compression, lower resolution, sometimes decorative fonts. This is the bread-and-butter use case.

### Specimen Collection (easy–medium)
Type specimen sheets — relatively clean, isolated text. Should be straightforward but may have unusual glyph forms (stencil cuts, hand-drawn styles).

## Baseline Results (March 2026)

First full benchmark run across 12 ground-truth images. This is the baseline that future parameter changes and the custom classifier should be measured against.

### Per-Image Summary

| Image | Exp | Det | Rate | F1 | Strategy | Time |
|-------|-----|-----|------|-----|----------|------|
| G-HaL-21.jpg (poster-HR) | 39 | 13 | 33% | 0.62 | S1 | 1.7s |
| G-StF-Pos-002 (poster-HR) | 67 | 12 | 18% | 0.62 | S1 | 1.8s |
| G-IrS-Pos-018 (poster-HR) | 64 | 11 | 17% | 0.55 | S1 | 2.0s |
| G-CPrv-Pos-009 (poster-LR) | 41 | 7 | 17% | 0.45 | S3 | 1.7s |
| G-CPrv-Pos-032 (poster-LR) | 35 | 7 | 20% | 0.52 | S1 | 1.2s |
| G-StR-Pos-008 (poster-LR) | 171 | 56 | 33% | 0.91 | S1 | 7.4s |
| G-WeE-Pos-013 (poster-LR) | 73 | 2 | 3% | 0.09 | S2 | 1.0s |
| technai.png (specimen) | 44 | 28 | 64% | 0.62 | S1 | 3.7s |
| פרמז׳אנו.png (specimen) | 6 | 9 | 150% | 0.67 | S1 | 2.0s |
| G-NaB-03 בזלת (specimen) | 428 | 214 | 50% | 1.00 | S1 | 26.9s |
| G-CoA-08 פלוני (specimen) | 105 | 66 | 63% | 0.92 | S1 | 8.6s |
| G-Rag-01 דרוגולין (specimen) | 29 | 37 | 128% | 0.96 | S1 | 5.4s |

### Collection-Level Story

| Collection | Chars Found | Avg Detection | Avg F1 |
|---|---|---|---|
| **poster-high-res** | 36/170 | **23%** | 0.60 |
| **poster-low-res** | 72/320 | **18%** | 0.49 |
| **specimen** | 354/612 | **91%** | 0.83 |

Specimens work reasonably well. Posters are fundamentally broken with Tesseract — even high-res posters land at 17–33%. The resolution isn't the problem; it's the display/decorative typefaces.

### Most Missed Characters (across all 12 images)

| Char | Missed In | Why |
|------|-----------|-----|
| **ם** (mem-sofit) | 9/12 | Closed form, easily confused with background shapes |
| **ע** (ayin) | 7/12 | Complex shape, Tesseract struggles with open curves |
| **ד** (dalet) | 6/12 | Very similar to ר, ambiguous in stylized fonts |
| **ת** (tav) | 6/12 | Confused with ח in many display fonts |
| **י** (yod) | 5/12 | Tiny — often below minimum glyph size or merged |
| **ר** (resh) | 5/12 | Confused with ד in stylized fonts |
| **ש** (shin) | 5/12 | Complex multi-stroke shape |
| **נ** (nun) | 5/12 | Similar to ג in some fonts |
| **ק** (qof) | 5/12 | |

### Easiest Characters (aggregate detection rate)

| Char | Rate | Why |
|------|------|-----|
| **ך** (kaf-sofit) | 114% | Distinctive descender shape |
| **ס** (samekh) | 71% | Round, distinctive |
| **כ** (kaf) | 67% | Clean rectangular shape |
| **ן** (nun-sofit) | 58% | Long vertical, distinct |

### Key Insights

**1. F1 and detection rate tell different stories.** `G-NaB-03 בזלת` has F1=1.00 (all 25 unique chars found) but only 50% detection rate (finds half the instances). The OCR *knows* the alphabet but inconsistently detects repeated occurrences. This is a segmentation problem, not a recognition problem.

**2. `G-StR-Pos-008` is the best poster** — F1=0.91, finding 20/24 unique chars. It's a text-heavy poster with more conventional lettering. Decorative/display posters (`G-WeE-Pos-013` at F1=0.09) are near-total failures. Font style is the dominant factor, not resolution.

**3. Specimens are reliable training data sources** — 50–150% detection with high inventory coverage. Exported training crops from these will have reasonably trustworthy labels.

**4. Posters need the custom classifier** — Tesseract is a ~20% detector on poster images. The path forward: use connected-component segmentation for *detection* (finding where glyphs are), then the custom classifier for *recognition* (labeling what each glyph is).

**5. Priority characters for manual review** — ע, ם, ד, ת, י are the most frequently missed/mislabeled. These should be prioritized in training data collection and manual review passes.

## Workflow: Using the Benchmark During Development

1. Run benchmark to establish a **baseline**: `python test_benchmark.py > baseline.txt`
2. Make a parameter change (e.g., adjust `MIN_CHAR_CONFIDENCE` in `ocr_engine.py`)
3. Re-run benchmark: `python test_benchmark.py`
4. Compare `benchmark_results.json` before/after
5. If aggregate metrics improved (or held steady) → keep the change
6. If specific collections regressed → investigate before merging

## Future Extensions

- **Bbox-level ground truth**: For problem images, add `expected_bboxes` and compute IoU
- **Confusion matrix**: Track which chars get confused with which (e.g., ו↔ן, ד↔ר)
- **Automated regression CI**: Run benchmark on PR, fail if F1 drops below threshold
- **Per-strategy scoring**: Score Strategy 1 vs 2 vs 3 independently to guide cascade tuning
- **Visual diff report**: HTML page showing expected vs detected overlaid on image
