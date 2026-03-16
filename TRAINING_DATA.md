# Training Data Export Pipeline

Export OCR-detected glyph crops into a structured dataset for training a custom Hebrew character classifier.

## Why Custom Training Data?

Tesseract was trained on standard book/document fonts. For typography research — posters, display type, stencil letters, hand-drawn glyphs — a custom classifier trained on *your actual collected specimens* will outperform any generic OCR engine. This pipeline bridges the gap: it uses the existing OCR for initial labeling, then exports crops in a format ready for model training.

## Pipeline Overview

```
Source Images                    OCR Engine                     Training Data
  poster/*.jpg    ──────>    HebrewOCREngine    ──────>     training_data/
  specimen/*.png            (detect + label)                   alef/
                                                                 image_0000_alef_raw.png
                                                                 image_0000_alef_gray.png
                                                                 image_0000_alef_bin.png
                                                                 image_0000_alef_inv.png
                                                               bet/
                                                                 ...
                                                               manifest.csv
                                                               export_summary.json
```

## Crop Variants

For each detected glyph, four image variants are generated:

| Variant | Filename Suffix | Description | Use Case |
|---------|----------------|-------------|----------|
| **Raw** | `_raw.png` | Original grayscale crop, variable size | Preserves full detail; useful for manual review |
| **Normalized Gray** | `_gray.png` | Resized to fixed canvas (64x64), centered, aspect-ratio preserved | Standard input for CNN classifiers |
| **Binarized** | `_bin.png` | Otsu-binarized, black text on white background | Clean signal for classifiers that expect binary input |
| **Inverted** | `_inv.png` | White text on black background | Some architectures (especially those pretrained on MNIST-like data) expect light-on-dark |

### Why Four Variants?

- **Raw** keeps the original detail — you might want to re-normalize differently later
- **Gray normalized** is the standard training input for most CNN architectures
- **Binarized** removes grayscale noise — useful for cleaner training signal and data augmentation
- **Inverted** matches the convention of many character recognition datasets (MNIST, EMNIST) where digits/letters are white on black

### Normalization Details

- Canvas size is configurable (default: 64x64, try 128x128 for more detail)
- Glyph is resized to fit inside the canvas with 10% padding on each side
- Aspect ratio is preserved (no stretching)
- Background is white (255) for gray/bin, black (0) for inverted
- Resize uses `INTER_AREA` interpolation (best for downscaling)

## Metadata

### manifest.csv

Every exported crop gets a row in `manifest.csv` with these fields:

| Field | Description |
|-------|-------------|
| `gray_path` | Relative path to normalized grayscale variant |
| `raw_path` | Relative path to raw crop |
| `bin_path` | Relative path to binarized variant |
| `inv_path` | Relative path to inverted variant |
| `label` | Hebrew character (e.g., א) |
| `label_name` | Filesystem-safe name (e.g., alef) |
| `unicode` | Unicode codepoint (e.g., U+05D0) |
| `confidence` | OCR confidence score (higher = more reliable label) |
| `source_image` | Which image this crop came from |
| `bbox` | Bounding box in source image (x0,y0,x1,y1) |
| `original_size` | Crop dimensions before normalization (WxH) |
| `canvas_size` | Normalized canvas size |
| `aspect_ratio` | Width/height ratio of original crop |
| `ink_coverage` | Fraction of dark pixels (quality signal) |
| `method` | Which OCR strategy detected this glyph |

### export_summary.json

Aggregate statistics: class distribution, skip counts, configuration used. Useful for checking dataset balance.

### Key Metadata Insights

- **`confidence`**: Use this to filter training data quality. High-confidence crops (>60) are likely correct labels. Low-confidence crops (<30) may be mislabeled — use them cautiously or review manually.
- **`ink_coverage`**: Very low (<0.05) means mostly background — likely a bad crop. Very high (>0.8) means mostly ink — likely a blob or artifact. Good crops typically fall in 0.1–0.5 range.
- **`aspect_ratio`**: Hebrew letters have characteristic aspect ratios. Outliers may indicate merged or split glyphs.
- **`method`**: Crops from `cc_matched` (connected-component matched to reference text) tend to have better labels than `cc_psm10` (individual CC recognition). Track which methods produce the best training data.

## Quality Gates

The export pipeline applies several filters:

1. **Confidence threshold** (default: 20) — drops obviously wrong detections
2. **Minimum crop size** — skips crops smaller than 3x3 pixels
3. **Hebrew-only filter** — drops non-Hebrew characters
4. **Perceptual hash deduplication** — prevents near-identical crops from inflating the dataset

## Usage

```bash
conda activate hebrew-ocr
cd backend

# Export from specimen collection (clean, high-quality crops)
python export_training_data.py --images ../assets/test_images/specimen-collection/

# Export from poster collection with larger canvas
python export_training_data.py --images ../assets/test_images/poster-collection/high-res/ --canvas-size 128

# Export a single image with strict confidence filter
python export_training_data.py --images ../assets/test_images/specimen-collection/technai.png --min-confidence 40

# Export to a custom directory, no deduplication
python export_training_data.py --images ../assets/test_images/ --output ../training_data_full --no-dedup
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--images` | (required) | Path to image file or directory |
| `--output` | `../training_data` | Output directory |
| `--canvas-size` | 64 | Square canvas size in pixels |
| `--min-confidence` | 20.0 | Minimum OCR confidence to include |
| `--no-dedup` | false | Disable perceptual hash deduplication |

## Directory Structure

```
training_data/
  alef/
    G-HaL-21_0000_alef_raw.png        # original crop
    G-HaL-21_0000_alef_gray.png       # 64x64 normalized grayscale
    G-HaL-21_0000_alef_bin.png        # 64x64 binarized
    G-HaL-21_0000_alef_inv.png        # 64x64 inverted
    technai_0001_alef_raw.png
    technai_0001_alef_gray.png
    technai_0001_alef_bin.png
    technai_0001_alef_inv.png
  bet/
    ...
  kaf-sofit/
    ...
  manifest.csv
  export_summary.json
```

Character directory names use transliterated Hebrew (alef, bet, gimel...) to avoid filesystem issues with Hebrew filenames. The full mapping:

| Char | Dir Name | Char | Dir Name |
|------|----------|------|----------|
| א | alef | נ | nun |
| ב | bet | ן | nun-sofit |
| ג | gimel | ס | samekh |
| ד | dalet | ע | ayin |
| ה | he | פ | pe |
| ו | vav | ף | pe-sofit |
| ז | zayin | צ | tsadi |
| ח | chet | ץ | tsadi-sofit |
| ט | tet | ק | qof |
| י | yod | ר | resh |
| כ | kaf | ש | shin |
| ך | kaf-sofit | ת | tav |
| ל | lamed | | |
| מ | mem | | |
| ם | mem-sofit | | |

## Building a Custom Classifier (Future)

This training data pipeline is designed to feed into a classification model. The recommended approach:

### Architecture Options

1. **Simple CNN** (good starting point)
   - 3-4 conv layers → global average pooling → dense → 27 classes
   - Input: 64x64 grayscale or binarized
   - Can train on CPU with a few hundred samples per class

2. **Transfer Learning** (better accuracy, less data)
   - MobileNetV2 or EfficientNet-B0 pretrained on ImageNet
   - Replace classification head with 27-class Hebrew output
   - Fine-tune last few layers on your crops

3. **Siamese/Contrastive** (best for few-shot)
   - Train an embedding network that maps glyphs to a latent space
   - Classify by nearest-neighbor to known exemplars
   - Works well when you have very few samples per font/style

### Data Considerations

- **Class balance**: Hebrew has 27 characters (22 base + 5 sofit forms). Some letters appear far more frequently in text. Use the class distribution from `export_summary.json` to identify under-represented classes.
- **Label noise**: OCR labels are imperfect. Consider a manual review pass on low-confidence crops before training. The `confidence` field in manifest.csv helps prioritize review.
- **Augmentation**: Useful augmentations for Hebrew glyphs:
  - Small rotations (±5°)
  - Scale jitter (90–110%)
  - Elastic distortion (simulates hand-drawn variation)
  - Erosion/dilation (simulates weight variation)
  - NOT horizontal flip (flipping changes the letter identity)
- **Train/val split**: Split by source image (not randomly) so validation images are truly unseen. This prevents the model from memorizing a specific font.

### Integration Path

1. Collect training data with this export script
2. Manually review and correct labels (especially low-confidence)
3. Train classifier
4. Integrate as a post-OCR verification step: run Tesseract for detection + bboxes, then run classifier on each crop for more accurate labeling
5. The benchmark (`test_benchmark.py`) measures the combined pipeline accuracy

## Using the Data for Training

The exported directory structure is the standard format that PyTorch and TensorFlow consume directly — no conversion needed.

### PyTorch (ImageFolder)

```python
from torchvision.datasets import ImageFolder
from torchvision import transforms

dataset = ImageFolder(
    root='training_data',
    transform=transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
)
# dataset[0] → (tensor, class_index)
# dataset.classes → ['alef', 'ayin', 'bet', ...]
```

### TensorFlow / Keras

```python
dataset = tf.keras.utils.image_dataset_from_directory(
    'training_data',
    image_size=(64, 64),
    color_mode='grayscale',
    label_mode='categorical'
)
```

Both auto-discover the 27 classes from the subdirectory names.

### Before Training — Checklist

1. **Pick a variant** — Train on `_gray.png` or `_bin.png`, not all four mixed in one pass. The loaders above pick up all PNGs in each folder, so either filter by suffix or put only one variant per folder. Use the other variants as augmentation.
2. **Train/val split** — Split by `source_image` (column in `manifest.csv`), not randomly, so the validation set contains fonts the model hasn't seen.
3. **Combine datasets** — Merge `training_data/` (specimens) and `training_data_posters/` (posters) for a more diverse training set of 2,876 crops.
4. **Review weak classes** — ף pe-sofit (24 samples) and ץ tsadi-sofit (18 samples) will need augmentation or targeted collection.

## Baseline Export Results (March 2026)

### Specimen Collection — Before vs After OCR Improvements

The export was run twice on `specimen-collection/`: once before OCR engine improvements (re-OCR scoring, Latin cross-check), and once after. The comparison shows the impact of stricter, more accurate detection on training data quality.

|  | Old (Mar 15) | New (Mar 16) | Change |
|--|--|--|--|
| Source images | 15 | 16 | +1 |
| Total crops found | 4,703 | 4,189 | -514 |
| Exported (base letters) | 2,315 | 2,028 | -287 |
| Niqqud junk crops | 60 | 0 | fixed |
| Non-Hebrew rejected | 0 | 59 | +59 |

#### Per-Character Change

| Char | Old | New | Delta | | Char | Old | New | Delta |
|------|-----|-----|-------|---|------|-----|-----|-------|
| א alef | 162 | 134 | -28 | | נ nun | 78 | 60 | -18 |
| ב bet | 117 | 111 | -6 | | ס samekh | 88 | 69 | -19 |
| ג gimel | 66 | 45 | -21 | | ע ayin | 69 | 55 | -14 |
| ד dalet | 106 | 89 | -17 | | פ pe | 63 | 57 | -6 |
| ה he | 114 | 110 | -4 | | ף pe-sofit | 22 | 22 | 0 |
| ו vav | 257 | 230 | -27 | | צ tsadi | 82 | 63 | -19 |
| ז zayin | 76 | 58 | -18 | | ץ tsadi-sofit | 17 | 14 | -3 |
| ח chet | 50 | 42 | -8 | | ק qof | 55 | 46 | -9 |
| ט tet | 106 | 86 | -20 | | ר resh | 106 | 105 | -1 |
| י yod | 115 | 118 | +3 | | ש shin | 105 | 85 | -20 |
| כ kaf | 79 | 80 | +1 | | ת tav | 71 | 60 | -11 |
| ך kaf-sofit | 34 | 35 | +1 | | | | | |
| ל lamed | 123 | 118 | -5 | | **TOTAL** | **2,315** | **2,028** | **-287** |
| מ mem | 64 | 55 | -9 | | | | | |
| ם mem-sofit | 49 | 45 | -4 | | | | | |

#### What Changed Since the First Run

| Improvement | Effect on Training Data |
|---|---|
| **Latin cross-check** (`[LATIN-REJECT]`) | Filters out 59 Hebrew detections that were actually Latin letterforms — would have been poison labels |
| **Re-OCR scoring** (`[REOCR]`) | Corrects borderline detections, changes some bounding boxes, eliminates false positives |
| **Niqqud filter** (base-letter regex) | Removes diacritical marks (ְ ִ ֶ ַ ָ ּ) that aren't useful for base-letter classification — 60 junk crops eliminated |

**Fewer crops, higher quality.** The new run exports 287 fewer base-letter crops. Every character class dropped except kaf (+1), kaf-sofit (+1), and yod (+3). The biggest drops (vav -27, alef -28, gimel -21, shin -20, tet -20) are likely false detections or mislabeled crops that the stricter engine now correctly rejects.

**Trade-off**: 2,028 clean-labeled crops will train a better classifier than 2,375 with ~350 noisy/mislabeled samples.

### Poster Collection Export

68 source images from `poster-collection/` (both `high-res/` and `low-res/`).

|  | Value |
|--|--|
| Source images | 68 |
| Total crops found | 908 |
| Exported | 848 |
| Skipped (low confidence) | 21 |
| Skipped (duplicate) | 37 |
| Skipped (non-Hebrew) | 2 |
| Crops per image | 12.5 avg |
| Export rate | 93% |

#### Key observations

- **93% export rate** vs 48% for specimens — poster crops have very low duplication (37 total), because each poster uses a different typeface. Specimens repeat the same alphabet across font weights, so deduplication removes half.
- **12.5 crops per image** from 68 poster images — confirms the benchmark finding that Tesseract detects ~17–33% of characters on posters. But the crops it *does* find are usable.
- **Only 21 skipped for low confidence** — the detections that survive are reasonably confident.

### Combined Dataset: Specimen + Poster

| Char | Specimen | Poster | Combined | | Char | Specimen | Poster | Combined |
|------|----------|--------|----------|---|------|----------|--------|----------|
| א alef | 134 | 55 | 189 | | נ nun | 60 | 27 | 87 |
| ב bet | 111 | 50 | 161 | | ן nun-sofit | 36 | 14 | 50 |
| ג gimel | 45 | 13 | 58 | | ס samekh | 69 | 15 | 84 |
| ד dalet | 89 | 38 | 127 | | ע ayin | 55 | 24 | 79 |
| ה he | 110 | 72 | 182 | | פ pe | 57 | 23 | 80 |
| ו vav | 230 | 128 | 358 | | ף pe-sofit | 22 | 2 | **24** |
| ז zayin | 58 | 19 | 77 | | צ tsadi | 63 | 16 | 79 |
| ח chet | 42 | 26 | 68 | | ץ tsadi-sofit | 14 | 4 | **18** |
| ט tet | 86 | 21 | 107 | | ק qof | 46 | 11 | 57 |
| י yod | 118 | 50 | 168 | | ר resh | 105 | 36 | 141 |
| כ kaf | 80 | 13 | 93 | | ש shin | 85 | 37 | 122 |
| ך kaf-sofit | 35 | 14 | 49 | | ת tav | 60 | 32 | 92 |
| ל lamed | 118 | 55 | 173 | | | | | |
| מ mem | 55 | 34 | 89 | | **TOTAL** | **2,028** | **848** | **2,876** |
| ם mem-sofit | 45 | 19 | 64 | | | | | |

#### Weakest classes (combined < 40 samples)

- **ף pe-sofit**: 24 samples (22 specimen + 2 poster) — very rare in text, final form
- **ץ tsadi-sofit**: 18 samples (14 specimen + 4 poster) — same issue, rare final form

These two classes will likely need targeted collection or heavy augmentation to train reliably.

#### What the poster data adds

Poster crops are stylistically diverse — each comes from a different typeface, era, and print quality. While specimens provide clean, repeated examples of the same fonts at multiple weights, poster crops add:
- Display/decorative type styles not seen in specimen sheets
- Real-world degradation (JPEG compression, scanning artifacts, uneven lighting)
- Variety in stroke weight, proportion, and ornamentation

This makes the combined dataset more robust for a classifier that needs to generalize beyond clean specimens.

## Remarks and Insights

### What Makes Hebrew Character Classification Hard

- **Similar shapes**: ד/ר, ו/ן, כ/ב, ח/ת have subtle differences that vary by font
- **Sofit forms**: Final forms (ך, ם, ן, ף, ץ) add 5 classes that look similar to their medial counterparts
- **Stylistic variation**: Display type and hand-drawn fonts have far more variation than book fonts
- **Dot/dagesh**: Some letters differ only by a dot (ב/בּ, כ/כּ, פ/פּ) — if the dot is part of the design, it's critical; if it's a printing artifact, it should be ignored

### Why Four Variants Help

- Training on binarized crops reduces the model's reliance on grayscale texture and forces it to learn shape
- Training on inverted crops makes the model invariant to polarity (some source images have light text on dark backgrounds)
- Raw crops preserve font-specific details (stroke weight, serifs, ink texture) that may help distinguish similar shapes
- Mixing variants during training acts as a form of data augmentation

### Ink Coverage as Quality Signal

The `ink_coverage` metric is a cheap proxy for crop quality:
- `0.0 – 0.05`: Almost empty crop → likely a false detection or tiny noise fragment
- `0.05 – 0.15`: Thin strokes or small glyph → might be correct but fragile
- `0.15 – 0.50`: Normal range for a well-detected character
- `0.50 – 0.80`: Very heavy/bold font or tight crop → probably fine
- `0.80 – 1.00`: Mostly ink → likely a blob, artifact, or severely tight crop
