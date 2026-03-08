# OCR Improvement Process Discussion

## Initial Question: Why is Google Lens/Translate Better Than Tesseract?

**Question:** Before we try to add preprocessing, let's rethink the tesseract. What OCR libraries are there without extra training, also for hebrew. The thing to think about is that google lens and google translate handles all these test images really well compared to tesseract. Why?

## Analysis: Why Google Lens/Translate Outperform Tesseract

### Tesseract (including Tesseract.js)

- **Classical engine** with some LSTM improvements, but segmentation is still largely heuristic (connected components, baselines, etc.)
- Trained mostly on **text-like layouts** and standard fonts, not on isolated glyphs, posters, display typography, or weird stencils
- Limited language model and context, especially for **single characters**
- In this use case, we're forcing it into the hardest mode: "symbol-level classification of unusual Hebrew glyph shapes", which is exactly where generic Tesseract is weakest

### Google Lens / Google Translate (on-device / cloud)

- Use **large modern deep models** (CNN + transformer-style recognizers) trained on:
  - Massive multilingual datasets (photos, scans, signs, UI screenshots)
  - Tons of noisy typography and layouts
- They do:
  - Robust **text detection** (separate model for finding text regions/lines)
  - More advanced **script detection**
  - Strong **language modeling / decoding** on top of raw character logits
- Their recognizer sees far more **variety of Hebrew fonts and contexts**, and can lean on context (neighboring chars / words) even when an individual glyph is ambiguous

**Conclusion:** Lens/Translate work better mainly because they use **bigger, more modern, heavily-trained vision+language stacks**, not just because of a different "OCR library".

## OCR Libraries Available (Without Extra Training, Hebrew Support)

### Open Source Options

1. **Tesseract 5 (native)**
   - What we currently have, but compiled; same `heb` traineddata
   - No magic gain for glyph-level use case

2. **Kraken** (https://github.com/mittagessen/kraken)
   - Modern sequence-to-sequence recognizer, good for historical / non-Latin scripts
   - Pretrained models for right-to-left scripts and Hebrew manuscripts
   - Requires Python, not browser-based

3. **Calamari OCR** (https://github.com/Calamari-OCR/calamari)
   - LSTM-based, some pretrained non-Latin models
   - More research-oriented

4. **PaddleOCR** (https://github.com/PaddlePaddle/PaddleOCR)
   - Strong open-source pipeline (detection + recognition)
   - Has multilingual models; Hebrew supported via Latin+Arabic-like heads
   - Quality varies; requires Python/ONNX runtime

5. **EasyOCR** (https://github.com/JaidedAI/EasyOCR)
   - Supports **Hebrew out of the box** (`lang_list=['he']`)
   - Deep-learning based; often better than Tesseract on tricky fonts
   - Still not at Google-Lens level
   - Python only

### Cloud APIs (No Training, Just Config)

- **Google Cloud Vision**: Excellent Hebrew support, very close to what Lens uses
- **Azure Computer Vision Read API**: Decent multilingual OCR, supports Hebrew
- **AWS Textract**: Multilingual OCR with Hebrew support
- **Pros**: Far better recognition out of the box
- **Cons**: Cost, network dependency, privacy concerns

## What This Means for Our Project

- For **per-glyph typography extraction**, no off-the-shelf OCR (including Google) is really optimized
- Even if Lens recognizes the *word* perfectly, it doesn't expose clean glyph bboxes/crops
- Switching from Tesseract to EasyOCR/Paddle/Cloud will likely:
  - Improve **word/line recognition** on full text images
  - Only **slightly** help with "symbol-level crops from arbitrary typography" unless we change segmentation strategy
- The path that scales best:
  - Use any engine (Tesseract or a detector) to **roughly locate letters/words**
  - Do **custom segmentation + padding** (like we're beginning to do)
  - Optionally build a small **custom classifier** later

## Follow-up Questions and Answers

### 1. Web vs Python Backend

**Question:** Web based is not a must, it's just convenient. We can have a python backend, like Pyodide.

**Answer:**
- **You don't need to stay web-only**
- A small Python backend (FastAPI/Flask) running on your machine with a browser UI is a very natural fit
- **Pyodide** is cool but adds complexity and size; if you're okay running a local server, a normal Python stack (Tesseract / EasyOCR / PaddleOCR) is simpler and faster

### 2. Two-Pass Tesseract Strategy

**Question:** Should we run tesseract twice for better recognition? Once to find words and once to find letters inside words?

**Answer:**
- Running Tesseract **twice** can help, but mostly as a better *segmenter*:
  1. **Pass 1 (words/lines):** Use `res.data.lines`/`words` to get clean text regions and word boxes; ignore the artwork/background
  2. **Pass 2 (symbols):** For each word box, crop that region yourself and run a second `recognize()` with `psm=7` or `10` on just that small strip, then take `symbols`
- **Benefit:** You constrain symbol segmentation to well-defined word regions instead of the whole page, which often reduces "between-letters" boxes
- **Worth trying** if we stay with Tesseract as the main engine

### 2.5. Robust Text Detection → Glyphs

**Question:** Thinking of this: robust text detection, first finding text regions/lines and then glyphs.

**Answer:**
- Conceptually ideal pipeline (no matter which recognizer you use):
  1. **Text detection**: Find text regions / lines (CRAFT, PaddleOCR detector, or Tesseract lines/words as a cheap version)
  2. **Line/word normalization**: Binarize, straighten, maybe resize
  3. **Glyph segmentation** inside each line/word (projection profiles / connected components, not necessarily Tesseract's symbols)
  4. **Glyph recognition** (Tesseract/EasyOCR/custom model)
- This "detector → segmenter → recognizer" split is closer to what Google and modern OCR stacks do, and is more flexible for typography use cases

### 3. Combining OCR Libraries

**Question:** Should we combine tesseract with one of the other python suggestions? Or use a combo of a few of the python ones?

**Answer:**
- **Reasonable combo:**
  - Use one engine with a good **detector** (e.g. PaddleOCR or EasyOCR) to get word/line boxes
  - Use your own segmentation for glyphs within those boxes
  - Use **Tesseract OR EasyOCR** as the recognizer for glyphs, not a big ensemble
- Running **multiple recognizers and ensembling** is usually overkill for this project
- Main gain will come from **better segmentation + normalization**, not from averaging 3 recognizers

### 4. Google Cloud Vision for Later

**Question:** Later we can think of switching to Google Cloud Vision. I assume they or others don't have a free api? What if it's for academia using ac email?

**Answer:**
- **Yes, it's paid**, but:
  - There is typically a **small free tier / trial credit**; beyond that you pay per image or per 1000 units
  - Having an academic email sometimes helps (research credits, grants), but you usually must apply via Google Cloud for Research or similar programs; it's not an automatic "free unlimited" tier
- **Realistic path:**
  - Build and iterate your **segmentation + UI + export** locally with open-source tools
  - Once the pipeline is solid, drop in **Cloud Vision as an optional recognizer**, used only when you really need its quality and are okay with sending images off-box

## Implementation Plan & Execution

### Decision: Python Backend with Two-Pass Tesseract

Based on the discussion above, we decided to:
1. Build a Python backend (FastAPI) running locally
2. Implement two-pass Tesseract strategy (words → symbols)
3. Implement full pipeline: Text Detection → Line/Word Normalization → Glyph Segmentation → Glyph Recognition
4. Keep browser-based Tesseract.js as fallback option
5. If results don't improve, consider combining Tesseract with other Python OCRs (EasyOCR/PaddleOCR)

### What Was Built

#### 1. Python Backend Structure (`backend/`)

**Files:**
- `requirements.txt`: FastAPI, uvicorn, pytesseract, opencv-python, pillow, numpy (+ optional easyocr)
- `ocr_engine.py`: Multi-strategy OCR engine
- `api.py`: FastAPI endpoints for OCR processing
- `README.md`: Setup and usage instructions

**Key Discovery — Why Single-Pass Failed:**
Tesseract's `image_to_data` with PSM 6 returns Hebrew words as single "symbol" entries at level 5 (e.g., `text='אבגדהוזחט'` with `conf=0.0`). It does NOT segment individual Hebrew characters. The 0.0 confidence is Tesseract honestly saying "I recognized this text but couldn't isolate individual characters."

**Fix — Multi-Strategy Pipeline:**
The engine now tries three strategies in order until one succeeds:

**Strategy 1 — `image_to_boxes` on full image (fast path):**
- `pytesseract.image_to_boxes()` uses Tesseract's box-file mode which often returns per-character bounding boxes even when `image_to_data` groups them
- If this produces ≥2 Hebrew chars, we use these results directly

**Strategy 2 — Two-pass (word regions → per-region extraction):**
- **Pass 1**: `image_to_data` PSM 6 → word bounding boxes (level 4) + reference text (level 5)
- **Pass 2a**: `image_to_boxes` PSM 7 (single text line) on each word region → per-char boxes
- **Pass 2b** (fallback): Connected-components segmentation + reference text matching or PSM 10
  - Finds individual ink blobs via `cv2.connectedComponentsWithStats`
  - Merges small components (dots, dagesh, niqqud) into nearest large neighbor
  - If CC count matches reference text length → assigns characters 1:1 (RTL order)
  - Otherwise → runs PSM 10 (single character) on each component

**Strategy 3 — Full-image connected components (last resort):**
- If no word regions found, runs CC + PSM 10 on the entire image

**Optional — EasyOCR cross-reference:**
- If installed, reads word regions with EasyOCR for better reference text
- Helps improve 1:1 matching in the CC strategy

**Also Fixed:**
- Hebrew whitelist was missing 5 letters (יכלמנ) — now includes all 27 (22 base + 5 final forms)
- All coordinates guaranteed native Python int (no numpy int64 JSON errors)
- Clear strategy-by-strategy logging for debugging

**Preprocessing Pipeline:**
- Grayscale conversion
- CLAHE contrast enhancement
- Otsu binarization for word regions

**Bounding Box Refinement:**
- Padding around detected bbox (`CROP_BOX_MARGIN = 4px`)
- Ink-based tightening using luminance threshold (`CROP_LUM_THR = 220`)
- Post-refine padding (`POST_REFINE_PADDING = 2px`)
- Minimum glyph size filter (`MIN_GLYPH_SIZE_PX = 8px`)
- Overlap deduplication (`OVERLAP_THRESHOLD = 0.45`)

**API Endpoints:**
- `POST /api/ocr/process`: Process single image → JSON with symbols, crops (base64), metadata
- `POST /api/ocr/process-batch`: Process multiple images in batch

#### 2. Frontend Integration

**Changes to `index.html`:**
- Checkbox: "Use Python backend (two-pass Tesseract)"
- Backend URL input (default: `http://localhost:8001`)

**Changes to `script.js`:**
- `runOCRBackend()` function calls the FastAPI backend
- `runOCR()` routes to backend or browser mode
- Backend mode maintains same UI/UX: gallery, ZIP downloads (by image, by char), TSV/JSON exports

**Dual Mode Support:**
- **Browser Mode** (default): Tesseract.js in browser (original)
- **Backend Mode**: Python multi-strategy engine (recommended)

#### 3. Pipeline Flow

```
Full Image
    │
    ├─ Strategy 1: image_to_boxes PSM 6
    │   └─ Returns per-char boxes? ──yes──→ Done ✓
    │                                no ↓
    ├─ Strategy 2: Two-Pass
    │   ├─ Pass 1: image_to_data PSM 6 → word regions + ref text
    │   │
    │   └─ For each word region:
    │       ├─ Pass 2a: image_to_boxes PSM 7 → per-char boxes?
    │       │   └─ yes → use these ✓
    │       │
    │       └─ Pass 2b: Connected Components
    │           ├─ Find ink blobs (merge dots into parents)
    │           ├─ CC count == ref text length? → assign 1:1 ✓
    │           └─ else → PSM 10 per blob ✓
    │
    └─ Strategy 3: CC on full image (last resort)
```

### How to Use

1. **Install Tesseract OCR:**
   - macOS: `brew install tesseract tesseract-lang`
   - Linux: `sudo apt-get install tesseract-ocr tesseract-ocr-heb`

2. **Install Backend Dependencies (conda recommended):**
```bash
conda create -n hebrew-ocr python=3.9
conda activate hebrew-ocr
conda install -c conda-forge fastapi uvicorn pillow numpy opencv
pip install pytesseract python-multipart
# Optional for EasyOCR cross-reference:
# pip install easyocr
```

3. **Start Backend Server:**
```bash
cd backend
python api.py
```

4. **Use Frontend:**
   - Open `index.html` in browser
   - Check "Use Python backend"
   - Verify backend URL (default: `http://localhost:8001`)
   - Select images and run OCR

### Configuration

All parameters are tunable at the top of `backend/ocr_engine.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CROP_BOX_MARGIN` | 4 | px padding around detected bbox |
| `POST_REFINE_PADDING` | 2 | px padding after ink-tightening |
| `CROP_LUM_THR` | 220 | Luminance threshold for ink detection (0–255) |
| `MIN_GLYPH_SIZE_PX` | 8 | Minimum glyph bbox width/height |
| `MIN_COMPONENT_AREA` | 40 | Minimum CC pixel area |
| `WORD_REGION_PADDING` | 6 | px padding when cropping word regions |
| `CONFIDENCE_FLOOR` | 40.0 | Default confidence when Tesseract returns 0 |
| `CC_SMALL_RATIO` | 0.20 | CCs below this fraction of median area → "dot" |
| `CC_MERGE_DISTANCE_X` | 20 | Max horizontal distance to merge dot into letter |
| `CC_MERGE_DISTANCE_Y` | 35 | Max vertical distance to merge dot into letter |
| `OVERLAP_THRESHOLD` | 0.45 | IoU threshold for deduplication |

### Next Steps if Results Don't Improve

1. **Enable EasyOCR**: Uncomment in `requirements.txt`, set `use_easyocr=True` in `api.py` — provides better reference text for CC matching
2. **Try PaddleOCR** for text detection (better word/line boxes)
3. **Vertical projection profiles** for glyph segmentation within word regions (handles touching letters)
4. **Google Cloud Vision** as optional recognizer (best quality, paid API)

---

## Improvement Round 1: Debug Confidence & Word Splitting

**Problem:** Tesseract `image_to_data` at level 5 returned entire words (e.g., `text='אבגדהוזחט'` with `conf=0.0`) instead of individual characters. Confidence was always 0 for the first two lines of Hebrew.

**Root Cause:** `image_to_boxes` / `image_to_data` PSM 6 groups Hebrew characters into word-level entries at level 5. It does NOT segment individual Hebrew characters — it returns the full word string with 0.0 confidence (Tesseract honestly saying "I recognized this text but couldn't isolate individual characters").

**Fix:** Updated extraction to detect when level 5 returns multi-character strings and split them into individual characters, estimating per-character bounding boxes by dividing the word bbox evenly.

**Outcome:** All characters now detected individually. Confidence remained at 0 for some entries (addressed in later rounds).

---

## Improvement Round 2: Full Multi-Strategy Pipeline Rewrite

**Problem:** The initial single-pass approach was fragile. Needed a robust pipeline that tries multiple strategies to maximize detection.

**What Changed:** Complete rewrite of `ocr_engine.py` to implement the three-strategy cascade:

- **Strategy 1 — `image_to_boxes` PSM 6 on full image (fast path):** Uses Tesseract's box-file mode which often returns per-character bounding boxes. If ≥2 Hebrew chars found, uses these directly.

- **Strategy 2 — Two-pass pipeline:**
  - **Pass 1:** `image_to_data` PSM 6 → word bounding boxes (level 4) + reference text (level 5)
  - **Pass 2a:** `image_to_boxes` PSM 7 (single text line) on each word region → per-char boxes
  - **Pass 2b (fallback):** Connected-components segmentation via `cv2.connectedComponentsWithStats`:
    - Finds individual ink blobs
    - Merges small components (dots, dagesh, niqqud) into nearest large neighbor
    - If CC count matches reference text length → assigns characters 1:1 (RTL order)
    - Otherwise → runs PSM 10 (single character) on each component

- **Strategy 3 — Full-image connected components (last resort):** If no word regions found, runs CC + PSM 10 on the entire image.

**Added PSM Documentation to file header:**
- PSM 6 = Assume a single uniform block of text (good for detecting words/lines)
- PSM 7 = Treat the image as a single text line (good for per-char extraction in a word)
- PSM 10 = Treat the image as a single character (good for recognizing one glyph)

**Outcome:** All 27 Hebrew characters (including 5 final forms) detected correctly on test images. Strategy 1 (fast path) succeeded on clean typography images.

---

## Improvement Round 3: Real Confidence Scoring

**Problem:** All detected characters showed confidence = 40 (the `CONFIDENCE_FLOOR` constant), because `image_to_boxes` doesn't return confidence scores — only character + bbox.

**Fix:** After `image_to_boxes` finds accurate bounding boxes, a new `_score_char_confidence()` method runs a quick PSM 10 `image_to_data` on each character crop to get Tesseract's actual confidence value.

**Trade-off:** Adds one PSM 10 call per character (small processing overhead) but provides meaningful confidence values for quality assessment.

**Outcome:** Real per-character confidence scores now visible (e.g., 85.2, 92.1) instead of flat 40s. Some characters still returned no score (addressed next).

---

## Improvement Round 4: Honest Confidence Labeling (N/A instead of fake scores)

**Problem:** Some characters still showed confidence = 40 despite the PSM 10 scoring. The arbitrary `CONFIDENCE_FLOOR` of 40 looked like a real score and was misleading.

**Fixes:**
1. **Try harder to get real confidence:** `_recognize_char()` now tries both binarized AND raw grayscale variants — picks whichever produces a real confidence score first.
2. **Honest "N/A" labeling:** Changed `CONFIDENCE_FLOOR` to `CONFIDENCE_UNSCORED = -1.0` as a sentinel value. Frontend displays this as **"N/A"** instead of a fake number.
3. **Only real scores shown:** If you see a number like `85.2`, it's Tesseract's actual confidence. If you see `N/A`, Tesseract couldn't score it (the detection came from `image_to_boxes` which has no confidence, and PSM 10 verification also failed).

**Outcome:** Clear distinction between scored and unscored characters. No more misleading confidence values.

---

## Improvement Round 5: Heavy Fonts & Latin Character Filtering

**Problem — Two issues identified during broader testing:**
1. **Heavy/bold fonts** cause misrecognition — thick strokes distort character shapes for Tesseract.
2. **Latin characters on mixed images** get force-mapped to Hebrew by the `tessedit_char_whitelist`, polluting detection folders with hundreds of misdetected characters (e.g., Latin "A" → nearest Hebrew letter).

**Fix for Latin noise (Issue 2) — Removed the whitelist approach entirely:**
- **Before:** `tessedit_char_whitelist` forced every detection to be Hebrew → Latin characters got mapped to wrong Hebrew chars
- **After:** Tesseract detects freely (no whitelist). Latin chars come through as Latin. The existing Hebrew Unicode regex filter (`\u0590-\u05FF`) drops non-Hebrew chars cleanly.
- Removed the `HEBREW_WHITELIST` constant entirely from `ocr_engine.py`
- Added `MIN_CHAR_CONFIDENCE = 15.0` — drops characters with real confidence below this threshold to further reduce noise

**Fix for heavy fonts (Issue 1) — Optional morphological thinning:**
- New parameter `ENABLE_THINNING = False` (set `True` for heavy/bold fonts)
- New parameter `THINNING_KERNEL_SIZE = 2` (try 2–3)
- When enabled, applies morphological erosion to thin heavy strokes before OCR processing
- Requires server restart after changing

**Console output now shows filtering steps:**
```
[OCR] Hebrew filter: kept 25/31 (dropped 6 non-Hebrew)
[OCR] Confidence filter (>=15.0): kept 24/25
```

**Outcome:** Mixed Hebrew+Latin images now cleanly extract only Hebrew characters. Heavy fonts can be handled by toggling thinning. Significantly reduced noise in detection output.

---

## Updated Configuration (after all rounds)

All parameters at the top of `backend/ocr_engine.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CROP_BOX_MARGIN` | 4 | px padding around detected bbox |
| `POST_REFINE_PADDING` | 2 | px padding after ink-tightening |
| `CROP_LUM_THR` | 220 | Luminance threshold for ink detection (0–255) |
| `MIN_GLYPH_SIZE_PX` | 8 | Minimum glyph bbox width/height |
| `MIN_COMPONENT_AREA` | 40 | Minimum CC pixel area |
| `WORD_REGION_PADDING` | 6 | px padding when cropping word regions |
| `CONFIDENCE_UNSCORED` | -1.0 | Sentinel for "no real confidence" (displayed as N/A) |
| `MIN_CHAR_CONFIDENCE` | 15.0 | Drop chars with real confidence below this |
| `CC_SMALL_RATIO` | 0.20 | CCs below this fraction of median area → "dot" |
| `CC_MERGE_DISTANCE_X` | 20 | Max horizontal distance to merge dot into letter |
| `CC_MERGE_DISTANCE_Y` | 35 | Max vertical distance to merge dot into letter |
| `MIN_CHARS_FOR_STRATEGY` | 2 | Minimum chars for a strategy to be accepted |
| `OVERLAP_THRESHOLD` | 0.45 | IoU threshold for deduplication |
| `TESSERACT_LANG` | 'heb' | Language model (use 'heb+eng' for mixed scripts) |
| `ENABLE_THINNING` | False | Morphological erosion for heavy/bold fonts |
| `THINNING_KERNEL_SIZE` | 2 | Erosion kernel size (larger = more thinning) |

## Improvement Round 6: Google Vision API Integration (Disabled by Default)

**Goal:** Have Google Cloud Vision available as an optional high-quality engine for select images, without disrupting the current Tesseract-based workflow.

**What was added:**
- **`ENABLE_GOOGLE_VISION = False`** flag at top of `ocr_engine.py` — clear gate, off by default
- **`process_with_google_vision()`** method in `HebrewOCREngine` — uses `google.cloud.vision.ImageAnnotatorClient` with Hebrew language hints
- **`POST /api/ocr/process-google-vision`** endpoint in `api.py` — same response format as the Tesseract endpoint
- **UI checkbox** "Use Google Cloud Vision API (requires setup)" — unchecked by default
- **`google-cloud-vision>=3.4.0`** in `requirements.txt` (commented out)

**To enable:**
1. `pip install google-cloud-vision`
2. Set `GOOGLE_APPLICATION_CREDENTIALS` env var to your service account JSON
3. Set `ENABLE_GOOGLE_VISION = True` in `ocr_engine.py`
4. Check both "Use Python backend" and "Use Google Cloud Vision API" in the UI

**Also changed:** Both UI checkboxes ("Keep only Hebrew" and "Use Python backend") now default to checked.

---

## Improvement Round 7: Single-Letter Crop Isolation

**Problem:** On complex posters and specimen images, two crop issues occurred:
1. **Multi-letter crops:** The crop around a correctly-detected single letter would show neighboring letters bleeding in.
2. **Shifted crops:** The crop was offset — showing half the detected letter and half the adjacent letter.

**Root Cause:** `refine_bbox_by_content()` found ALL ink pixels (below luminance threshold) within the padded region. When the `CROP_BOX_MARGIN` padding captured ink from neighboring letters, the refinement expanded or shifted the crop to include that neighbor ink.

**Fix — `isolate_main_component()` in `ocr_engine.py`:**
1. After expanding the detection bbox by `CROP_BOX_MARGIN`, run `cv2.connectedComponentsWithStats` on the padded region's ink mask
2. Find the connected component whose centroid is **closest to the center of the original detection bbox** — this is the actual detected letter
3. Merge nearby small CCs (dots, dagesh, niqqud) that likely belong to the same letter (using `CC_MERGE_DISTANCE_X/Y`)
4. Return only that component's tight bounding box
5. Apply `POST_REFINE_PADDING` for breathing room

Falls back to the old `refine_bbox_by_content()` if component isolation fails (e.g., no ink found).

**Outcome:** Each crop now isolates exactly one letter, anchored to the detection center. Neighbor ink is excluded even in tightly-spaced typography.

---

## Improvement Round 8: PDF & Multi-Page TIFF Support

**Problem:** Some specimen sources are multi-page PDFs or multi-page TIFFs. The backend only handled single-image files (JPEG, PNG, single-page TIFF).

**What was added:**
- **`pdf2image>=1.16.0`** in `requirements.txt` + `poppler` system dependency (via `brew install poppler`)
- **`split_to_page_images()`** helper in `api.py`:
  - **PDFs:** converts each page to PNG at 300 DPI via `pdf2image.convert_from_path()`
  - **Multi-page TIFFs:** iterates PIL frames (`pil_img.seek(i)`), saves each as PNG
  - **Single images:** pass through unchanged
- **Multi-page response format:** When input has multiple pages, the API returns `{ pages: [...], total_pages: N }` with per-page results. Single-page files return the same format as before (backward compatible).
- **Frontend handles both formats:** `runOCRBackend()` normalizes multi-page responses via `result.pages || [result]` and processes each page, labeling crops with page numbers.
- **File input accepts PDFs:** Updated `accept` attribute to include `.pdf,.PDF`

**Setup:**
```bash
conda activate hebrew-ocr
pip install pdf2image
brew install poppler   # macOS — provides pdftoppm
```

---

## Improvement Round 9: Color Crop Previews

**Problem:** The backend cropped from the preprocessed grayscale image, so all gallery previews were black and white. For visual inspection of detection quality on colorful posters/typography, color crops are more useful.

**What was added:**
- **Backend sends both crops:** `_generate_crops()` helper in `api.py` returns `image_data` (BW) and `image_data_color` (color from original BGR image) for every symbol
- **UI checkbox:** "Color crop previews (downloads always BW)" — checked by default
- **Gallery respects toggle:** When color checkbox is on, preview uses `image_data_color`; when off, uses `image_data` (BW)
- **Downloads always BW:** ZIP files use `image_data` regardless of the toggle — consistent for training data

**Also refactored:** Frontend code deduplicated into shared `processPageResult()` and `finalizeCharBuckets()` helpers, used by both the Tesseract and Google Vision code paths. Eliminated ~150 lines of duplicated logic.

---

## Current Status

- **Detection accuracy:** All 27 Hebrew characters (22 base + 5 final forms) detected correctly on clean typography
- **Single-letter isolation:** Crops anchored to the main connected component closest to detection center
- **Mixed-script images:** Latin and non-Hebrew characters cleanly filtered out via Unicode regex
- **Confidence scoring:** Real Tesseract confidence per character, with honest N/A for unscored
- **Heavy fonts:** Optional thinning toggle for bold/heavy typography
- **Multi-strategy cascade:** Automatically selects the best extraction approach per image
- **PDF & multi-page TIFF:** Automatically split into per-page processing
- **Color previews:** Gallery shows color crops by default, downloads always BW
- **Google Vision:** Available as optional engine (disabled by default, requires credentials)

---

## Remaining Improvement Branches (Before Declaring "Best Case")

These are concrete branches still worth exploring with the current architecture, roughly ordered by expected impact-to-effort ratio:

### Branch A: Adaptive Preprocessing Per Image Type (High Impact, Medium Effort)
The current preprocessing is one-size-fits-all: CLAHE + optional thinning. Different image types need different treatment:
- **Light fonts on dark backgrounds** — inversion before binarization
- **Low-contrast images** — more aggressive CLAHE or adaptive thresholding (instead of Otsu)
- **Colored text on colored backgrounds** — color-channel separation before grayscale
- **Noisy/textured backgrounds** — denoising (Gaussian/median blur) before binarization

**Approach:** Add an auto-detect step that inspects histogram/foreground stats and picks the best preprocessing path. Or expose a "preprocessing preset" dropdown in the UI.

### Branch B: Vertical Projection Profiles for Glyph Segmentation (Medium Impact, Medium Effort)
Connected-components work well for separated letters, but **touching/overlapping glyphs** (common in cursive, serif, or tight typography) need a different approach:
- Compute vertical projection profile (column-wise ink density) within each word region
- Find valleys (low-ink columns) as split points between characters
- This handles cases where CC merges two touching letters into one blob

### Branch C: EasyOCR as Primary Recognizer (Medium Impact, Low Effort)
Currently EasyOCR is only used for optional cross-reference text. Using it as the **primary recognizer** (instead of Tesseract PSM 10) for individual character crops could improve accuracy on stylized fonts, since EasyOCR's deep-learning model has seen more font variety.

**Approach:** In `_recognize_char()`, try EasyOCR first, fall back to Tesseract. Or run both and pick the higher-confidence result.

### Branch D: Image DPI / Resolution Normalization (Low-Medium Impact, Low Effort)
Tesseract expects ~300 DPI. Very high-res or very low-res images get worse results.
- Auto-detect resolution and resize to ~300 DPI equivalent before OCR
- For character crops sent to PSM 10, upscale small crops (e.g., below 32px) to a minimum size

### Branch E: Post-OCR Confusion Correction (Low Impact, Low Effort)
Some Hebrew characters are systematically confused (e.g., ד/ר, ו/ז, ב/כ, ח/ת).
- Track confusion patterns from the test set
- Apply rule-based corrections (e.g., if PSM 10 returns ר with low confidence but the crop shape is more square than tall, flip to ד)
- Or train a tiny classifier on the confusion pairs

### Branch F: Multi-Engine Ensemble Voting (Medium Impact, High Effort)
Run both Tesseract and EasyOCR (and optionally Google Vision) on the same character crop. Pick the result with highest confidence, or use majority voting if 2+ engines agree.

### When to Declare "Best Case"
Stop iterating when:
1. The **test score** (see Testing Methodology below) plateaus across 2+ consecutive changes
2. Remaining false positives are **image-quality issues** (blur, extreme distortion) rather than engine bugs
3. The effort to fix the remaining errors exceeds the effort to manually delete them

---

## Testing Methodology: OCR Benchmark Score

### Why
Random ad-hoc testing gives no baseline. We need a repeatable test that produces a numeric score, so we can measure whether each change is an improvement, regression, or neutral.

### Ground Truth Set: 10 Reference Images
Select 10 images that cover the range of difficulty:

| # | Image Type | Purpose |
|---|-----------|---------|
| 1 | Clean black text on white, standard font | Baseline — should be near-perfect |
| 2 | Clean black text, serif/decorative font | Standard + font variety |
| 3 | Bold/heavy font | Tests thinning |
| 4 | Light/thin font | Tests contrast enhancement |
| 5 | Mixed Hebrew + Latin text | Tests script filtering |
| 6 | Colored text on colored background | Tests preprocessing |
| 7 | Low resolution / small text | Tests DPI handling |
| 8 | Dense/tight letter spacing | Tests CC splitting |
| 9 | Poster/artwork with Hebrew | Tests real-world noise |
| 10 | Handwritten or highly stylized | Edge case — expected lower accuracy |

### Ground Truth Annotation
For each image, manually create a JSON file listing the **expected characters** and their approximate bounding boxes:

```json
{
  "image": "test_01_clean_standard.tiff",
  "expected_chars": ["א", "ב", "ג", "ד", "ה", "ו", "ז", "ח", "ט"],
  "total_expected": 9,
  "notes": "3 rows of 3 chars each, clean background"
}
```

### Scoring Formula

For each image, compute:
- **True Positives (TP):** OCR detected a correct Hebrew char that exists in the ground truth
- **False Positives (FP):** OCR detected something that isn't in the ground truth (wrong char or phantom detection)
- **False Negatives (FN):** A ground-truth char that OCR missed entirely

Then:
- **Precision** = TP / (TP + FP) — "how many detections are correct"
- **Recall** = TP / (TP + FN) — "how many real chars did we find"
- **F1** = 2 × (Precision × Recall) / (Precision + Recall) — balanced score

**Overall OCR Score** = average F1 across all 10 images (0–100 scale).

### Running the Test
A test script (`backend/test_benchmark.py`) that:
1. Reads the 10 images from a `test_images/` folder
2. Reads the ground-truth JSON files from `test_images/ground_truth/`
3. Runs the OCR pipeline on each
4. Compares detected chars vs expected chars (character-level matching)
5. Outputs a scorecard:

```
Image                      | TP  | FP  | FN  | Precision | Recall | F1
test_01_clean_standard     |  9  |  0  |  0  |   100.0%  | 100.0% | 100.0
test_05_mixed_hebrew_latin |  15 |  2  |  1  |    88.2%  |  93.8% |  90.9
...
─────────────────────────────────────────────────────────────
OVERALL                    |     |     |     |    91.3%  |  94.7% |  93.0
```

### How to Use It
1. Run benchmark **before** making a change → record score
2. Make the change
3. Run benchmark **after** → compare score
4. If score improved or held → keep the change
5. If score dropped → investigate which images regressed and why

---

## Image Count Estimation: How Many Source Images for ~100 Crops Per Character

### The Math

Hebrew has **27 characters** (22 base + 5 final forms: ך ם ן ף ץ).

The average yield per image depends on image type:
- **Alphabet reference images** (all 27 chars in rows): ~25 chars/image
- **Word/sentence images**: ~10–20 unique chars/image (some chars repeat, some are rare)
- **Real-world typography**: ~5–15 unique chars/image (highly variable)

**Key insight:** Not all characters appear equally. Common chars (א, ב, ה, ו, ל, מ, ר, ת) appear in almost every word. Rare chars (especially final forms: ך, ץ, ף) appear much less frequently.

### Estimates

| Source Type | Chars/image (avg) | Images for 100/char (common) | Images for 100/char (rare finals) |
|---|---|---|---|
| Alphabet sheets (all 27 in each) | 27 | ~100 | ~100 |
| Word/sentence typography | ~15 unique | ~150–200 | ~300–500 |
| Mixed real-world images | ~10 unique | ~200–300 | ~500–1000 |

### Practical Recommendation

- **Fastest path to 100/char:** Create **~100 alphabet reference images** in different fonts/styles. Each image contains all 27 characters → 100 images gives exactly 100 crops per char.
- **For variety:** Supplement with **~50–100 word/sentence images** to add natural context (different sizes, positions, neighboring-letter effects).
- **Total estimate:** **100–150 images** if they're mostly full-alphabet sheets, or **200–400 images** if they're natural word/sentence typography.

### Final-Form Characters (ך ם ן ף ץ)
These only appear at the end of words, so:
- In alphabet sheets: they appear once each → same as others
- In word images: they appear ~3× less often than common letters
- **Mitigation:** Include words that specifically end with each final form, or create dedicated sheets for finals

### Bottom Line
If you design images intentionally (alphabet sheets in varied fonts/styles), **~100–120 images** should close the chapter with 100+ crops per character folder, including the rare finals.

