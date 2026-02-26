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

