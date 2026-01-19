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

## Next Steps

Consider sketching a concrete Python pipeline (Tesseract + simple word detection + existing glyph-cropping logic) that mirrors the current JS UI but with better control over passes 1 and 2.

