# Hebrew OCR Backend

Python backend implementing two-pass Tesseract OCR pipeline for Hebrew glyph extraction.

## Pipeline

1. **Text Detection (Pass 1)**: Detect words/lines using Tesseract with word-level PSM
2. **Line/Word Normalization**: Binarize and normalize each word region
3. **Glyph Segmentation (Pass 2)**: Extract symbols from each word using character-level PSM
4. **Glyph Recognition**: Refine bounding boxes and extract crops

## Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install Tesseract OCR:
   - **macOS**: `brew install tesseract tesseract-lang`
   - **Linux**: `sudo apt-get install tesseract-ocr tesseract-ocr-heb`
   - **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

3. Verify Tesseract installation:
```bash
tesseract --version
```

## Running

Start the FastAPI server:
```bash
python api.py
```

Or with uvicorn:
```bash
uvicorn api:app --reload --port 8001
```

The API will be available at `http://localhost:8001`

## API Endpoints

### POST `/api/ocr/process`
Process a single image.

**Request:**
- `file`: Image file (multipart/form-data)
- `only_hebrew`: Boolean (default: true)

**Response:**
```json
{
  "source_image": "filename.jpg",
  "width": 1920,
  "height": 1080,
  "count": 42,
  "words": [...],
  "symbols": [...],
  "crops": [...],
  "stats_by_char": [...]
}
```

### POST `/api/ocr/process-batch`
Process multiple images in batch.

**Request:**
- `files`: Array of image files
- `only_hebrew`: Boolean (default: true)

**Response:**
```json
{
  "results": [
    { ... },
    { ... }
  ]
}
```

## Configuration

Edit `ocr_engine.py` to adjust:
- `CROP_BOX_MARGIN`: Padding around Tesseract bbox
- `POST_REFINE_PADDING`: Padding after ink-tightening
- `CROP_LUM_THR`: Luminance threshold for ink detection
- `MIN_GLYPH_SIZE_PX`: Minimum glyph size filter

