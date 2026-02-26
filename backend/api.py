"""
FastAPI backend for Hebrew OCR processing.
Serves the multi-strategy OCR engine via REST API.
"""
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import tempfile
import shutil
from typing import List
from ocr_engine import HebrewOCREngine
import cv2
import numpy as np
import base64
import traceback


def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    return obj


app = FastAPI(title="Hebrew OCR API")

# Allow browser frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OCR engine (set use_easyocr=True if you installed easyocr)
ocr_engine = HebrewOCREngine(use_easyocr=False)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Hebrew OCR API", "status": "running"}


@app.post("/api/ocr/process")
async def process_image(
    file: UploadFile = File(...),
    only_hebrew: bool = Form(True)
):
    """
    Process a single image through the multi-strategy OCR pipeline.
    Returns JSON with metadata, symbols, crops (base64), and stats.
    """
    # Determine file extension from filename or content-type
    original_ext = os.path.splitext(file.filename)[1] if file.filename else ''
    if not original_ext and hasattr(file, 'content_type'):
        ct = file.content_type or ''
        if 'jpeg' in ct or 'jpg' in ct:
            original_ext = '.jpg'
        elif 'png' in ct:
            original_ext = '.png'
        elif 'tiff' in ct:
            original_ext = '.tiff'
    if not original_ext:
        original_ext = '.jpg'

    print(f"\n[API] Processing: {file.filename} (type={getattr(file, 'content_type', '?')}), ext={original_ext}")

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=original_ext) as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_path = tmp_file.name

    try:
        # Validate image can be loaded
        img_bgr = cv2.imread(tmp_path)
        if img_bgr is None:
            raise ValueError(f"Could not load image: {tmp_path} (format may be unsupported)")

        # Run OCR pipeline
        result = ocr_engine.process_image(tmp_path, only_hebrew=only_hebrew)

        # Use original image (not preprocessed) for crop extraction
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr
        img_h, img_w = img_gray.shape[:2]

        # Generate base64-encoded crop images for each symbol
        crops = []
        for i, sym in enumerate(result['symbols']):
            bbox = sym['bbox']
            x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            # Clamp to image bounds
            x0 = max(0, min(x0, img_w - 1))
            y0 = max(0, min(y0, img_h - 1))
            x1 = max(x0 + 1, min(x1, img_w))
            y1 = max(y0 + 1, min(y1, img_h))

            crop = img_gray[y0:y1, x0:x1]
            if crop.size == 0:
                continue

            _, buffer = cv2.imencode('.png', crop)
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            crops.append({
                'index': i,
                'text': sym['text'],
                'confidence': sym['confidence'],
                'method': sym.get('method', 'unknown'),
                'bbox': {
                    'left': x0,
                    'top': y0,
                    'width': x1 - x0,
                    'height': y1 - y0
                },
                'image_data': f"data:image/png;base64,{img_base64}"
            })

        print(f"[API] Generated {len(crops)} crops from {len(result['symbols'])} symbols")

        result['crops'] = crops
        result['source_image'] = file.filename

        # Ensure JSON-safe types
        result = convert_numpy_types(result)
        return JSONResponse(content=result)

    except Exception as e:
        error_detail = {
            "error": str(e),
            "type": type(e).__name__,
            "traceback": traceback.format_exc()
        }
        print(f"[API] ERROR: {error_detail}")
        raise HTTPException(status_code=500, detail=error_detail)

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/api/ocr/process-batch")
async def process_batch(
    files: List[UploadFile] = File(...),
    only_hebrew: bool = Form(True)
):
    """Process multiple images in batch. Returns list of per-image results."""
    results = []

    for file in files:
        ext = os.path.splitext(file.filename)[1] if file.filename else '.jpg'
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name

        try:
            result = ocr_engine.process_image(tmp_path, only_hebrew=only_hebrew)

            img_bgr = cv2.imread(tmp_path)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            img_h, img_w = img_gray.shape[:2]

            crops = []
            for i, sym in enumerate(result['symbols']):
                x0, y0, x1, y1 = sym['bbox']
                x0 = max(0, min(int(x0), img_w - 1))
                y0 = max(0, min(int(y0), img_h - 1))
                x1 = max(x0 + 1, min(int(x1), img_w))
                y1 = max(y0 + 1, min(int(y1), img_h))

                crop = img_gray[y0:y1, x0:x1]
                if crop.size == 0:
                    continue

                _, buffer = cv2.imencode('.png', crop)
                img_base64 = base64.b64encode(buffer).decode('utf-8')

                crops.append({
                    'index': i,
                    'text': sym['text'],
                    'confidence': sym['confidence'],
                    'method': sym.get('method', 'unknown'),
                    'bbox': {'left': x0, 'top': y0, 'width': x1 - x0, 'height': y1 - y0},
                    'image_data': f"data:image/png;base64,{img_base64}"
                })

            result['crops'] = crops
            result['source_image'] = file.filename
            result = convert_numpy_types(result)
            results.append(result)

        except Exception as e:
            results.append({'source_image': file.filename, 'error': str(e)})

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return JSONResponse(content={'results': results})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
