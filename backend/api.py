"""
FastAPI backend for Hebrew OCR processing.
Serves the multi-strategy OCR engine via REST API.
Supports JPEG, PNG, TIFF (multi-page), and PDF (multi-page) inputs.
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
from PIL import Image as PILImage
import base64
import traceback

# PDF → image conversion (requires poppler installed on system)
try:
    from pdf2image import convert_from_path
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False


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


def split_to_page_images(file_path: str, ext: str) -> List[str]:
    """
    Convert a file (PDF, multi-page TIFF, or single image) into a list of
    single-page image file paths. Caller must delete the returned temp files.
    """
    ext_lower = ext.lower()

    # PDF → one PNG per page
    if ext_lower == '.pdf':
        if not PDF_SUPPORT:
            raise ValueError("PDF support requires pdf2image + poppler. Install: pip install pdf2image && brew install poppler")
        pages = convert_from_path(file_path, dpi=300)
        paths = []
        for i, page in enumerate(pages):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            page.save(tmp.name, 'PNG')
            tmp.close()
            paths.append(tmp.name)
            print(f"  [PDF] page {i+1}/{len(pages)} → {tmp.name}")
        return paths

    # Multi-page TIFF → one PNG per page
    if ext_lower in ('.tif', '.tiff'):
        pil_img = PILImage.open(file_path)
        n_frames = getattr(pil_img, 'n_frames', 1)
        if n_frames > 1:
            paths = []
            for i in range(n_frames):
                pil_img.seek(i)
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                pil_img.save(tmp.name, 'PNG')
                tmp.close()
                paths.append(tmp.name)
                print(f"  [TIFF] frame {i+1}/{n_frames} → {tmp.name}")
            return paths

    # Single image — return as-is
    return [file_path]


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


def _generate_crops(img_bgr: np.ndarray, symbols: list) -> list:
    """Generate both color and BW base64 crop images for each symbol."""
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr
    img_h, img_w = img_gray.shape[:2]

    crops = []
    for i, sym in enumerate(symbols):
        bbox = sym['bbox']
        x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        x0 = max(0, min(x0, img_w - 1))
        y0 = max(0, min(y0, img_h - 1))
        x1 = max(x0 + 1, min(x1, img_w))
        y1 = max(y0 + 1, min(y1, img_h))

        crop_bw = img_gray[y0:y1, x0:x1]
        if crop_bw.size == 0:
            continue

        _, buf_bw = cv2.imencode('.png', crop_bw)
        bw_b64 = base64.b64encode(buf_bw).decode('utf-8')

        # Color crop from original BGR image
        crop_color = img_bgr[y0:y1, x0:x1]
        _, buf_color = cv2.imencode('.png', crop_color)
        color_b64 = base64.b64encode(buf_color).decode('utf-8')

        crops.append({
            'index': i,
            'text': sym['text'],
            'confidence': sym['confidence'],
            'method': sym.get('method', 'unknown'),
            'bbox': {'left': x0, 'top': y0, 'width': x1 - x0, 'height': y1 - y0},
            'image_data': f"data:image/png;base64,{bw_b64}",
            'image_data_color': f"data:image/png;base64,{color_b64}"
        })
    return crops


@app.post("/api/ocr/process")
async def process_image(
    file: UploadFile = File(...),
    only_hebrew: bool = Form(True)
):
    """
    Process a single image (or multi-page PDF/TIFF) through the OCR pipeline.
    Multi-page inputs are split into per-page results.
    Returns JSON with symbols, crops (color + BW), and stats.
    """
    original_ext = os.path.splitext(file.filename)[1] if file.filename else ''
    if not original_ext and hasattr(file, 'content_type'):
        ct = file.content_type or ''
        if 'jpeg' in ct or 'jpg' in ct:
            original_ext = '.jpg'
        elif 'png' in ct:
            original_ext = '.png'
        elif 'tiff' in ct:
            original_ext = '.tiff'
        elif 'pdf' in ct:
            original_ext = '.pdf'
    if not original_ext:
        original_ext = '.jpg'

    print(f"\n[API] Processing: {file.filename} (type={getattr(file, 'content_type', '?')}), ext={original_ext}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=original_ext) as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_path = tmp_file.name

    page_paths = []
    try:
        # Split multi-page files into individual page images
        page_paths = split_to_page_images(tmp_path, original_ext)
        is_multipage = len(page_paths) > 1

        all_page_results = []
        for page_idx, page_path in enumerate(page_paths):
            img_bgr = cv2.imread(page_path)
            if img_bgr is None:
                raise ValueError(f"Could not load image: {page_path}")

            result = ocr_engine.process_image(page_path, only_hebrew=only_hebrew)
            crops = _generate_crops(img_bgr, result['symbols'])

            print(f"[API] Page {page_idx+1}: {len(crops)} crops from {len(result['symbols'])} symbols")

            result['crops'] = crops
            result['source_image'] = file.filename
            result['page'] = page_idx + 1
            result = convert_numpy_types(result)
            all_page_results.append(result)

        # Single page → return directly; multi-page → wrap in pages array
        if not is_multipage:
            return JSONResponse(content=all_page_results[0])
        return JSONResponse(content={
            'source_image': file.filename,
            'total_pages': len(all_page_results),
            'pages': all_page_results
        })

    except Exception as e:
        error_detail = {
            "error": str(e),
            "type": type(e).__name__,
            "traceback": traceback.format_exc()
        }
        print(f"[API] ERROR: {error_detail}")
        raise HTTPException(status_code=500, detail=error_detail)

    finally:
        # Clean up all temp files
        for p in page_paths:
            if p != tmp_path and os.path.exists(p):
                os.unlink(p)
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/api/ocr/process-google-vision")
async def process_google_vision(
    file: UploadFile = File(...),
    only_hebrew: bool = Form(True)
):
    """
    Process a single image using Google Cloud Vision API.
    Requires ENABLE_GOOGLE_VISION=True in ocr_engine.py and google-cloud-vision installed.
    """
    original_ext = os.path.splitext(file.filename)[1] if file.filename else '.jpg'
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

    print(f"\n[API] Google Vision: {file.filename} (type={getattr(file, 'content_type', '?')})")

    with tempfile.NamedTemporaryFile(delete=False, suffix=original_ext) as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_path = tmp_file.name

    try:
        result = ocr_engine.process_with_google_vision(tmp_path, only_hebrew=only_hebrew)

        img_bgr = cv2.imread(tmp_path)
        crops = _generate_crops(img_bgr, result['symbols'])

        result['crops'] = crops
        result['source_image'] = file.filename
        result = convert_numpy_types(result)
        return JSONResponse(content=result)

    except Exception as e:
        error_detail = {
            "error": str(e),
            "type": type(e).__name__,
            "traceback": traceback.format_exc()
        }
        print(f"[API] Google Vision ERROR: {error_detail}")
        raise HTTPException(status_code=500, detail=error_detail)

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/api/ocr/process-batch")
async def process_batch(
    files: List[UploadFile] = File(...),
    only_hebrew: bool = Form(True)
):
    """Process multiple images/PDFs in batch. Returns list of per-image results."""
    results = []

    for file in files:
        ext = os.path.splitext(file.filename)[1] if file.filename else '.jpg'
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name

        page_paths = []
        try:
            page_paths = split_to_page_images(tmp_path, ext)
            for page_idx, page_path in enumerate(page_paths):
                img_bgr = cv2.imread(page_path)
                if img_bgr is None:
                    continue
                result = ocr_engine.process_image(page_path, only_hebrew=only_hebrew)
                crops = _generate_crops(img_bgr, result['symbols'])
                result['crops'] = crops
                result['source_image'] = file.filename
                result['page'] = page_idx + 1
                result = convert_numpy_types(result)
                results.append(result)

        except Exception as e:
            results.append({'source_image': file.filename, 'error': str(e)})

        finally:
            for p in page_paths:
                if p != tmp_path and os.path.exists(p):
                    os.unlink(p)
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return JSONResponse(content={'results': results})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
