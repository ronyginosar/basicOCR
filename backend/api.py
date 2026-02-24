"""
FastAPI backend for Hebrew OCR processing
"""
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import tempfile
import shutil
from typing import List, Optional
from ocr_engine import HebrewOCREngine
import cv2
import numpy as np
from PIL import Image
import base64
import io
import json


def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization"""
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

# CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OCR engine
ocr_engine = HebrewOCREngine()


@app.get("/")
async def root():
    return {"message": "Hebrew OCR API", "status": "running"}


@app.post("/api/ocr/process")
async def process_image(
    file: UploadFile = File(...),
    only_hebrew: bool = Form(True)
):
    """
    Process a single image through two-pass OCR pipeline
    
    Returns:
        JSON with metadata, words, symbols, and crop data
    """
    # Get file extension from original filename, default to .jpg if missing
    # Note: Browser may send files with temp names, so we check content type too
    original_ext = os.path.splitext(file.filename)[1] if file.filename else ''
    if not original_ext and hasattr(file, 'content_type'):
        # Infer from content type
        if 'jpeg' in file.content_type or 'jpg' in file.content_type:
            original_ext = '.jpg'
        elif 'png' in file.content_type:
            original_ext = '.png'
        elif 'tiff' in file.content_type:
            original_ext = '.tiff'
    if not original_ext:
        original_ext = '.jpg'  # Default extension
    
    print(f"Processing file: {file.filename} (content-type: {getattr(file, 'content_type', 'unknown')}), using extension: {original_ext}")
    
    # Save uploaded file temporarily with correct extension
    with tempfile.NamedTemporaryFile(delete=False, suffix=original_ext) as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_path = tmp_file.name
    
    print(f"Saved temp file to: {tmp_path}")  # Debug log
    
    try:
        # Load image first to check if it's valid
        img_bgr = cv2.imread(tmp_path)
        if img_bgr is None:
            raise ValueError(f"Could not load image: {tmp_path}. File may be corrupted or unsupported format.")
        
        # Process image
        result = ocr_engine.process_image(tmp_path, only_hebrew=only_hebrew)
        
        # Convert to grayscale for crop extraction
        if len(img_bgr.shape) == 3:
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img_bgr
        
        # Generate crop images (base64 encoded)
        crops = []
        img_h, img_w = img_gray.shape[:2]
        print(f"Image size: {img_w}x{img_h}, processing {len(result['symbols'])} symbols")
        
        for i, sym in enumerate(result['symbols']):
            # bbox is a tuple (x0, y0, x1, y1)
            bbox = sym['bbox']
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                x0, y0, x1, y1 = bbox
            elif isinstance(bbox, dict):
                x0 = bbox.get('left', bbox.get('x0', 0))
                y0 = bbox.get('top', bbox.get('y0', 0))
                x1 = x0 + bbox.get('width', bbox.get('x1', 0) - x0)
                y1 = y0 + bbox.get('height', bbox.get('y1', 0) - y0)
            else:
                print(f"Warning: Unexpected bbox format for symbol {i}: {bbox}")
                continue
            
            # Ensure valid coordinates
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            w = x1 - x0
            h = y1 - y0
            
            # Validate bbox
            if w <= 0 or h <= 0 or x0 < 0 or y0 < 0:
                print(f"Warning: Invalid bbox for symbol {i} ('{sym['text']}'): ({x0}, {y0}, {x1}, {y1})")
                continue
            
            # Clamp to image bounds
            x0 = max(0, min(x0, img_w - 1))
            y0 = max(0, min(y0, img_h - 1))
            x1 = max(x0 + 1, min(x1, img_w))
            y1 = max(y0 + 1, min(y1, img_h))
            
            # Debug: print first few bboxes
            if i < 5:
                print(f"Symbol {i} ('{sym['text']}'): bbox=({x0}, {y0}, {x1}, {y1}), size={x1-x0}x{y1-y0}")
            
            # Extract crop
            crop = img_gray[y0:y1, x0:x1]
            
            if crop.size == 0:
                print(f"Warning: Empty crop for symbol {i} ('{sym['text']}')")
                continue
            
            # Convert to base64
            _, buffer = cv2.imencode('.png', crop)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            crops.append({
                'index': i,
                'text': sym['text'],
                'confidence': sym['confidence'],
                'bbox': {
                    'left': x0,
                    'top': y0,
                    'width': x1 - x0,
                    'height': y1 - y0
                },
                'image_data': f"data:image/png;base64,{img_base64}"
            })
        
        print(f"Generated {len(crops)} crop images from {len(result['symbols'])} symbols")
        
        result['crops'] = crops
        result['source_image'] = file.filename
        
        # Convert numpy types to native Python types for JSON serialization
        result = convert_numpy_types(result)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        import traceback
        error_detail = {
            "error": str(e),
            "type": type(e).__name__,
            "traceback": traceback.format_exc()
        }
        print(f"ERROR processing image: {error_detail}")  # Log to server console
        raise HTTPException(status_code=500, detail=error_detail)
    
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/api/ocr/process-batch")
async def process_batch(
    files: List[UploadFile] = File(...),
    only_hebrew: bool = Form(True)
):
    """
    Process multiple images in batch
    
    Returns:
        List of results, one per image
    """
    results = []
    
    for file in files:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
        
        try:
            # Process image
            result = ocr_engine.process_image(tmp_path, only_hebrew=only_hebrew)
            
            # Load image for crop extraction
            img_bgr = cv2.imread(tmp_path)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
            # Generate crop images (base64 encoded)
            crops = []
            for i, sym in enumerate(result['symbols']):
                x0, y0, x1, y1 = sym['bbox']
                crop = img_gray[y0:y1, x0:x1]
                
                # Convert to base64
                _, buffer = cv2.imencode('.png', crop)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                crops.append({
                    'index': i,
                    'text': sym['text'],
                    'confidence': sym['confidence'],
                    'bbox': {
                        'left': x0,
                        'top': y0,
                        'width': x1 - x0,
                        'height': y1 - y0
                    },
                    'image_data': f"data:image/png;base64,{img_base64}"
                })
            
            result['crops'] = crops
            result['source_image'] = file.filename
            
            # Convert numpy types to native Python types for JSON serialization
            result = convert_numpy_types(result)
            results.append(result)
        
        except Exception as e:
            results.append({
                'source_image': file.filename,
                'error': str(e)
            })
        
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    return JSONResponse(content={'results': results})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

