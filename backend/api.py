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
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
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

