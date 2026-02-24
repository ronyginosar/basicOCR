"""
OCR Engine with Two-Pass Tesseract Strategy
Pipeline: Text Detection → Line/Word Normalization → Glyph Segmentation → Glyph Recognition
"""
import cv2
import numpy as np
from PIL import Image
import pytesseract
from typing import List, Dict, Tuple, Optional
import re


class HebrewOCREngine:
    """Two-pass Tesseract OCR engine optimized for Hebrew glyph extraction"""
    
    # Hebrew Unicode range
    HEBREW_RANGE = '\u0590-\u05FF'
    HEBREW_REGEX = re.compile(f'[{HEBREW_RANGE}]')
    
    # Cropping parameters (matching frontend constants)
    CROP_BOX_MARGIN = 4
    POST_REFINE_PADDING = 2
    CROP_LUM_THR = 220
    MIN_GLYPH_SIZE_PX = 8
    
    def __init__(self, lang='heb', psm_word=6, psm_symbol=10):
        """
        Initialize OCR engine
        
        Args:
            lang: Tesseract language code (default: 'heb' for Hebrew)
            psm_word: Page segmentation mode for word/line detection (default: 6 = uniform block)
            psm_symbol: Page segmentation mode for symbol detection (default: 10 = single character)
        """
        self.lang = lang
        self.psm_word = psm_word
        self.psm_symbol = psm_symbol
        
        # Configure Tesseract for Hebrew
        self.tesseract_config_word = f'--psm {psm_word} -c tessedit_char_whitelist=אבגדהוזחטיכסעפצקרשתךםןףץ'
        self.tesseract_config_symbol = f'--psm {psm_symbol} -c tessedit_char_whitelist=אבגדהוזחטיכסעפצקרשתךםןףץ'
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image: convert to grayscale, enhance contrast, optionally binarize
        
        Args:
            image: Input image (BGR or RGB)
            
        Returns:
            Preprocessed grayscale image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Enhance contrast (CLAHE - Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        return enhanced
    
    def normalize_word_region(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Normalize a word/line region: crop, binarize, optionally deskew
        
        Args:
            image: Full image
            bbox: (x0, y0, x1, y1) bounding box
            
        Returns:
            Normalized word region image
        """
        x0, y0, x1, y1 = bbox
        # Add small padding
        h, w = image.shape[:2]
        x0 = max(0, x0 - 2)
        y0 = max(0, y0 - 2)
        x1 = min(w, x1 + 2)
        y1 = min(h, y1 + 2)
        
        word_region = image[y0:y1, x0:x1]
        
        # Binarize (Otsu's method)
        if len(word_region.shape) == 2:
            _, binary = cv2.threshold(word_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            gray = cv2.cvtColor(word_region, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def pass1_detect_words_lines(self, image: np.ndarray) -> List[Dict]:
        """
        Pass 1: Detect words/lines in the image
        
        Args:
            image: Preprocessed image
            
        Returns:
            List of word/line dictionaries with bbox and text
        """
        # Convert numpy array to PIL Image for pytesseract
        pil_image = Image.fromarray(image)
        
        # Get word-level data
        data = pytesseract.image_to_data(
            pil_image,
            lang=self.lang,
            config=self.tesseract_config_word,
            output_type=pytesseract.Output.DICT
        )
        
        words = []
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            text = data['text'][i].strip()
            conf = float(data['conf'][i])
            
            # Skip empty or low-confidence detections
            if not text or conf < 0:
                continue
            
            # Extract bounding box (convert to native Python int)
            x = int(data['left'][i])
            y = int(data['top'][i])
            w = int(data['width'][i])
            h = int(data['height'][i])
            
            words.append({
                'text': text,
                'confidence': float(conf),  # Ensure float, not numpy float
                'bbox': (x, y, x + w, y + h),  # (x0, y0, x1, y1) - all ints
                'level': 'word'
            })
        
        return words
    
    def pass1_extract_symbols_direct(self, image: np.ndarray) -> List[Dict]:
        """
        Extract symbols directly from full image (like browser Tesseract.js approach)
        This is the primary method for getting individual characters
        """
        pil_image = Image.fromarray(image)
        
        # Get symbol-level data directly from full image
        # Use PSM 6 (uniform block) with symbol-level output
        data = pytesseract.image_to_data(
            pil_image,
            lang=self.lang,
            config=f'--psm 6 -c tessedit_char_whitelist=אבגדהוזחטסעפצקרשתךםןףץ',
            output_type=pytesseract.Output.DICT
        )
        
        symbols = []
        n_boxes = len(data['text'])
        
        # Debug: count by level
        level_counts = {}
        for i in range(n_boxes):
            level = int(data['level'][i])
            level_counts[level] = level_counts.get(level, 0) + 1
        print(f"Tesseract returned {n_boxes} boxes: level distribution {level_counts}")
        
        for i in range(n_boxes):
            text = data['text'][i].strip()
            conf = float(data['conf'][i])
            level = int(data['level'][i])
            
            # Extract bounding box
            x = int(data['left'][i])
            y = int(data['top'][i])
            w = int(data['width'][i])
            h = int(data['height'][i])
            
            # Debug first few level 4 and 5 boxes
            if level in [4, 5] and len(symbols) < 10:
                print(f"  Level {level}: text='{text}' conf={conf} bbox=({x},{y},{x+w},{y+h}) size={w}x{h}")
            
            # Skip empty or low-confidence detections
            if not text or conf < 0:
                if level in [4, 5]:
                    print(f"  Skipped level {level} box {i}: empty text or low conf")
                continue
            
            # Size filter
            if w < self.MIN_GLYPH_SIZE_PX or h < self.MIN_GLYPH_SIZE_PX:
                if level in [4, 5]:
                    print(f"  Skipped level {level} box {i}: too small ({w}x{h} < {self.MIN_GLYPH_SIZE_PX})")
                continue
            
            # Process based on level
            if level == 5:
                # Symbol-level: accept single characters
                if len(text) == 1:
                    symbols.append({
                        'text': text,
                        'confidence': float(conf),
                        'bbox': (x, y, x + w, y + h),
                        'level': 'symbol'
                    })
                else:
                    print(f"  Skipped level 5 box: text length {len(text)} (not single char): '{text}'")
            elif level == 4:
                # Word-level: split into individual characters
                if len(text) > 0:
                    char_width = w / len(text) if len(text) > 0 else w
                    for char_idx, char in enumerate(text):
                        char_x = int(x + char_idx * char_width)
                        char_w = int(char_width)
                        symbols.append({
                            'text': char,
                            'confidence': float(conf),
                            'bbox': (char_x, y, char_x + char_w, y + h),
                            'level': 'symbol'
                        })
        
        print(f"Extracted {len(symbols)} symbols from Tesseract data")
        return symbols
    
    def pass2_extract_symbols(self, image: np.ndarray, word_bbox: Tuple[int, int, int, int]) -> List[Dict]:
        """
        Pass 2: Extract symbols (glyphs) from a word/line region
        
        Args:
            image: Full preprocessed image
            word_bbox: (x0, y0, x1, y1) bounding box of the word
            
        Returns:
            List of symbol dictionaries with bbox and text
        """
        # Normalize the word region
        word_region = self.normalize_word_region(image, word_bbox)
        
        # Check if word region is too small
        if word_region.shape[0] < 10 or word_region.shape[1] < 10:
            return []
        
        # Convert to PIL Image
        pil_word = Image.fromarray(word_region)
        
        # Get symbol-level data with single character PSM
        data = pytesseract.image_to_data(
            pil_word,
            lang=self.lang,
            config=self.tesseract_config_symbol,
            output_type=pytesseract.Output.DICT
        )
        
        symbols = []
        n_boxes = len(data['text'])
        
        # Get word region offset for absolute coordinates
        x0_word, y0_word, x1_word, y1_word = word_bbox
        
        for i in range(n_boxes):
            text = data['text'][i].strip()
            conf = float(data['conf'][i])
            level = int(data['level'][i])  # 5 = symbol, 4 = word, etc.
            
            # Only process symbol-level detections (level 5)
            # Skip word-level or higher (we want individual characters)
            if level != 5:
                continue
            
            # Skip empty or low-confidence detections
            if not text or conf < 0:
                continue
            
            # Extract bounding box (relative to word region)
            x_rel = int(data['left'][i])
            y_rel = int(data['top'][i])
            w = int(data['width'][i])
            h = int(data['height'][i])
            
            # Convert to absolute coordinates
            x0 = int(x0_word + x_rel)
            y0 = int(y0_word + y_rel)
            x1 = int(x0 + w)
            y1 = int(y0 + h)
            
            # Size filter
            if w < self.MIN_GLYPH_SIZE_PX or h < self.MIN_GLYPH_SIZE_PX:
                continue
            
            # Only add if text is a single character (or very short)
            # Tesseract sometimes groups characters, so filter those out
            if len(text) > 1:
                # If it's multiple characters, skip it (we want individual glyphs)
                continue
            
            symbols.append({
                'text': text,
                'confidence': float(conf),  # Ensure float, not numpy float
                'bbox': (x0, y0, x1, y1),  # Absolute coordinates (all ints)
                'level': 'symbol'
            })
        
        return symbols
    
    def refine_bbox_by_content(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[Tuple[int, int, int, int]]:
        """
        Tighten bounding box to actual ink content (matching frontend logic)
        
        Args:
            image: Grayscale image
            bbox: (x0, y0, x1, y1) initial bounding box
            
        Returns:
            Refined (x0, y0, x1, y1) or None if no ink found
        """
        x0, y0, x1, y1 = bbox
        w = max(1, x1 - x0)
        h = max(1, y1 - y0)
        
        # Extract region
        region = image[y0:y1, x0:x1]
        if region.size == 0:
            return None
        
        # Find ink pixels (below threshold)
        ink_mask = region < self.CROP_LUM_THR
        
        if not np.any(ink_mask):
            return None
        
        # Find bounding box of ink
        rows = np.any(ink_mask, axis=1)
        cols = np.any(ink_mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return None
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        # Convert numpy int64 to native Python int
        return (int(x0 + x_min), int(y0 + y_min), int(x0 + x_max + 1), int(y0 + y_max + 1))
    
    def expand_box(self, bbox: Tuple[int, int, int, int], margin: int, max_w: int, max_h: int) -> Tuple[int, int, int, int]:
        """Expand bounding box by margin, clamped to image bounds"""
        x0, y0, x1, y1 = bbox
        return (
            int(max(0, x0 - margin)),
            int(max(0, y0 - margin)),
            int(min(max_w, x1 + margin)),
            int(min(max_h, y1 + margin))
        )
    
    def process_image(self, image_path: str, only_hebrew: bool = True) -> Dict:
        """
        Full pipeline: Process image through two-pass OCR
        
        Args:
            image_path: Path to input image
            only_hebrew: Filter to only Hebrew characters
            
        Returns:
            Dictionary with metadata, words, and symbols
        """
        # Load and preprocess image
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        img_preprocessed = self.preprocess_image(img_bgr)
        img_h, img_w = img_preprocessed.shape[:2]
        
        # Primary method: Extract symbols directly from full image (like browser version)
        # This is more reliable for getting individual characters
        all_symbols = self.pass1_extract_symbols_direct(img_preprocessed)
        print(f"Direct symbol extraction found {len(all_symbols)} symbols")
        
        # Optional: Also detect words for reference (not used for symbol extraction)
        words = self.pass1_detect_words_lines(img_preprocessed)
        print(f"Pass 1 (reference) found {len(words)} words/lines")
        
        # Filter Hebrew if requested
        if only_hebrew:
            before = len(all_symbols)
            all_symbols = [s for s in all_symbols if self.HEBREW_REGEX.search(s['text'])]
            print(f"After Hebrew filter: {len(all_symbols)}/{before} symbols")
        
        # Refine bounding boxes and prepare final crops
        refined_symbols = []
        for sym in all_symbols:
            # Expand box with margin
            expanded = self.expand_box(
                sym['bbox'],
                self.CROP_BOX_MARGIN,
                img_w,
                img_h
            )
            
            # Refine to ink content
            refined = self.refine_bbox_by_content(img_preprocessed, expanded)
            if refined is None:
                refined = expanded
            
            # Add post-refine padding
            final_bbox = self.expand_box(
                refined,
                self.POST_REFINE_PADDING,
                img_w,
                img_h
            )
            
            # Ensure bbox is a tuple, not dict
            refined_symbols.append({
                'text': sym['text'],
                'confidence': sym['confidence'],
                'bbox': final_bbox,  # Tuple (x0, y0, x1, y1)
                'bbox_original': sym['bbox']
            })
        
        # Calculate stats
        char_stats = {}
        for sym in refined_symbols:
            char = sym['text']
            if char not in char_stats:
                char_stats[char] = {
                    'count': 0,
                    'sum_conf': 0,
                    'min_conf': 100,
                    'max_conf': 0
                }
            stats = char_stats[char]
            conf = sym['confidence']
            stats['count'] += 1
            stats['sum_conf'] += conf
            stats['min_conf'] = min(stats['min_conf'], conf)
            stats['max_conf'] = max(stats['max_conf'], conf)
        
        return {
            'source_image': image_path,
            'width': int(img_w),
            'height': int(img_h),
            'count': len(refined_symbols),
            'words': words,
            'symbols': refined_symbols,
            'stats_by_char': [
                {
                    'char': ch,
                    'count': int(s['count']),
                    'avg_conf': float(s['sum_conf'] / s['count']),
                    'min_conf': float(s['min_conf']),
                    'max_conf': float(s['max_conf'])
                }
                for ch, s in char_stats.items()
            ]
        }

