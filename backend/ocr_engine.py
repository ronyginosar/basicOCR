"""
Hebrew OCR Engine — Multi-strategy glyph extraction
====================================================
PSM = Page Segmentation Mode — tells Tesseract how to interpret the image layout.
Key modes used here:
  PSM 6  – Assume a single uniform block of text (good for detecting words/lines)
  PSM 7  – Treat the image as a single text line (good for per-char extraction in a word)
  PSM 10 – Treat the image as a single character (good for recognizing one glyph)

Pipeline (tries strategies in order until one succeeds):
  Strategy 1 – image_to_boxes PSM 6 on full image (fast path)
  Strategy 2 – Two-pass:
      Pass 1: Tesseract PSM 6 → word/line regions + reference text
      Pass 2a: image_to_boxes PSM 7 per word region → individual chars
      Pass 2b: Connected-components segmentation + reference text matching / PSM 10
  Strategy 3 – Connected components on full image (last resort)
  Optional  – EasyOCR cross-reference (if installed)
"""

import cv2
import numpy as np
from PIL import Image
import pytesseract
from typing import List, Dict, Tuple, Optional
import re


# ──────────────────────────────────────────────────────
# TUNEABLE PARAMETERS — adjust these to taste
# ──────────────────────────────────────────────────────

CROP_BOX_MARGIN = 4            # px padding around detected bbox (pre-refine)
POST_REFINE_PADDING = 2        # px padding after ink-tightening
CROP_LUM_THR = 220             # 0..255 luminance threshold for ink detection
MIN_GLYPH_SIZE_PX = 8          # minimum width/height to keep a glyph bbox
MIN_COMPONENT_AREA = 40        # minimum pixel area for a connected component
WORD_REGION_PADDING = 6        # px padding when cropping word regions for Pass 2
CONFIDENCE_UNSCORED = -1.0     # sentinel: "no real confidence available" (displayed as N/A)
MIN_CHAR_CONFIDENCE = 15.0     # drop chars with real confidence below this (reduces noise)
CC_SMALL_RATIO = 0.20          # CCs smaller than this fraction of median → "dot/niqqud"
CC_MERGE_DISTANCE_X = 20       # max horizontal px to merge a small CC into a large one
CC_MERGE_DISTANCE_Y = 35       # max vertical px to merge a small CC into a large one
MIN_CHARS_FOR_STRATEGY = 2     # minimum chars for a strategy to be accepted
OVERLAP_THRESHOLD = 0.45       # IoU threshold for deduplication

TESSERACT_LANG = 'heb'         # Use 'heb+eng' if you have eng traineddata for better script separation
ENABLE_THINNING = False        # Set True for heavy/bold fonts — morphological erosion thins strokes
THINNING_KERNEL_SIZE = 2       # Erosion kernel size (larger = more thinning, try 2–3)

# ── Google Cloud Vision API (disabled by default) ──
# Set to True AND provide credentials to use Google Vision for select images.
# Requires: pip install google-cloud-vision
# Auth: set GOOGLE_APPLICATION_CREDENTIALS env var to your service account JSON path
ENABLE_GOOGLE_VISION = False

# Hebrew Unicode detection
HEBREW_RANGE = '\u0590-\u05FF'
HEBREW_REGEX = re.compile(f'[{HEBREW_RANGE}]')

# Optional EasyOCR
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

# Optional Google Cloud Vision
try:
    from google.cloud import vision as google_vision
    GOOGLE_VISION_AVAILABLE = True
except ImportError:
    GOOGLE_VISION_AVAILABLE = False


class HebrewOCREngine:
    """Multi-strategy OCR engine for extracting individual Hebrew glyphs."""

    def __init__(self, use_easyocr=False):
        """Init engine. EasyOCR loaded only if requested and available."""
        self.easyocr_reader = None
        if use_easyocr and EASYOCR_AVAILABLE:
            try:
                self.easyocr_reader = easyocr.Reader(['he'], gpu=False)
                print("[OCR] EasyOCR initialized (Hebrew)")
            except Exception as e:
                print(f"[OCR] EasyOCR init failed, Tesseract only: {e}")
        elif use_easyocr:
            print("[OCR] easyocr not installed — pip install easyocr to enable")

    # ──────────────────────────────────────────────────
    # Image preprocessing
    # ──────────────────────────────────────────────────

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Grayscale + CLAHE contrast enhancement + optional thinning for heavy fonts."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        if ENABLE_THINNING:
            enhanced = self._thin_heavy_strokes(enhanced)
        return enhanced

    def _thin_heavy_strokes(self, gray: np.ndarray) -> np.ndarray:
        """Morphological dilation on grayscale = erodes dark strokes → thins heavy/bold fonts."""
        kernel = np.ones((THINNING_KERNEL_SIZE, THINNING_KERNEL_SIZE), np.uint8)
        return cv2.dilate(gray, kernel, iterations=1)

    def binarize(self, gray: np.ndarray) -> np.ndarray:
        """Otsu binarization → black text on white background."""
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    # ──────────────────────────────────────────────────
    # Strategy 1 — image_to_boxes on full image
    # ──────────────────────────────────────────────────

    def strategy1_full_image_boxes(self, image: np.ndarray) -> List[Dict]:
        """
        Fast path: pytesseract.image_to_boxes returns per-character bboxes
        directly — sometimes works even when image_to_data groups characters.
        """
        pil_image = Image.fromarray(image)
        img_h = image.shape[0]

        try:
            boxes_str = pytesseract.image_to_boxes(
                pil_image,
                lang=TESSERACT_LANG,
                config='--psm 6'
            )
        except Exception as e:
            print(f"[S1] image_to_boxes failed: {e}")
            return []

        symbols = self._parse_boxes_string(boxes_str, img_h, offset=(0, 0))

        # Score each character with real Tesseract confidence via PSM 10
        for sym in symbols:
            sym['confidence'] = self._score_char_confidence(image, sym['bbox'])

        print(f"[S1] Full-image boxes → {len(symbols)} chars")
        return symbols

    def _parse_boxes_string(self, boxes_str: str, img_h: int, offset: Tuple[int, int] = (0, 0)) -> List[Dict]:
        """
        Parse pytesseract.image_to_boxes output string.
        Format per line: 'char left bottom right top page'
        Coords are from bottom-left; we convert to top-left origin.
        """
        ox, oy = offset
        symbols = []

        if not boxes_str or not boxes_str.strip():
            return symbols

        for line in boxes_str.strip().split('\n'):
            parts = line.split()
            if len(parts) < 5:
                continue

            char = parts[0]
            # image_to_boxes coords: left, bottom, right, top (from bottom-left!)
            left = int(parts[1])
            bottom = int(parts[2])
            right = int(parts[3])
            top = int(parts[4])

            # Convert to standard top-left origin
            x0 = ox + left
            y0 = oy + (img_h - top)
            x1 = ox + right
            y1 = oy + (img_h - bottom)

            w, h = x1 - x0, y1 - y0
            if w < MIN_GLYPH_SIZE_PX or h < MIN_GLYPH_SIZE_PX:
                continue

            symbols.append({
                'text': char,
                'confidence': CONFIDENCE_UNSCORED,  # overwritten by _score_char_confidence later
                'bbox': (int(x0), int(y0), int(x1), int(y1)),
                'method': 'boxes_psm6'
            })

        return symbols

    # ──────────────────────────────────────────────────
    # Strategy 2 — Two-pass (word regions → per-region)
    # ──────────────────────────────────────────────────

    def pass1_detect_words(self, image: np.ndarray) -> List[Dict]:
        """
        Pass 1: Tesseract PSM 6 on full image → word bounding boxes + reference text.
        Level 4 = word bboxes (often empty text), Level 5 = grouped text per word.
        """
        pil_image = Image.fromarray(image)
        data = pytesseract.image_to_data(
            pil_image,
            lang=TESSERACT_LANG,
            config='--psm 6',
            output_type=pytesseract.Output.DICT
        )

        n = len(data['text'])

        # Group by word hierarchy key: (block, par, line, word)
        word_bboxes = {}
        word_texts = {}

        for i in range(n):
            level = int(data['level'][i])
            x, y = int(data['left'][i]), int(data['top'][i])
            w, h = int(data['width'][i]), int(data['height'][i])
            text = data['text'][i].strip()
            key = (int(data['block_num'][i]), int(data['par_num'][i]),
                   int(data['line_num'][i]), int(data['word_num'][i]))

            # Level 4 = word bbox (may have empty text, that's fine)
            if level == 4 and w >= MIN_GLYPH_SIZE_PX and h >= MIN_GLYPH_SIZE_PX:
                word_bboxes[key] = (x, y, x + w, y + h)

            # Level 5 = symbol text (grouped into the word hierarchy)
            if level == 5 and text:
                word_texts.setdefault(key, []).append(text)

        # Build word regions pairing bboxes with their reference text
        word_regions = []
        for key, bbox in word_bboxes.items():
            ref_text = ''.join(word_texts.get(key, []))
            word_regions.append({'bbox': bbox, 'reference_text': ref_text})

        print(f"[Pass 1] {len(word_regions)} word regions:")
        for wr in word_regions:
            x0, y0, x1, y1 = wr['bbox']
            print(f"  bbox=({x0},{y0},{x1},{y1}) {x1-x0}x{y1-y0}px text='{wr['reference_text']}'")

        return word_regions

    def pass2a_boxes_per_region(self, image: np.ndarray, word_bbox: Tuple) -> List[Dict]:
        """
        Pass 2a: Run image_to_boxes with PSM 7 (single text line) on a word region.
        Returns individual character symbols with bboxes in absolute coordinates.
        """
        x0, y0, x1, y1 = word_bbox
        h_img, w_img = image.shape[:2]
        pad = WORD_REGION_PADDING
        rx0, ry0 = max(0, x0 - pad), max(0, y0 - pad)
        rx1, ry1 = min(w_img, x1 + pad), min(h_img, y1 + pad)

        region = image[ry0:ry1, rx0:rx1]
        if region.shape[0] < 10 or region.shape[1] < 10:
            return []

        binary = self.binarize(region)
        pil_region = Image.fromarray(binary)
        reg_h = binary.shape[0]

        try:
            boxes_str = pytesseract.image_to_boxes(
                pil_region,
                lang=TESSERACT_LANG,
                config='--psm 7'
            )
        except Exception as e:
            print(f"  [Pass 2a] boxes failed: {e}")
            return []

        symbols = self._parse_boxes_string(boxes_str, reg_h, offset=(rx0, ry0))
        # Tag method + get real confidence via PSM 10 per char
        for s in symbols:
            s['method'] = 'boxes_psm7'
            s['confidence'] = self._score_char_confidence(image, s['bbox'])

        print(f"  [Pass 2a] PSM 7 boxes → {len(symbols)} chars")
        return symbols

    def pass2b_connected_components(self, image: np.ndarray, word_bbox: Tuple,
                                     reference_text: str = "") -> List[Dict]:
        """
        Pass 2b: Segment via connected components, then:
        - If CC count matches reference text length → assign chars 1:1 (RTL order)
        - Otherwise → run PSM 10 on each component for recognition
        """
        x0, y0, x1, y1 = word_bbox
        h_img, w_img = image.shape[:2]
        pad = WORD_REGION_PADDING
        rx0, ry0 = max(0, x0 - pad), max(0, y0 - pad)
        rx1, ry1 = min(w_img, x1 + pad), min(h_img, y1 + pad)

        region = image[ry0:ry1, rx0:rx1]
        binary = self.binarize(region)
        components = self._find_components(binary, offset=(rx0, ry0))

        # Filter reference text to Hebrew-only for matching
        ref_hebrew = ''.join(c for c in reference_text if HEBREW_REGEX.search(c))
        print(f"  [Pass 2b] CC → {len(components)} blobs, ref='{ref_hebrew}' ({len(ref_hebrew)} chars)")

        symbols = []

        # 1:1 matching: CCs sorted right-to-left match text string left-to-right (Hebrew RTL)
        if ref_hebrew and len(components) == len(ref_hebrew):
            print(f"  [Pass 2b] CC count matches ref text → assigning 1:1")
            for idx, comp in enumerate(components):
                # Get real confidence by running PSM 10 on the character crop
                conf = self._score_char_confidence(image, comp['bbox'])
                symbols.append({
                    'text': ref_hebrew[idx],
                    'confidence': conf,
                    'bbox': comp['bbox'],
                    'method': 'cc_matched'
                })
        elif ref_hebrew and len(components) > 0 and abs(len(components) - len(ref_hebrew)) <= 2:
            # Close match: try assigning anyway with EasyOCR/PSM10 fallback for extras
            print(f"  [Pass 2b] CC/ref close ({len(components)} vs {len(ref_hebrew)}), using PSM 10 per blob")
            for comp in components:
                result = self._recognize_char(image, comp['bbox'])
                if result and HEBREW_REGEX.search(result['text']):
                    symbols.append({
                        'text': result['text'],
                        'confidence': result['confidence'],
                        'bbox': comp['bbox'],
                        'method': 'cc_psm10'
                    })
        else:
            # No reference or big mismatch: PSM 10 each component
            for comp in components:
                result = self._recognize_char(image, comp['bbox'])
                if result and HEBREW_REGEX.search(result['text']):
                    symbols.append({
                        'text': result['text'],
                        'confidence': result['confidence'],
                        'bbox': comp['bbox'],
                        'method': 'cc_psm10'
                    })

        return symbols

    # ──────────────────────────────────────────────────
    # Connected-components helpers
    # ──────────────────────────────────────────────────

    def _find_components(self, binary: np.ndarray, offset: Tuple[int, int]) -> List[Dict]:
        """
        Find connected components, merge small ones (dots/niqqud) into nearest large neighbor.
        Returns list sorted right-to-left (Hebrew reading order).
        """
        ox, oy = offset

        # Ink should be foreground (white pixels in CC analysis)
        inv = cv2.bitwise_not(binary) if np.mean(binary) > 127 else binary.copy()
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inv, connectivity=8)

        # Collect all CCs above minimum area
        raw_ccs = []
        for lid in range(1, num_labels):  # skip background label 0
            x = int(stats[lid, cv2.CC_STAT_LEFT])
            y = int(stats[lid, cv2.CC_STAT_TOP])
            w = int(stats[lid, cv2.CC_STAT_WIDTH])
            h = int(stats[lid, cv2.CC_STAT_HEIGHT])
            area = int(stats[lid, cv2.CC_STAT_AREA])
            cx, cy = float(centroids[lid][0]), float(centroids[lid][1])
            if area < MIN_COMPONENT_AREA:
                continue
            raw_ccs.append({'bbox_local': (x, y, x + w, y + h), 'area': area, 'cx': cx, 'cy': cy})

        if not raw_ccs:
            return []

        # Split into large (letter bodies) vs small (dots, dagesh, niqqud)
        areas = sorted([cc['area'] for cc in raw_ccs])
        median_area = areas[len(areas) // 2]
        threshold_area = median_area * CC_SMALL_RATIO

        large = [cc for cc in raw_ccs if cc['area'] >= threshold_area]
        small = [cc for cc in raw_ccs if cc['area'] < threshold_area]

        # Merge each small CC into its nearest large neighbor (dagesh dots → parent letter)
        for s in small:
            best_dist, best_lg = float('inf'), None
            for lg in large:
                dx = abs(s['cx'] - lg['cx'])
                dy = abs(s['cy'] - lg['cy'])
                if dx < CC_MERGE_DISTANCE_X and dy < CC_MERGE_DISTANCE_Y:
                    dist = dx + dy
                    if dist < best_dist:
                        best_dist, best_lg = dist, lg
            if best_lg:
                # Expand the large CC bbox to include the small one
                lx0, ly0, lx1, ly1 = best_lg['bbox_local']
                sx0, sy0, sx1, sy1 = s['bbox_local']
                best_lg['bbox_local'] = (min(lx0, sx0), min(ly0, sy0), max(lx1, sx1), max(ly1, sy1))

        # Convert to absolute coords, sort right-to-left for Hebrew reading order
        components = []
        for cc in large:
            lx0, ly0, lx1, ly1 = cc['bbox_local']
            components.append({
                'bbox': (ox + lx0, oy + ly0, ox + lx1, oy + ly1),
                'area': cc['area'],
                'cx': cc['cx']
            })
        components.sort(key=lambda c: -c['cx'])

        return components

    def _recognize_char(self, image: np.ndarray, bbox: Tuple) -> Optional[Dict]:
        """
        Run Tesseract PSM 10 on a crop to get text + real confidence.
        Tries binarized first, falls back to raw grayscale if that fails.
        """
        x0, y0, x1, y1 = bbox
        h_img, w_img = image.shape[:2]
        pad = 4
        rx0, ry0 = max(0, x0 - pad), max(0, y0 - pad)
        rx1, ry1 = min(w_img, x1 + pad), min(h_img, y1 + pad)

        region = image[ry0:ry1, rx0:rx1]
        if region.shape[0] < 5 or region.shape[1] < 5:
            return None

        # Try both binarized and raw grayscale — pick whichever gives real confidence
        candidates = [self.binarize(region), region]
        for variant in candidates:
            try:
                pil_region = Image.fromarray(variant)
                data = pytesseract.image_to_data(
                    pil_region,
                    lang=TESSERACT_LANG,
                    config='--psm 10',
                    output_type=pytesseract.Output.DICT
                )
                for i in range(len(data['text'])):
                    text = data['text'][i].strip()
                    conf = float(data['conf'][i])
                    if text and len(text) == 1 and conf > 0:
                        return {'text': text, 'confidence': conf}
            except Exception:
                continue
        return None

    def _score_char_confidence(self, image: np.ndarray, bbox: Tuple) -> float:
        """Get Tesseract's confidence for a character at the given bbox (PSM 10)."""
        result = self._recognize_char(image, bbox)
        return result['confidence'] if result else CONFIDENCE_UNSCORED

    # ──────────────────────────────────────────────────
    # Optional EasyOCR cross-reference
    # ──────────────────────────────────────────────────

    def _easyocr_read_word(self, image: np.ndarray, word_bbox: Tuple) -> str:
        """Use EasyOCR to read a word region (returns Hebrew-only text for cross-ref)."""
        if not self.easyocr_reader:
            return ""
        x0, y0, x1, y1 = word_bbox
        h_img, w_img = image.shape[:2]
        pad = WORD_REGION_PADDING
        region = image[max(0, y0 - pad):min(h_img, y1 + pad),
                       max(0, x0 - pad):min(w_img, x1 + pad)]
        try:
            results = self.easyocr_reader.readtext(region, detail=0)
            full = ''.join(results)
            return ''.join(c for c in full if HEBREW_REGEX.search(c))
        except Exception:
            return ""

    # ──────────────────────────────────────────────────
    # Optional Google Cloud Vision API
    # ──────────────────────────────────────────────────

    def process_with_google_vision(self, image_path: str, only_hebrew: bool = True) -> Dict:
        """
        Use Google Cloud Vision API for character-level detection.
        Returns same format as process_image() for seamless substitution.
        """
        if not GOOGLE_VISION_AVAILABLE:
            raise RuntimeError("google-cloud-vision not installed — pip install google-cloud-vision")
        if not ENABLE_GOOGLE_VISION:
            raise RuntimeError("Google Vision is disabled — set ENABLE_GOOGLE_VISION = True in ocr_engine.py")

        client = google_vision.ImageAnnotatorClient()
        with open(image_path, 'rb') as f:
            content = f.read()

        gv_image = google_vision.Image(content=content)
        response = client.text_detection(image=gv_image, image_context={'language_hints': ['he']})

        if response.error.message:
            raise RuntimeError(f"Google Vision API error: {response.error.message}")

        img_bgr = cv2.imread(image_path)
        img_h, img_w = img_bgr.shape[:2]
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr

        all_symbols = []
        # text_annotations[0] is the full text; [1:] are individual symbols
        for annotation in response.text_annotations[1:]:
            text = annotation.description.strip()
            if not text or len(text) != 1:
                continue

            verts = annotation.bounding_poly.vertices
            x0 = min(v.x for v in verts)
            y0 = min(v.y for v in verts)
            x1 = max(v.x for v in verts)
            y1 = max(v.y for v in verts)

            if (x1 - x0) < MIN_GLYPH_SIZE_PX or (y1 - y0) < MIN_GLYPH_SIZE_PX:
                continue

            all_symbols.append({
                'text': text,
                'confidence': annotation.score * 100 if hasattr(annotation, 'score') and annotation.score else CONFIDENCE_UNSCORED,
                'bbox': (int(x0), int(y0), int(x1), int(y1)),
                'method': 'google_vision'
            })

        # Hebrew filter
        if only_hebrew:
            all_symbols = [s for s in all_symbols if HEBREW_REGEX.search(s['text'])]

        all_symbols = self._deduplicate(all_symbols)

        # Refine bounding boxes
        refined = []
        for sym in all_symbols:
            expanded = self.expand_box(sym['bbox'], CROP_BOX_MARGIN, img_w, img_h)
            tight = self.refine_bbox_by_content(img_gray, expanded)
            if tight is None:
                tight = expanded
            final = self.expand_box(tight, POST_REFINE_PADDING, img_w, img_h)
            refined.append({
                'text': sym['text'],
                'confidence': float(sym['confidence']),
                'bbox': tuple(int(v) for v in final),
                'bbox_original': tuple(int(v) for v in sym['bbox']),
                'method': 'google_vision'
            })

        # Per-character stats
        char_stats = {}
        for sym in refined:
            ch = sym['text']
            s = char_stats.setdefault(ch, {'count': 0, 'sum_conf': 0, 'min_conf': 100, 'max_conf': 0})
            c = sym['confidence']
            s['count'] += 1
            s['sum_conf'] += c
            s['min_conf'] = min(s['min_conf'], c)
            s['max_conf'] = max(s['max_conf'], c)

        print(f"[Google Vision] {len(refined)} Hebrew glyphs extracted")

        return {
            'source_image': image_path,
            'width': int(img_w),
            'height': int(img_h),
            'count': len(refined),
            'strategy': 'google_vision',
            'words': [],
            'symbols': refined,
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

    # ──────────────────────────────────────────────────
    # Bbox refinement utilities
    # ──────────────────────────────────────────────────

    def refine_bbox_by_content(self, image: np.ndarray, bbox: Tuple) -> Optional[Tuple]:
        """Tighten bounding box to actual ink pixels (below luminance threshold)."""
        x0, y0, x1, y1 = bbox
        region = image[y0:y1, x0:x1]
        if region.size == 0:
            return None
        ink_mask = region < CROP_LUM_THR
        if not np.any(ink_mask):
            return None
        rows = np.any(ink_mask, axis=1)
        cols = np.any(ink_mask, axis=0)
        if not np.any(rows) or not np.any(cols):
            return None
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        return (int(x0 + x_min), int(y0 + y_min), int(x0 + x_max + 1), int(y0 + y_max + 1))

    def isolate_main_component(self, image: np.ndarray, padded_bbox: Tuple,
                                original_center: Tuple[float, float]) -> Optional[Tuple]:
        """
        Within a padded region, find the connected component whose centroid is
        closest to original_center and return its tight bounding box.
        Prevents neighbor-letter ink from contaminating the crop.
        """
        x0, y0, x1, y1 = padded_bbox
        region = image[y0:y1, x0:x1]
        if region.size == 0:
            return None

        ink_mask = (region < CROP_LUM_THR).astype(np.uint8)
        if not np.any(ink_mask):
            return None

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ink_mask, connectivity=8)
        if num_labels <= 1:
            return None

        # Target center relative to the padded region
        tcx = original_center[0] - x0
        tcy = original_center[1] - y0

        # Find the CC closest to the original detection center (skip label 0 = background)
        best_label, best_dist = -1, float('inf')
        for lid in range(1, num_labels):
            area = int(stats[lid, cv2.CC_STAT_AREA])
            if area < MIN_COMPONENT_AREA:
                continue
            cx, cy = float(centroids[lid][0]), float(centroids[lid][1])
            dist = abs(cx - tcx) + abs(cy - tcy)
            if dist < best_dist:
                best_dist, best_label = dist, lid

        if best_label < 0:
            return None

        # Merge nearby small CCs that likely belong to the same letter (dots, dagesh)
        main_cx = float(centroids[best_label][0])
        main_cy = float(centroids[best_label][1])
        merged_mask = (labels == best_label)
        for lid in range(1, num_labels):
            if lid == best_label:
                continue
            area = int(stats[lid, cv2.CC_STAT_AREA])
            cx, cy = float(centroids[lid][0]), float(centroids[lid][1])
            dx, dy = abs(cx - main_cx), abs(cy - main_cy)
            if dx < CC_MERGE_DISTANCE_X and dy < CC_MERGE_DISTANCE_Y:
                merged_mask |= (labels == lid)

        rows = np.any(merged_mask, axis=1)
        cols = np.any(merged_mask, axis=0)
        if not np.any(rows) or not np.any(cols):
            return None
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        return (int(x0 + x_min), int(y0 + y_min), int(x0 + x_max + 1), int(y0 + y_max + 1))

    def expand_box(self, bbox: Tuple, margin: int, max_w: int, max_h: int) -> Tuple:
        """Expand bounding box by margin, clamped to image bounds."""
        x0, y0, x1, y1 = bbox
        return (max(0, x0 - margin), max(0, y0 - margin),
                min(max_w, x1 + margin), min(max_h, y1 + margin))

    def _deduplicate(self, symbols: List[Dict]) -> List[Dict]:
        """Remove overlapping detections, keep highest confidence."""
        if len(symbols) <= 1:
            return symbols
        sorted_syms = sorted(symbols, key=lambda s: -s['confidence'])
        kept = []
        for sym in sorted_syms:
            sx0, sy0, sx1, sy1 = sym['bbox']
            is_dup = False
            for k in kept:
                kx0, ky0, kx1, ky1 = k['bbox']
                ix0, iy0 = max(sx0, kx0), max(sy0, ky0)
                ix1, iy1 = min(sx1, kx1), min(sy1, ky1)
                if ix0 < ix1 and iy0 < iy1:
                    inter = (ix1 - ix0) * (iy1 - iy0)
                    min_area = min((sx1 - sx0) * (sy1 - sy0), (kx1 - kx0) * (ky1 - ky0))
                    if min_area > 0 and inter / min_area > OVERLAP_THRESHOLD:
                        is_dup = True
                        break
            if not is_dup:
                kept.append(sym)
        return kept

    # ──────────────────────────────────────────────────
    # Main pipeline
    # ──────────────────────────────────────────────────

    def process_image(self, image_path: str, only_hebrew: bool = True) -> Dict:
        """
        Full multi-strategy pipeline for Hebrew glyph extraction.
        Tries strategies in order until one produces enough characters.
        """
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Could not load image: {image_path}")

        img_pre = self.preprocess(img_bgr)
        img_h, img_w = img_pre.shape[:2]
        print(f"\n{'='*60}")
        print(f"[OCR] Processing {image_path} ({img_w}x{img_h})")
        print(f"{'='*60}")

        all_symbols = []
        word_regions = []
        strategy_used = "none"

        # ── Strategy 1: image_to_boxes on full image (fast path) ──
        s1_symbols = self.strategy1_full_image_boxes(img_pre)
        # Keep only Hebrew for counting
        s1_hebrew = [s for s in s1_symbols if HEBREW_REGEX.search(s['text'])]

        if len(s1_hebrew) >= MIN_CHARS_FOR_STRATEGY:
            all_symbols = s1_hebrew
            strategy_used = "S1 (full-image boxes)"
            print(f"[OCR] ✓ Strategy 1 succeeded: {len(all_symbols)} Hebrew chars")
        else:
            print(f"[OCR] Strategy 1 found only {len(s1_hebrew)} Hebrew chars → trying two-pass")

            # ── Strategy 2: Two-pass (word regions → per-region extraction) ──
            word_regions = self.pass1_detect_words(img_pre)

            for wr in word_regions:
                bbox = wr['bbox']
                ref_text = wr['reference_text']

                # Optional EasyOCR cross-reference for better reference text
                if self.easyocr_reader:
                    easy_text = self._easyocr_read_word(img_pre, bbox)
                    if easy_text:
                        print(f"  [EasyOCR] '{easy_text}'")
                        if not ref_text:
                            ref_text = easy_text

                # Pass 2a: image_to_boxes PSM 7 on word region
                p2a_syms = self.pass2a_boxes_per_region(img_pre, bbox)
                p2a_hebrew = [s for s in p2a_syms if HEBREW_REGEX.search(s['text'])]

                if len(p2a_hebrew) >= MIN_CHARS_FOR_STRATEGY:
                    all_symbols.extend(p2a_hebrew)
                    continue

                # Pass 2b: connected components + matching / PSM 10
                p2b_syms = self.pass2b_connected_components(img_pre, bbox, ref_text)
                all_symbols.extend(p2b_syms)

            strategy_used = "S2 (two-pass)"
            print(f"[OCR] Strategy 2 produced {len(all_symbols)} symbols")

            # ── Strategy 3: CC on full image if nothing found ──
            if not all_symbols:
                print(f"[OCR] No symbols from S2 → trying full-image CC")
                full_cc = self.pass2b_connected_components(img_pre, (0, 0, img_w, img_h), "")
                all_symbols.extend(full_cc)
                strategy_used = "S3 (full-image CC)"

        # ── Hebrew filter — drop non-Hebrew chars (Latin, punctuation, etc.) ──
        if only_hebrew:
            before = len(all_symbols)
            all_symbols = [s for s in all_symbols if HEBREW_REGEX.search(s['text'])]
            print(f"[OCR] Hebrew filter: kept {len(all_symbols)}/{before} (dropped {before - len(all_symbols)} non-Hebrew)")

        # ── Confidence filter — drop low-confidence noise ──
        before = len(all_symbols)
        all_symbols = [s for s in all_symbols
                       if s['confidence'] == CONFIDENCE_UNSCORED  # keep unscored (N/A)
                       or s['confidence'] >= MIN_CHAR_CONFIDENCE]
        if before != len(all_symbols):
            print(f"[OCR] Confidence filter (>={MIN_CHAR_CONFIDENCE}): kept {len(all_symbols)}/{before}")

        # ── Deduplicate overlapping detections ──
        all_symbols = self._deduplicate(all_symbols)
        print(f"[OCR] After dedup: {len(all_symbols)} symbols")

        # ── Refine bounding boxes: isolate single letter, then tighten ──
        refined = []
        for sym in all_symbols:
            orig = sym['bbox']
            orig_cx = (orig[0] + orig[2]) / 2.0
            orig_cy = (orig[1] + orig[3]) / 2.0
            expanded = self.expand_box(orig, CROP_BOX_MARGIN, img_w, img_h)

            # Isolate the main connected component closest to the detection center
            isolated = self.isolate_main_component(img_pre, expanded, (orig_cx, orig_cy))
            if isolated is None:
                isolated = self.refine_bbox_by_content(img_pre, expanded)
            if isolated is None:
                isolated = expanded

            final = self.expand_box(isolated, POST_REFINE_PADDING, img_w, img_h)
            refined.append({
                'text': sym['text'],
                'confidence': float(sym['confidence']),
                'bbox': tuple(int(v) for v in final),
                'bbox_original': tuple(int(v) for v in sym['bbox']),
                'method': sym.get('method', 'unknown')
            })

        # ── Per-character stats ──
        char_stats = {}
        for sym in refined:
            ch = sym['text']
            s = char_stats.setdefault(ch, {'count': 0, 'sum_conf': 0, 'min_conf': 100, 'max_conf': 0})
            c = sym['confidence']
            s['count'] += 1
            s['sum_conf'] += c
            s['min_conf'] = min(s['min_conf'], c)
            s['max_conf'] = max(s['max_conf'], c)

        # ── Summary ──
        print(f"\n[OCR] ── RESULT ({strategy_used}) ──")
        print(f"[OCR] {len(refined)} glyphs extracted:")
        method_counts = {}
        for sym in refined:
            m = sym['method']
            method_counts[m] = method_counts.get(m, 0) + 1
            print(f"  '{sym['text']}' conf={sym['confidence']:.0f} bbox={sym['bbox']} via {m}")
        print(f"[OCR] Methods used: {method_counts}")

        return {
            'source_image': image_path,
            'width': int(img_w),
            'height': int(img_h),
            'count': len(refined),
            'strategy': strategy_used,
            'words': [{'bbox': tuple(int(v) for v in wr['bbox']), 'text': wr['reference_text']}
                      for wr in word_regions],
            'symbols': refined,
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
