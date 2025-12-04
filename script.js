/* global Tesseract, JSZip, saveAs */
/* script.js — Tesseract.js v5 (browser)
- origin chatgpt, edits cursor
   - Create worker with langs in arg #1 (v5 API)
   - NO load()/loadLanguage()/initialize() calls (deprecated)
   - Pass File/Blob directly to recognize() to avoid cloning/format issues
   - Use canvas only for cropping previews
*/

  // ---- Cropping params (tunable here in code) ----
const CROP_BOX_MARGIN      = 4;   // px padding around Tesseract bbox (pre-refine)
const POST_REFINE_PADDING  = 2;   // px padding after ink-tightening
const CROP_LUM_THR         = 220; // 0..255; lower = tighter (more aggressive)
const MIN_GLYPH_SIZE_PX    = 8;   // minimum width/height of a glyph bbox to keep


(() => {
  const fileInput    = document.getElementById('fileInput');
  const runBtn       = document.getElementById('runBtn');
  const dlZipBtn     = document.getElementById('dlZipBtn');
  const dlCharZipBtn = document.getElementById('dlCharZipBtn');
  // const levelSel    = document.getElementById('level');
  const onlyHebrewC = document.getElementById('onlyHebrew');
  // const langPathInp = document.getElementById('langPath');
  const logEl       = document.getElementById('log');
  const gallery     = document.getElementById('gallery');
  const workCanvas  = document.getElementById('workCanvas');



  const hebrewRegex = /[\u0590-\u05FF]/;
  let lastZipBlob = null;
  let lastCharZipBlob = null;

  function log(msg) {
    logEl.textContent += msg + '\n';
    logEl.scrollTop = logEl.scrollHeight;
  }
  function clearUI() {
    logEl.textContent = '';
    gallery.innerHTML = '';
    lastZipBlob = null;
    lastCharZipBlob = null;
    dlZipBtn.disabled = true;
    dlCharZipBtn.disabled = true;
  }
  function clamp(v, min, max) { return Math.max(min, Math.min(max, v)); }
  function fmtConf(c) { return (typeof c === 'number' && isFinite(c)) ? c.toFixed(1) : String(c ?? ''); }
  function safeTextForFilename(t) {
    const txt = (t || '').trim();
    const rep = txt.length ? txt.replace(/\s/g, '␠') : 'space';
    return rep.replace(/[\\\/:*?"<>|]/g, '_');
  }
  function dataURLToUint8(dataURL) {
    const b64 = dataURL.split(',')[1];
    const bin = atob(b64);
    const u8 = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    return u8;
  }
  function toTSV(items, sourceName, level) {
    const header = 'idx\tlevel\ttext\tconf\tleft\ttop\twidth\theight\tsource';
    const rows = items.map((it, i) => {
      const { text, confidence, bbox } = it;
      const w = bbox.x1 - bbox.x0, h = bbox.y1 - bbox.y0;
      return [
        i, level, (text ?? '').replace(/\s/g, '␠'),
        fmtConf(confidence), bbox.x0, bbox.y0, w, h, sourceName
      ].join('\t');
    });
    return [header, ...rows].join('\n');
  }

  function expandBox({ x0, y0, x1, y1 }, margin, maxW, maxH) {
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  return {
    x0: clamp(x0 - margin, 0, maxW),
    y0: clamp(y0 - margin, 0, maxH),
    x1: clamp(x1 + margin, 0, maxW),
    y1: clamp(y1 + margin, 0, maxH),
  };
}

function refineBoxByContent(ctx, region, thr /* 0..255 */) {
  const { x0, y0, x1, y1 } = region;
  const w = Math.max(1, (x1 - x0) | 0), h = Math.max(1, (y1 - y0) | 0);
  const img = ctx.getImageData(x0, y0, w, h);
  const data = img.data;
  let minX = w, minY = h, maxX = -1, maxY = -1;

  for (let yy = 0; yy < h; yy++) {
    for (let xx = 0; xx < w; xx++) {
      const i = (yy * w + xx) * 4;
      const lum = 0.2126*data[i] + 0.7152*data[i+1] + 0.0722*data[i+2];
      if (lum < thr) { // treat as “ink”
        if (xx < minX) minX = xx; if (yy < minY) minY = yy;
        if (xx > maxX) maxX = xx; if (yy > maxY) maxY = yy;
      }
    }
  }
  if (maxX < minX || maxY < minY) return null; // no ink
  return { x0: x0 + minX, y0: y0 + minY, x1: x0 + maxX + 1, y1: y0 + maxY + 1 };
}


  function loadHTMLImageFromFile(file) {
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => { URL.revokeObjectURL(url); resolve(img); };
      img.onerror = reject;
      img.src = url;
    });
  }

  async function runOCR() {
    clearUI();

    const files = Array.from(fileInput.files || []);
    if (!files.length) { log('No files selected.'); return; }

    // const level = levelSel.value; // 'symbols' | 'words'
    // see index.html: I'm removing level since I only want symbols (glyphs), but keeping it here for future option
    const level = 'symbols';
    const onlyHebrew = !!onlyHebrewC.checked;

    // Create the worker — v5 API: languages as first arg; options as 3rd
    // const lp = (langPathInp.value || '').trim();
    const lp = "https://tessdata.projectnaptha.com/4.0.0"; // hardcode for now
    const worker = await Tesseract.createWorker('heb', 1, lp ? { langPath: lp } : undefined);

    const zipByImage = new JSZip();
    const zipByChar = new JSZip();
    const charBuckets = new Map(); // key: char label -> { folder, safeLabel, entries: [] }
      function getCharBucket(labelRaw) {
        const key = (labelRaw && labelRaw.length) ? labelRaw : 'blank';
        if (!charBuckets.has(key)) {
          const safeLabel = safeTextForFilename(key);
          const folderName = `char_${safeLabel}`;
          charBuckets.set(key, {
            label: key,
            safeLabel,
            folderName,
            folder: zipByChar.folder(folderName),
            entries: []
          });
        }
        return charBuckets.get(key);
      }

    try {
      // v5 workers are already loaded & initialized — do NOT call load/loadLanguage/initialize

      // Optional params (strings only)
      // Limit recognition to Hebrew characters to reduce cross-script confusions.
      await worker.setParameters({
        preserve_interword_spaces: '1',
        tessedit_char_whitelist: 'אבגדהוזחטיכלמנסעפצקרשתךםןףץ'
      });

      for (const file of files) {
        log(`\n=== ${file.name} ===`);

        // 1) OCR: pass the File/Blob directly (clone-safe; correct format)
        const res = await worker.recognize(file);
        const items = (level === 'words') ? res.data.words : res.data.symbols;

        // 2) Draw once to canvas for cropping
        const img = await loadHTMLImageFromFile(file);
        const imgW = img.naturalWidth || img.width;
        const imgH = img.naturalHeight || img.height;
        workCanvas.width = imgW;
        workCanvas.height = imgH;
        const ctx = workCanvas.getContext('2d', { willReadFrequently: true });
        ctx.clearRect(0, 0, imgW, imgH);
        ctx.drawImage(img, 0, 0);

        // 3) Filter items
        const filtered = items
          .filter(it => it && it.bbox && (it.text ?? '').trim().length > 0)
          // size sanity check to drop tiny / between-letters fragments
          .filter(it => {
            const { x0, y0, x1, y1 } = it.bbox;
            const w = x1 - x0;
            const h = y1 - y0;
            return w >= MIN_GLYPH_SIZE_PX && h >= MIN_GLYPH_SIZE_PX;
          })
          .filter(it => !onlyHebrew || hebrewRegex.test(it.text));

        // 4) TSV + metadata
        const tsv = toTSV(filtered, file.name, level);

        // Per-glyph stats to help inspect confusion patterns.
        const charStats = {};
        for (const it of filtered) {
          const txt = (it.text || '').trim();
          if (!txt) continue;
          const ch = txt; // glyph-level: expect a single character
          if (!charStats[ch]) {
            charStats[ch] = { count: 0, sumConf: 0, minConf: 100, maxConf: 0 };
          }
          const s = charStats[ch];
          const c = typeof it.confidence === 'number' && isFinite(it.confidence) ? it.confidence : 0;
          s.count += 1;
          s.sumConf += c;
          if (c < s.minConf) s.minConf = c;
          if (c > s.maxConf) s.maxConf = c;
        }

        const metadata = {
          source_image: file.name,
          level,
          onlyHebrew,
          width: imgW,
          height: imgH,
          count: filtered.length,
          stats_by_char: Object.entries(charStats).map(([ch, s]) => ({
            char: ch,
            count: s.count,
            avg_conf: s.count ? s.sumConf / s.count : 0,
            min_conf: s.minConf,
            max_conf: s.maxConf
          })),
          items: filtered.map((it, i) => ({
            index: i,
            text: it.text,
            confidence: it.confidence,
            bbox: {
              left: it.bbox.x0,
              top: it.bbox.y0,
              width: it.bbox.x1 - it.bbox.x0,
              height: it.bbox.y1 - it.bbox.y0
            }
          }))
        };
        zipByImage.file(`${file.name}.tsv.txt`, tsv);
        zipByImage.file(`${file.name}.metadata.${level}.json`, JSON.stringify(metadata, null, 2));

        // 5) Crops → ZIP + gallery
        const baseName = file.name.replace(/\s/g, '_');
        const folder = zipByImage.folder(`${file.name}_crops_${level}`);
        for (let i = 0; i < filtered.length; i++) {
          const it = filtered[i];
          // const { x0, y0, x1, y1 } = it.bbox;
          // const left = clamp(x0, 0, imgW);
          // const top  = clamp(y0, 0, imgH);
          // const w    = clamp(x1 - x0, 1, imgW - left);
          // const h    = clamp(y1 - y0, 1, imgH - top);

          // 1) start from Tesseract bbox
let { x0, y0, x1, y1 } = it.bbox;

// 2) pad using global margin
let padded = expandBox({ x0, y0, x1, y1 }, CROP_BOX_MARGIN, imgW, imgH);

// 3) tighten to ink (fallback to padded if none found)
let refined = refineBoxByContent(ctx, padded, CROP_LUM_THR) || padded;

// 4) add post-refine padding to keep a ring around the ink
let finalBox = expandBox(refined, POST_REFINE_PADDING, imgW, imgH);

// 5) final integers + clamping
const left = Math.max(0, finalBox.x0 | 0);
const top  = Math.max(0, finalBox.y0 | 0);
const w    = Math.max(1, (finalBox.x1 - finalBox.x0) | 0);
const h    = Math.max(1, (finalBox.y1 - finalBox.y0) | 0);

// (keep your existing drawImage to crop)

          const crop = document.createElement('canvas');
          crop.width = w; crop.height = h;
          const cctx = crop.getContext('2d');
          cctx.drawImage(workCanvas, left, top, w, h, 0, 0, w, h);

          const dataURL = crop.toDataURL('image/png');

          // gallery
          const card = document.createElement('div');
          card.className = 'thumb';
          const preview = new Image();
          preview.src = dataURL;
          const metaDiv = document.createElement('div');
          metaDiv.className = 'meta';
          metaDiv.textContent = `#${String(i).padStart(4,'0')} | ${level} | ${(it.text || '').replace(/\s/g,'␠')} | conf=${fmtConf(it.confidence)}`;
          card.appendChild(preview);
          card.appendChild(metaDiv);
          gallery.appendChild(card);

          // zip
          const u8 = dataURLToUint8(dataURL);
          // File naming: search "CROP_FILENAME" in code if you need to change format.
const fname = `${baseName}_crops_${String(i).padStart(5,'0')}_${safeTextForFilename(it.text)}.png`;
          folder.file(fname, u8, { binary: true });
          const charLabel = (it.text || '').trim();
          const bucket = getCharBucket(charLabel);
          bucket.folder.file(fname, u8, { binary: true });
          bucket.entries.push({
            source_image: file.name,
            crop_filename: fname,
            text: it.text,
            confidence: it.confidence,
            bbox: {
              left: it.bbox.x0,
              top: it.bbox.y0,
              width: it.bbox.x1 - it.bbox.x0,
              height: it.bbox.y1 - it.bbox.y0
            }
          });
        }

        log(`Crops: ${filtered.length}`);
      }
    } catch (err) {
      console.error(err);
      log('ERROR: ' + (err && err.message ? err.message : String(err)));
    } finally {
      try { await worker.terminate(); } catch {}
      log('Worker terminated.');
    }

    // Emit per-char TSV/JSON files inside each char folder
    for (const bucket of charBuckets.values()) {
      const header = 'idx\tsource_image\tcrop_filename\ttext\tconf\tleft\ttop\twidth\theight';
      const rows = bucket.entries.map((entry, idx) => {
        const { source_image, crop_filename, text, confidence, bbox } = entry;
        return [
          idx,
          source_image,
          crop_filename,
          (text ?? '').replace(/\s/g, '␠'),
          fmtConf(confidence),
          bbox.left,
          bbox.top,
          bbox.width,
          bbox.height
        ].join('\t');
      });
      const tsvContent = [header, ...rows].join('\n');
      bucket.folder.file(`${bucket.folderName}.tsv.txt`, tsvContent);

      const metadata = {
        char: bucket.label,
        count: bucket.entries.length,
        items: bucket.entries
      };
      bucket.folder.file(`${bucket.folderName}.metadata.json`, JSON.stringify(metadata, null, 2));
    }

    log('Building ZIPs…');
    lastZipBlob = await zipByImage.generateAsync({ type: 'blob', compression: 'DEFLATE' });
    lastCharZipBlob = await zipByChar.generateAsync({ type: 'blob', compression: 'DEFLATE' });
    dlZipBtn.disabled = false;
    dlCharZipBtn.disabled = false;
    log('Ready. Use the download buttons.');
  }

  runBtn.addEventListener('click', runOCR);
  dlZipBtn.addEventListener('click', () => {
    if (!lastZipBlob) return;
    saveAs(lastZipBlob, 'ocr_crops_by_image.zip');
  });
  dlCharZipBtn.addEventListener('click', () => {
    if (!lastCharZipBlob) return;
    saveAs(lastCharZipBlob, 'ocr_crops_by_char.zip');
  });

  window.addEventListener('unhandledrejection', (e) => {
    e.preventDefault();
    console.error('Unhandled rejection:', e.reason);
    log('Unhandled Promise Rejection: ' + (e.reason?.message || String(e.reason)));
  });
})();
