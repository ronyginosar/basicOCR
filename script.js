/* global Tesseract, JSZip, saveAs */
/* script.js — Tesseract.js v5 (browser)
   - Create worker with langs in arg #1 (v5 API)
   - NO load()/loadLanguage()/initialize() calls (deprecated)
   - Pass File/Blob directly to recognize() to avoid cloning/format issues
   - Use canvas only for cropping previews
*/

  // ---- Cropping params ----
const CROP_BOX_MARGIN = 2;     // px padding around Tesseract bbox
const CROP_LUM_THR    = 220;   // 0..255; lower = tighter (more aggressive)


(() => {
  const fileInput   = document.getElementById('fileInput');
  const runBtn      = document.getElementById('runBtn');
  const dlZipBtn    = document.getElementById('dlZipBtn');
  // const levelSel    = document.getElementById('level');
  const onlyHebrewC = document.getElementById('onlyHebrew');
  // const langPathInp = document.getElementById('langPath');
  const logEl       = document.getElementById('log');
  const gallery     = document.getElementById('gallery');
  const workCanvas  = document.getElementById('workCanvas');



  const hebrewRegex = /[\u0590-\u05FF]/;
  let lastZipBlob = null;

  function log(msg) {
    logEl.textContent += msg + '\n';
    logEl.scrollTop = logEl.scrollHeight;
  }
  function clearUI() {
    logEl.textContent = '';
    gallery.innerHTML = '';
    lastZipBlob = null;
    dlZipBtn.disabled = true;
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

    const zip = new JSZip();

    try {
      // v5 workers are already loaded & initialized — do NOT call load/loadLanguage/initialize

      // Optional params (strings only)
      await worker.setParameters({
        preserve_interword_spaces: '1'
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
          .filter(it => !onlyHebrew || hebrewRegex.test(it.text));

        // 4) TSV + metadata
        const tsv = toTSV(filtered, file.name, level);
        const metadata = {
          source_image: file.name,
          level,
          onlyHebrew,
          width: imgW,
          height: imgH,
          count: filtered.length,
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
        zip.file(`${file.name}.tsv.txt`, tsv);
        zip.file(`${file.name}.metadata.${level}.json`, JSON.stringify(metadata, null, 2));

        // 5) Crops → ZIP + gallery
        const folder = zip.folder(`${file.name}_crops_${level}`);
        for (let i = 0; i < filtered.length; i++) {
          const it = filtered[i];
          // const { x0, y0, x1, y1 } = it.bbox;
          // const left = clamp(x0, 0, imgW);
          // const top  = clamp(y0, 0, imgH);
          // const w    = clamp(x1 - x0, 1, imgW - left);
          // const h    = clamp(y1 - y0, 1, imgH - top);

          // 1) start from Tesseract bbox
let { x0, y0, x1, y1 } = it.bbox;

// 2) pad
let padded = expandBox({ x0, y0, x1, y1 }, CROP_BOX_MARGIN, imgW, imgH);

// 3) tighten to ink (fallback to padded if none found)
let refined = refineBoxByContent(ctx, padded, CROP_LUM_THR) || padded;

// 4) final integers + clamping
const left = Math.max(0, refined.x0 | 0);
const top  = Math.max(0, refined.y0 | 0);
const w    = Math.max(1, (refined.x1 - refined.x0) | 0);
const h    = Math.max(1, (refined.y1 - refined.y0) | 0);

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
          const fname = `${String(i).padStart(5,'0')}_${safeTextForFilename(it.text)}.png`;
          folder.file(fname, u8, { binary: true });
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

    log('Building ZIP…');
    lastZipBlob = await zip.generateAsync({ type: 'blob', compression: 'DEFLATE' });
    dlZipBtn.disabled = false;
    log('Ready. Click "Download ZIP".');
  }

  runBtn.addEventListener('click', runOCR);
  dlZipBtn.addEventListener('click', () => {
    if (!lastZipBlob) return;
    saveAs(lastZipBlob, 'ocr_crops_bundle.zip');
  });

  window.addEventListener('unhandledrejection', (e) => {
    e.preventDefault();
    console.error('Unhandled rejection:', e.reason);
    log('Unhandled Promise Rejection: ' + (e.reason?.message || String(e.reason)));
  });
})();
