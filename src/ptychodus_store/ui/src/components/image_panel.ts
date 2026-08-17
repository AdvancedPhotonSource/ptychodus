import type { RenderedImage } from '../api.js';

export interface ImagePanel {
  el: HTMLElement;
  setEmpty: (message?: string) => void;
  setLoading: (label: string) => void;
  setError: (err: Error) => void;
  setImage: (img: RenderedImage, title: string) => void;
}

export function createImagePanel(): ImagePanel {
  const el = document.createElement('div');
  el.className = 'image-panel';

  const wrap = document.createElement('div');
  wrap.className = 'image-wrap';
  const empty = document.createElement('div');
  empty.className = 'empty';
  empty.textContent = 'Select an item to preview.';
  wrap.appendChild(empty);
  const caption = document.createElement('div');
  caption.className = 'caption';
  el.append(wrap, caption);

  return {
    el,
    setEmpty(msg = 'Select an item to preview.') {
      wrap.replaceChildren(makeMessage('empty', msg));
      caption.textContent = '';
    },
    setLoading(label) {
      wrap.replaceChildren(makeMessage('status', `Loading ${label}…`));
      caption.textContent = '';
    },
    setError(err) {
      wrap.replaceChildren(makeMessage('error', err.message));
      caption.textContent = '';
    },
    setImage(img, title) {
      const el = document.createElement('img');
      el.src = `data:${img.mime_type};base64,${img.png_base64}`;
      el.alt = title;
      wrap.replaceChildren(el);
      const pw_um = img.pixel_width_m * 1e6;
      const ph_um = img.pixel_height_m * 1e6;
      caption.innerHTML = `
        <div><strong>${escapeHtml(title)}</strong></div>
        <div>${escapeHtml(img.value_label)} — range [${fmt(img.color_value_min)}, ${fmt(img.color_value_max)}]</div>
        <div>${img.shape_w_px} × ${img.shape_h_px} px — pixel ${pw_um.toFixed(3)} × ${ph_um.toFixed(3)} µm</div>
      `;
    },
  };
}

function makeMessage(cls: string, text: string): HTMLElement {
  const div = document.createElement('div');
  div.className = cls;
  div.textContent = text;
  return div;
}

function fmt(x: number): string {
  if (!Number.isFinite(x)) return String(x);
  const abs = Math.abs(x);
  if (abs !== 0 && (abs >= 1e4 || abs < 1e-2)) return x.toExponential(3);
  return x.toPrecision(4);
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
