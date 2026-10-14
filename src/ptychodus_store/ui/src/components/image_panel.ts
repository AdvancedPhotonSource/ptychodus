import { api, type PlotImage, type RenderParams, type RenderedImage } from '../api.js';

export interface ImagePanel {
  el: HTMLElement;
  /**
   * Show the Colorize and Data Range controls, re-fetching through `refresh`
   * whenever one changes. Omitted on a page whose output is a plot: a plot carries
   * its own axes and takes no render parameters, exactly as the desktop panel has
   * no ribbon there.
   */
  enableControls: (refresh: (params: Partial<RenderParams>) => void) => void;
  setEmpty: (message?: string) => void;
  setLoading: (label: string) => void;
  setError: (err: Error) => void;
  setImage: (img: RenderedImage, title: string) => void;
  // A plot carries its own axes, so it has no colorbar or pixel size to caption.
  setPlot: (img: PlotImage, title: string) => void;
}

export function createImagePanel(): ImagePanel {
  const el = document.createElement('div');
  el.className = 'image-panel';

  const ribbon = document.createElement('div');
  ribbon.className = 'image-ribbon';
  ribbon.hidden = true;

  const wrap = document.createElement('div');
  wrap.className = 'image-wrap';
  const empty = document.createElement('div');
  empty.className = 'empty';
  empty.textContent = 'Select an item to preview.';
  wrap.appendChild(empty);
  const caption = document.createElement('div');
  caption.className = 'caption';
  el.append(ribbon, wrap, caption);

  // Mirrors the desktop ImageRibbon, minus the image tools: the pan, ruler and
  // line-cut buttons act on a canvas this panel does not have.
  function enableControls(refresh: (params: Partial<RenderParams>) => void): void {
    const params: Partial<RenderParams> = {};
    const apply = (): void => refresh({ ...params });

    const select = (label: string, key: keyof RenderParams, values: string[], blank = false) => {
      const wrapper = document.createElement('label');
      wrapper.textContent = label;
      const el = document.createElement('select');
      if (blank) el.appendChild(new Option('—', ''));
      for (const v of values) el.appendChild(new Option(v, v));
      el.addEventListener('change', () => {
        (params as Record<string, unknown>)[key] = el.value === '' ? null : el.value;
        apply();
      });
      wrapper.appendChild(el);
      return wrapper;
    };

    const number = (label: string, key: 'value_min' | 'value_max') => {
      const wrapper = document.createElement('label');
      wrapper.textContent = label;
      const el = document.createElement('input');
      el.type = 'number';
      el.step = 'any';
      el.addEventListener('change', () => {
        params[key] = el.value === '' ? null : Number(el.value);
        apply();
      });
      wrapper.appendChild(el);
      return wrapper;
    };

    api
      .visualizationOptions()
      .then((opts) => {
        const colorize = group('Colorize');
        colorize.append(
          select('colormap', 'colormap', [...opts.colormaps_linear, ...opts.colormaps_cyclic]),
          select('transform', 'transform', opts.transforms),
          select('component', 'component', opts.components, true),
          select('color model', 'color_model', opts.color_models, true)
        );

        const range = group('Data Range');
        const auto = document.createElement('button');
        auto.type = 'button';
        auto.textContent = 'Auto';
        auto.addEventListener('click', () => {
          params.value_min = null;
          params.value_max = null;
          for (const input of range.querySelectorAll('input[type=number]')) {
            (input as HTMLInputElement).value = '';
          }
          apply();
        });

        const clip = document.createElement('label');
        clip.textContent = 'clip';
        const clipBox = document.createElement('input');
        clipBox.type = 'checkbox';
        clipBox.addEventListener('change', () => {
          params.clip = clipBox.checked;
          apply();
        });
        clip.appendChild(clipBox);

        range.append(number('min', 'value_min'), number('max', 'value_max'), auto, clip);
        ribbon.replaceChildren(colorize, range);
        ribbon.hidden = false;
      })
      .catch(() => {
        // The ribbon is an enhancement; a failed options fetch leaves the default
        // render in place rather than blanking the panel.
        ribbon.hidden = true;
      });
  }

  return {
    el,
    enableControls,
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
    setPlot(img, title) {
      const el = document.createElement('img');
      el.src = `data:${img.mime_type};base64,${img.png_base64}`;
      el.alt = title;
      wrap.replaceChildren(el);
      caption.innerHTML = `<div><strong>${escapeHtml(title)}</strong></div>`;
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

function group(title: string): HTMLElement {
  const el = document.createElement('fieldset');
  el.className = 'ribbon-group';
  const legend = document.createElement('legend');
  legend.textContent = title;
  el.appendChild(legend);
  return el;
}
