import {
  api,
  type PlotImage,
  type RenderParams,
  type RenderedImage,
  type VisualizationOptions,
} from '../api.js';
import { formatG } from '../format.js';

export interface ImagePanel {
  el: HTMLElement;
  /**
   * Show the Colorize and Data Range controls, re-fetching through `refresh`
   * whenever one changes. Omitted on a page whose output is a plot: a plot carries
   * its own axes and takes no render parameters, exactly as the desktop panel has
   * no ribbon there. `isComplex` offers the complex renderers, as the desktop does
   * for the probe and object but not for patterns or element maps.
   */
  enableControls: (
    refresh: (params: Partial<RenderParams>) => void,
    options: { isComplex: boolean }
  ) => void;
  /** What the ribbon currently asks for; render a newly selected item with these. */
  params: () => Partial<RenderParams>;
  setEmpty: (message?: string) => void;
  setLoading: (label: string) => void;
  setError: (err: Error) => void;
  setImage: (img: RenderedImage, title: string) => void;
  // A plot carries its own axes, so it has no colorbar or pixel size to caption.
  setPlot: (img: PlotImage, title: string) => void;
}

/** The renderer the desktop's Complex entry stands for: a color model, not a component. */
const COMPLEX = 'complex';

interface Choice {
  value: string;
  label: string;
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

  let current: Partial<RenderParams> = {};
  let legendVisible = false;
  let lastImage: RenderedImage | null = null;
  // Written back after every render, as the desktop slider is after autoscaling.
  let rangeInputs: { min: HTMLInputElement; max: HTMLInputElement } | null = null;

  // Mirrors the desktop ImageRibbon, minus the image tools: the pan, ruler and
  // line-cut buttons act on a canvas this panel does not have.
  function enableControls(
    refresh: (params: Partial<RenderParams>) => void,
    { isComplex }: { isComplex: boolean }
  ): void {
    // The desktop's defaults, usable before the options arrive: Complex with HSV-V
    // for a complex array, Real with gray otherwise.
    let renderer = isComplex ? COMPLEX : 'real';
    let transform = 'identity';
    // One choice per kind of variant, kept across renderer changes as the desktop
    // shares one colormap parameter among its acyclic renderers.
    const variant = { linear: 'gray', cyclic: 'colorwheel', model: 'hsv_value' };
    let range: { min: number | null; max: number | null } = { min: null, max: null };
    let clip = false;

    const variantKind = (): keyof typeof variant =>
      renderer === COMPLEX ? 'model' : renderer === 'phase_rad' ? 'cyclic' : 'linear';

    const compose = (): Partial<RenderParams> => {
      const params: Partial<RenderParams> = {
        transform,
        value_min: range.min,
        value_max: range.max,
        clip,
      };
      if (renderer === COMPLEX) {
        params.color_model = variant.model;
      } else {
        params.colormap = variantKind() === 'cyclic' ? variant.cyclic : variant.linear;
        // A real-valued array takes no component: it is already the Real renderer.
        if (isComplex) params.component = renderer;
      }
      return params;
    };

    const apply = (): void => {
      current = compose();
      refresh({ ...current });
    };

    current = compose();

    api
      .visualizationOptions()
      .then((opts) => {
        const variantSelect = document.createElement('select');
        variantSelect.title = 'Variant';
        const fillVariants = (): void => {
          const kind = variantKind();
          variantSelect.replaceChildren(...variantChoices(opts, kind).map(toOption));
          variantSelect.value = variant[kind];
          // A default the service does not offer falls back to its first choice.
          variant[kind] = variantSelect.value;
        };
        variantSelect.addEventListener('change', () => {
          variant[variantKind()] = variantSelect.value;
          apply();
        });

        const rendererSelect = select(
          'Array Component',
          rendererChoices(opts, isComplex),
          renderer
        );
        rendererSelect.addEventListener('change', () => {
          renderer = rendererSelect.value;
          fillVariants();
          apply();
        });

        const transformSelect = select(
          'Transformation',
          opts.transforms.map((v) => ({ value: v, label: opts.transform_labels[v] ?? v })),
          transform
        );
        transformSelect.addEventListener('change', () => {
          transform = transformSelect.value;
          apply();
        });

        fillVariants();
        const colorize = group('Colorize');
        colorize.append(rendererSelect, transformSelect, variantSelect);

        const min = numberInput('Display minimum');
        const max = numberInput('Display maximum');
        rangeInputs = { min, max };
        // Editing either bound pins both, so the other keeps the value it shows
        // rather than snapping back to the data's range.
        const pin = (): void => {
          range = {
            min: min.value === '' ? null : Number(min.value),
            max: max.value === '' ? null : Number(max.value),
          };
          apply();
        };
        min.addEventListener('change', pin);
        max.addEventListener('change', pin);

        const auto = button('Auto', 'Rescale to Data Range');
        auto.addEventListener('click', () => {
          range = { min: null, max: null };
          apply();
        });

        const legend = button('Color Legend', 'Toggle Color Legend Visibility');
        legend.setAttribute('aria-pressed', String(legendVisible));
        legend.addEventListener('click', () => {
          legendVisible = !legendVisible;
          legend.setAttribute('aria-pressed', String(legendVisible));
          drawLegend();
        });

        const clipLabel = document.createElement('label');
        clipLabel.textContent = 'clip';
        const clipBox = document.createElement('input');
        clipBox.type = 'checkbox';
        clipBox.addEventListener('change', () => {
          clip = clipBox.checked;
          apply();
        });
        clipLabel.appendChild(clipBox);

        const dataRange = group('Data Range');
        dataRange.append(min, max, auto, legend, clipLabel);
        ribbon.replaceChildren(colorize, dataRange);
        ribbon.hidden = false;

        if (lastImage !== null) writeBackRange(lastImage);
      })
      .catch(() => {
        // The ribbon is an enhancement; a failed options fetch leaves the default
        // render in place rather than blanking the panel.
        ribbon.hidden = true;
      });
  }

  function writeBackRange(img: RenderedImage): void {
    if (rangeInputs === null) return;
    rangeInputs.min.value = formatG(img.color_value_min, 6);
    rangeInputs.max.value = formatG(img.color_value_max, 6);
  }

  function drawLegend(): void {
    wrap.querySelector('.color-legend')?.remove();
    if (!legendVisible || lastImage === null || lastImage.legend_rgba.length === 0) return;
    wrap.appendChild(buildLegend(lastImage));
  }

  return {
    el,
    enableControls,
    params: () => ({ ...current }),
    setEmpty(msg = 'Select an item to preview.') {
      lastImage = null;
      wrap.replaceChildren(makeMessage('empty', msg));
      caption.textContent = '';
    },
    setLoading(label) {
      wrap.replaceChildren(makeMessage('status', `Loading ${label}…`));
      caption.textContent = '';
    },
    setError(err) {
      lastImage = null;
      wrap.replaceChildren(makeMessage('error', err.message));
      caption.textContent = '';
    },
    setPlot(img, title) {
      lastImage = null;
      const el = document.createElement('img');
      el.src = `data:${img.mime_type};base64,${img.png_base64}`;
      el.alt = title;
      wrap.replaceChildren(el);
      caption.innerHTML = `<div><strong>${escapeHtml(title)}</strong></div>`;
    },
    setImage(img, title) {
      lastImage = img;
      const el = document.createElement('img');
      el.src = `data:${img.mime_type};base64,${img.png_base64}`;
      el.alt = title;
      wrap.replaceChildren(el);
      drawLegend();
      writeBackRange(img);
      const pw_um = img.pixel_width_m * 1e6;
      const ph_um = img.pixel_height_m * 1e6;
      caption.innerHTML = `
        <div><strong>${escapeHtml(title)}</strong></div>
        <div>${escapeHtml(img.value_label)} — range [${formatG(img.color_value_min)}, ${formatG(img.color_value_max)}]</div>
        <div>${img.shape_w_px} × ${img.shape_h_px} px — pixel ${pw_um.toFixed(3)} × ${ph_um.toFixed(3)} µm</div>
      `;
    },
  };
}

/** The desktop's renderer combo: every component plus Complex, sorted by name. */
function rendererChoices(opts: VisualizationOptions, isComplex: boolean): Choice[] {
  if (!isComplex) {
    return [{ value: 'real', label: opts.component_labels['real'] ?? 'Real' }];
  }

  const choices = opts.components.map((v) => ({ value: v, label: opts.component_labels[v] ?? v }));
  choices.push({ value: COMPLEX, label: 'Complex' });
  return choices.sort((a, b) => a.label.localeCompare(b.label));
}

function variantChoices(opts: VisualizationOptions, kind: 'linear' | 'cyclic' | 'model'): Choice[] {
  switch (kind) {
    case 'model':
      return opts.color_models.map((v) => ({ value: v, label: opts.color_model_labels[v] ?? v }));
    case 'cyclic':
      return opts.colormaps_cyclic.map((v) => ({ value: v, label: v }));
    case 'linear':
      return opts.colormaps_linear.map((v) => ({ value: v, label: v }));
  }
}

function buildLegend(img: RenderedImage): HTMLElement {
  const colors = img.legend_rgba.map(
    ([r, g, b, a]) =>
      `rgba(${Math.round(255 * r)}, ${Math.round(255 * g)}, ${Math.round(255 * b)}, ${a})`
  );
  const legend = document.createElement('div');

  if (img.legend_cyclic) {
    // Counterclockwise from twelve o'clock, as the desktop's conical gradient runs.
    legend.className = 'color-legend cyclic';
    legend.style.background = `conic-gradient(${[...colors].reverse().join(', ')})`;
    return legend;
  }

  legend.className = 'color-legend linear';
  const bar = document.createElement('div');
  bar.className = 'color-legend-bar';
  bar.style.background = `linear-gradient(to top, ${colors.join(', ')})`;

  const ticks = document.createElement('div');
  ticks.className = 'color-legend-ticks';
  const numTicks = 5;
  for (let i = numTicks - 1; i >= 0; i--) {
    const a = i / (numTicks - 1);
    const value = (1 - a) * img.color_value_min + a * img.color_value_max;
    const tick = document.createElement('span');
    tick.textContent = formatG(value, 6);
    ticks.appendChild(tick);
  }

  legend.append(bar, ticks);
  return legend;
}

function toOption(choice: Choice): HTMLOptionElement {
  return new Option(choice.label, choice.value);
}

function select(title: string, choices: Choice[], value: string): HTMLSelectElement {
  const el = document.createElement('select');
  el.title = title;
  el.setAttribute('aria-label', title);
  el.append(...choices.map(toOption));
  el.value = value;
  return el;
}

function numberInput(title: string): HTMLInputElement {
  const el = document.createElement('input');
  el.type = 'number';
  el.step = 'any';
  el.title = title;
  el.setAttribute('aria-label', title);
  return el;
}

function button(text: string, title: string): HTMLButtonElement {
  const el = document.createElement('button');
  el.type = 'button';
  el.textContent = text;
  el.title = title;
  return el;
}

function makeMessage(cls: string, text: string): HTMLElement {
  const div = document.createElement('div');
  div.className = cls;
  div.textContent = text;
  return div;
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
