import { api, type ProductRead } from '../api.js';
import { createImagePanel } from '../components/image_panel.js';
import { createProductPicker } from '../components/product_picker.js';
import { buildPageLayout } from '../layout.js';

export async function mountObject(root: HTMLElement): Promise<void> {
  const { page, left, right, setActiveTab } = buildPageLayout('object');
  root.replaceChildren(page);

  const image = createImagePanel();
  const picker = document.createElement('div');
  picker.className = 'sub-picker';
  right.replaceChildren(picker, image.el);
  image.setEmpty('Select a product to view its reconstructed object.');

  let selected: ProductRead | null = null;
  let layer = 0;

  const productPicker = createProductPicker((product) => selectProduct(product));
  left.replaceChildren(productPicker.el);
  await productPicker.load();

  function selectProduct(product: ProductRead): void {
    selected = product;
    layer = 0;
    rebuildPicker();
    render();
  }

  function rebuildPicker(): void {
    picker.replaceChildren();
    if (!selected) return;
    const layers = selected.object_layers ?? 1;
    if (layers <= 1) return;
    picker.appendChild(indexInput(layer, layers - 1, (v) => {
      layer = v;
      render();
    }));
  }

  function render(): void {
    if (!selected) return;
    setActiveTab('right');
    const clamped = clampInt(layer, 0, (selected.object_layers ?? 1) - 1);
    image.setLoading(`object layer ${clamped}`);
    api
      .productObjectImage(selected.uuid, clamped)
      .then((img) => image.setImage(img, `${labelFor(selected!)} — object[${clamped}]`))
      .catch((err: Error) => image.setError(err));
  }
}

function labelFor(p: ProductRead): string {
  return p.name ?? p.uuid.slice(0, 8);
}

function clampInt(v: number, lo: number, hi: number): number {
  if (hi < lo) return lo;
  return Math.max(lo, Math.min(hi, v | 0));
}

function indexInput(value: number, max: number, onChange: (v: number) => void): HTMLElement {
  const wrap = document.createElement('label');
  wrap.style.display = 'inline-flex';
  wrap.style.alignItems = 'center';
  wrap.style.gap = '0.25rem';
  const label = document.createElement('span');
  label.textContent = `layer (0–${max})`;
  label.style.color = 'var(--fg-muted)';
  label.style.fontSize = '0.85em';
  const input = document.createElement('input');
  input.type = 'number';
  input.min = '0';
  input.max = String(max);
  input.value = String(value);
  input.addEventListener('change', () => {
    const v = parseInt(input.value, 10);
    if (Number.isFinite(v)) onChange(v);
  });
  wrap.append(label, input);
  return wrap;
}
