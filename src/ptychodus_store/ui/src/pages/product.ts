import { api, type ProductRead } from '../api.js';
import { createImagePanel } from '../components/image_panel.js';
import { createTable } from '../components/table.js';
import { buildPageLayout } from '../layout.js';

type SubKind = 'object' | 'probe' | 'positions';

export async function mountProduct(root: HTMLElement): Promise<void> {
  const { page, left, right, setActiveTab } = buildPageLayout('product');
  root.replaceChildren(page);

  const image = createImagePanel();
  const picker = document.createElement('div');
  picker.className = 'sub-picker';
  right.replaceChildren(picker, image.el);
  image.setEmpty('Select a product row.');

  let items: ProductRead[] = [];
  try {
    const listing = await api.listProduct();
    items = listing.items;
  } catch (err) {
    left.replaceChildren(errorBlock(err as Error));
    return;
  }

  if (items.length === 0) {
    left.replaceChildren(emptyBlock('No products in store.'));
    return;
  }

  const table = createTable<ProductRead>(
    [
      { header: 'Name', render: (p) => p.name ?? p.uuid.slice(0, 8) },
      { header: 'Layers', render: (p) => str(p.object_layers) },
      { header: 'Modes', render: (p) => str(p.probe_modes) },
      { header: 'Scan pts', render: (p) => str(p.num_scan_points) },
      { header: 'State', render: (p) => p.ingest_state },
    ],
    (row) => selectProduct(row)
  );
  left.replaceChildren(table.el);
  table.setRows(items);

  let selected: ProductRead | null = null;
  let subKind: SubKind = 'object';
  let objectLayer = 0;
  let probeMode = 0;

  function refresh(): void {
    if (!selected) return;
    setActiveTab('right');
    if (subKind === 'object') {
      const layer = clampInt(objectLayer, 0, (selected.object_layers ?? 1) - 1);
      image.setLoading(`object layer ${layer}`);
      api
        .productObjectImage(selected.uuid, layer)
        .then((img) => image.setImage(img, `${labelFor(selected!)} — object[${layer}]`))
        .catch((err: Error) => image.setError(err));
    } else if (subKind === 'probe') {
      const mode = clampInt(probeMode, 0, (selected.probe_modes ?? 1) - 1);
      image.setLoading(`probe mode ${mode}`);
      api
        .productProbeImage(selected.uuid, mode)
        .then((img) => image.setImage(img, `${labelFor(selected!)} — probe mode ${mode}`))
        .catch((err: Error) => image.setError(err));
    } else {
      image.setLoading('probe positions');
      api
        .productPositionsImage(selected.uuid)
        .then((img) => image.setImage(img, `${labelFor(selected!)} — positions`))
        .catch((err: Error) => image.setError(err));
    }
  }

  function rebuildPicker(): void {
    picker.replaceChildren();
    if (!selected) return;
    const kinds: { key: SubKind; label: string }[] = [
      { key: 'object', label: 'Object' },
      { key: 'probe', label: 'Probe' },
      { key: 'positions', label: 'Positions' },
    ];
    for (const k of kinds) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.textContent = k.label;
      if (k.key === subKind) btn.classList.add('active');
      btn.addEventListener('click', () => {
        subKind = k.key;
        rebuildPicker();
        refresh();
      });
      picker.appendChild(btn);
    }
    if (subKind === 'object' && (selected.object_layers ?? 1) > 1) {
      picker.appendChild(indexInput(objectLayer, selected.object_layers! - 1, (v) => {
        objectLayer = v;
        refresh();
      }));
    }
    if (subKind === 'probe' && (selected.probe_modes ?? 1) > 1) {
      picker.appendChild(indexInput(probeMode, selected.probe_modes! - 1, (v) => {
        probeMode = v;
        refresh();
      }));
    }
  }

  function selectProduct(row: ProductRead): void {
    selected = row;
    subKind = 'object';
    objectLayer = 0;
    probeMode = 0;
    rebuildPicker();
    refresh();
  }
}

function labelFor(p: ProductRead): string {
  return p.name ?? p.uuid.slice(0, 8);
}

function str(x: number | null): string {
  return x === null ? '—' : String(x);
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
  label.textContent = `index (0–${max})`;
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

function errorBlock(err: Error): HTMLElement {
  const el = document.createElement('div');
  el.style.color = '#ff8080';
  el.style.padding = '1rem';
  el.textContent = err.message;
  return el;
}

function emptyBlock(msg: string): HTMLElement {
  const el = document.createElement('div');
  el.style.color = 'var(--fg-muted)';
  el.style.padding = '1rem';
  el.style.fontStyle = 'italic';
  el.textContent = msg;
  return el;
}
