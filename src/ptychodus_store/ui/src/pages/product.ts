import { api, type ProductRead } from '../api.js';
import { createDownloadBar } from '../components/download_bar.js';
import { createTable } from '../components/table.js';
import { buildPageLayout } from '../layout.js';

export async function mountProduct(root: HTMLElement): Promise<void> {
  const { page, left, right, setActiveTab } = buildPageLayout('product');
  root.replaceChildren(page);

  const detail = document.createElement('div');
  detail.className = 'detail-panel';
  right.replaceChildren(detail);
  showEmpty(detail);

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
      { header: 'Detector-Object\nDistance [m]', render: (p) => fmt(p.detector_distance_m) },
      { header: 'Probe Energy\n[keV]', render: (p) => fmt(scale(p.probe_energy_eV, 1e-3)) },
      { header: 'Probe Photon\nCount', render: (p) => fmt(p.probe_photon_count) },
      { header: 'Pixel Width\n[nm]', render: (p) => fmt(scale(p.object_pixel_width_m, 1e9)) },
      { header: 'Pixel Height\n[nm]', render: (p) => fmt(scale(p.object_pixel_height_m, 1e9)) },
      { header: 'State', render: (p) => p.ingest_state },
    ],
    (row) => {
      setActiveTab('right');
      showDetail(detail, row);
    }
  );
  left.replaceChildren(table.el);
  table.setRows(items);
}

function showEmpty(host: HTMLElement): void {
  host.replaceChildren(emptyBlock('Select a product row.'));
}

function showDetail(host: HTMLElement, p: ProductRead): void {
  host.replaceChildren();
  host.appendChild(createDownloadBar(api.productFileUrl(p.uuid), `${p.name ?? p.uuid}.h5`));

  const title = document.createElement('h2');
  title.textContent = p.name ?? p.uuid;
  host.appendChild(title);

  const dl = document.createElement('dl');
  dl.className = 'detail-list';
  const rows: [string, string][] = [
    ['UUID', p.uuid],
    ['State', p.ingest_state],
    ['Detector-Object Distance [m]', fmt(p.detector_distance_m)],
    ['Probe Energy [keV]', fmt(scale(p.probe_energy_eV, 1e-3))],
    ['Probe Photon Count', fmt(p.probe_photon_count)],
    ['Probe Modes', fmt(p.probe_modes)],
    ['Probe Shape [px]', shape(p.probe_height_px, p.probe_width_px)],
    ['Object Layers', fmt(p.object_layers)],
    ['Object Shape [px]', shape(p.object_height_px, p.object_width_px)],
    ['Object Pixel Width [nm]', fmt(scale(p.object_pixel_width_m, 1e9))],
    ['Object Pixel Height [nm]', fmt(scale(p.object_pixel_height_m, 1e9))],
    ['Scan Points', fmt(p.num_scan_points)],
    ['Tomography Angle [deg]', fmt(p.tomography_angle_deg)],
    ['Tilt Angle [deg]', fmt(p.tilt_angle_deg)],
    ['Polarization', p.polarization ?? '—'],
  ];
  for (const [k, v] of rows) {
    const dt = document.createElement('dt');
    dt.textContent = k;
    const dd = document.createElement('dd');
    dd.textContent = v;
    dl.append(dt, dd);
  }
  host.appendChild(dl);

  if (p.comments) {
    const h3 = document.createElement('h3');
    h3.textContent = 'Comments';
    const pre = document.createElement('pre');
    pre.className = 'detail-comments';
    pre.textContent = p.comments;
    host.append(h3, pre);
  }
}

function scale(x: number | null, factor: number): number | null {
  return x === null ? null : x * factor;
}

function fmt(x: number | null): string {
  if (x === null || !Number.isFinite(x)) return '—';
  return Number(x).toPrecision(4);
}

function shape(h: number | null, w: number | null): string {
  if (h === null || w === null) return '—';
  return `${h} × ${w}`;
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
