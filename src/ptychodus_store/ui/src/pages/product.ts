import { api, type ProductRead } from '../api.js';
import { createDownloadBar } from '../components/download_bar.js';
import { createTable } from '../components/table.js';
import { buildPageLayout } from '../layout.js';
import { fmt, fmtBytes, scale } from '../format.js';

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
      {
        header: 'Diffraction\nDataset',
        render: (p) => p.derived_from.find((e) => e.kind === 'diffraction')?.uuid.slice(0, 8) ?? '—',
      },
      {
        header: 'Detector-Object\nDistance [m]',
        render: (p) => fmt(p.detector_distance_m),
        sortKey: (p) => p.detector_distance_m,
        numeric: true,
      },
      {
        header: 'Photon Energy\n[keV]',
        render: (p) => fmt(scale(p.photon_energy_eV, 1e-3)),
        sortKey: (p) => p.photon_energy_eV,
        numeric: true,
      },
      {
        header: 'Probe Photon\nCount',
        render: (p) => fmt(p.probe_photon_count),
        sortKey: (p) => p.probe_photon_count,
        numeric: true,
      },
      {
        header: 'Pixel Width\n[nm]',
        render: (p) => fmt(scale(p.object_pixel_width_m, 1e9)),
        sortKey: (p) => p.object_pixel_width_m,
        numeric: true,
      },
      {
        header: 'Pixel Height\n[nm]',
        render: (p) => fmt(scale(p.object_pixel_height_m, 1e9)),
        sortKey: (p) => p.object_pixel_height_m,
        numeric: true,
      },
      {
        header: 'Size',
        render: (p) => fmtBytes(totalBytes(p)),
        sortKey: (p) => totalBytes(p),
        numeric: true,
      },
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
    ['Photon Wavelength [nm]', fmt(scale(reciprocal(p.probe_wavenumber_per_m), 1e9))],
    ['Photon Wavenumber [1/nm]', fmt(scale(p.probe_wavenumber_per_m, 1e-9))],
    ['Photon Angular Wavenumber [rad/nm]', fmt(scale(p.probe_angular_wavenumber_rad_per_m, 1e-9))],
    ['Probe Photon Flux [ph/s]', fmt(p.probe_photon_flux_per_s)],
    ['Probe Power [W]', fmt(p.probe_power_W)],
    ['Object Plane Pixel Width [nm]', fmt(scale(p.object_pixel_width_m, 1e9))],
    ['Object Plane Pixel Height [nm]', fmt(scale(p.object_pixel_height_m, 1e9))],
    ['Exposure Time [s]', fmt(p.exposure_time_s)],
    ['Mass Attenuation [m²/kg]', fmt(p.mass_attenuation_m2_per_kg)],
    ['Tomography Angle [deg]', fmt(p.tomography_angle_deg)],
    ['Tilt Angle [deg]', fmt(p.tilt_angle_deg)],
    ['Polarization', p.polarization ?? '—'],
    ['Fresnel Number', fmt(p.fresnel_number)],
    ['Detector Numerical Aperture', fmt(p.detector_numerical_aperture)],
    ['Depth of Field [nm]', fmt(scale(p.depth_of_field_m, 1e9))],
    [
      'Diffraction Dataset',
      p.derived_from.find((e) => e.kind === 'diffraction')?.uuid ?? '—',
    ],
    ['Focus-Object Distance [mm]', fmt(scale(p.focus_object_distance_m, 1e3))],
    ['Far Field', p.far_field === null ? '—' : p.far_field ? 'Far Field' : 'Near Field'],
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

/** Wavelength from the wavenumber the service derived, so nothing is recomputed here. */
function reciprocal(x: number | null): number | null {
  return x === null || x === 0 ? null : 1 / x;
}

function totalBytes(p: ProductRead): number | null {
  const parts = [p.probe_nbytes, p.object_nbytes, p.scan_nbytes].filter(
    (n): n is number => n !== null
  );
  return parts.length === 0 ? null : parts.reduce((a, b) => a + b, 0);
}
