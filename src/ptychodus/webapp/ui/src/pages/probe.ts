import { api, type ProductRead, type RenderParams } from '../api.js';
import { createImagePanel } from '../components/image_panel.js';
import { createTree, type TreeNode } from '../components/tree.js';
import { buildPageLayout, infoLabel } from '../layout.js';
import { fmt, fmtBytes, fmtInt, fmtPercent, ingestRowState, infoText, scale } from '../format.js';

const COLUMNS = [
  'Name',
  'Relative Power',
  'Data Type',
  'Width [px]',
  'Height [px]',
  'Pixel Width\n[nm]',
  'Pixel Height\n[nm]',
  'Size',
];

export async function mountProbe(root: HTMLElement): Promise<void> {
  const { page, left, right, setActiveTab } = buildPageLayout('probe');
  root.replaceChildren(page);

  const image = createImagePanel();
  right.replaceChildren(image.el);
  image.setEmpty('Select a probe or one of its modes.');

  let redraw: ((params: Partial<RenderParams>) => void) | null = null;
  image.enableControls((params) => redraw?.(params), { isComplex: true });

  const tree = createTree(COLUMNS);
  left.replaceChildren(tree.el);

  let items: ProductRead[] = [];
  let total = 0;
  try {
    const listing = await api.listProduct();
    items = listing.items;
    total = listing.total;
  } catch (err) {
    left.replaceChildren(errorBlock(err as Error));
    return;
  }

  const show = (p: ProductRead, title: string, incoherent: number | null) => {
    const label = p.name ?? p.uuid.slice(0, 8);
    const draw = (params: Partial<RenderParams>): void => {
      image.setLoading(title);
      const fetch =
        incoherent === null
          ? api.productProbeModesImage(p.uuid, params)
          : api.productProbeImage(p.uuid, incoherent, params);
      fetch
        .then((img) => image.setImage(img, `${label} — ${title}`))
        .catch((err: Error) => image.setError(err));
    };
    setActiveTab('right');
    redraw = draw;
    draw(image.params());
  };

  tree.setNodes(
    items.map((p): TreeNode => {
      const modes = p.probe_modes ?? 0;
      const state = ingestRowState(p);
      return {
        id: `probe:${p.uuid}`,
        label: p.name ?? p.uuid.slice(0, 8),
        state,
        // The probe's own row reports its coherence, the root of the summed squared
        // mode powers; each mode reports its share.
        cells: [
          fmtPercent(p.probe_coherence),
          p.probe_dtype ?? '—',
          fmtInt(p.probe_width_px),
          fmtInt(p.probe_height_px),
          fmt(scale(p.probe_pixel_width_m, 1e9)),
          fmt(scale(p.probe_pixel_height_m, 1e9)),
          fmtBytes(p.probe_nbytes),
        ],
        onSelect: () => show(p, 'all modes', null),
        children: Array.from({ length: modes }, (_, i) => ({
          id: `probe:${p.uuid}:${i}`,
          label: `Mode ${i + 1}`,
          state,
          cells: [fmtPercent(p.probe_mode_relative_power[i]), '', '', '', '', '', ''],
          onSelect: () => show(p, `mode ${i + 1}`, i),
        })),
      };
    })
  );
  left.appendChild(infoLabel(infoText('Probes', total, items.map((p) => p.probe_nbytes))));
}

function errorBlock(err: Error): HTMLElement {
  const el = document.createElement('div');
  el.style.color = '#ff8080';
  el.style.padding = '1rem';
  el.textContent = err.message;
  return el;
}
