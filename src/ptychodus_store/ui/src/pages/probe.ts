import { api, type ProductRead } from '../api.js';
import { createImagePanel } from '../components/image_panel.js';
import { createTree, type TreeNode } from '../components/tree.js';
import { buildPageLayout } from '../layout.js';
import { fmt, fmtBytes, fmtPercent, scale } from '../format.js';

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

  const tree = createTree(COLUMNS);
  left.replaceChildren(tree.el);

  let items: ProductRead[] = [];
  try {
    items = (await api.listProduct()).items;
  } catch (err) {
    left.replaceChildren(errorBlock(err as Error));
    return;
  }

  const show = (p: ProductRead, title: string, incoherent: number | null) => {
    setActiveTab('right');
    image.setLoading(title);
    const label = p.name ?? p.uuid.slice(0, 8);
    const fetch =
      incoherent === null
        ? api.productProbeModesImage(p.uuid)
        : api.productProbeImage(p.uuid, incoherent);
    fetch
      .then((img) => image.setImage(img, `${label} — ${title}`))
      .catch((err: Error) => image.setError(err));
  };

  tree.setNodes(
    items.map((p): TreeNode => {
      const modes = p.probe_modes ?? 0;
      const common = [
        p.probe_dtype ?? '—',
        fmt(p.probe_width_px),
        fmt(p.probe_height_px),
        fmt(scale(p.object_pixel_width_m, 1e9)),
        fmt(scale(p.object_pixel_height_m, 1e9)),
      ];
      return {
        id: `probe:${p.uuid}`,
        label: p.name ?? p.uuid.slice(0, 8),
        cells: ['—', ...common, fmtBytes(p.probe_nbytes)],
        onSelect: () => show(p, 'all modes', null),
        children: Array.from({ length: modes }, (_, i) => ({
          id: `probe:${p.uuid}:${i}`,
          label: `Mode ${i}`,
          cells: [fmtPercent(p.probe_mode_relative_power[i]), ...common, '—'],
          onSelect: () => show(p, `mode ${i}`, i),
        })),
      };
    })
  );
}

function errorBlock(err: Error): HTMLElement {
  const el = document.createElement('div');
  el.style.color = '#ff8080';
  el.style.padding = '1rem';
  el.textContent = err.message;
  return el;
}
