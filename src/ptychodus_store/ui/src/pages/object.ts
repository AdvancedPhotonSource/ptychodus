import { api, type ProductRead } from '../api.js';
import { createImagePanel } from '../components/image_panel.js';
import { createTree, type TreeNode } from '../components/tree.js';
import { buildPageLayout } from '../layout.js';
import { fmt, fmtBytes, scale } from '../format.js';

const COLUMNS = [
  'Name',
  'Distance [m]',
  'Data Type',
  'Width [px]',
  'Height [px]',
  'Pixel Width\n[nm]',
  'Pixel Height\n[nm]',
  'Size',
];

export async function mountObject(root: HTMLElement): Promise<void> {
  const { page, left, right, setActiveTab } = buildPageLayout('object');
  root.replaceChildren(page);

  const image = createImagePanel();
  right.replaceChildren(image.el);
  image.setEmpty('Select an object layer.');

  const tree = createTree(COLUMNS);
  left.replaceChildren(tree.el);

  let items: ProductRead[] = [];
  try {
    items = (await api.listProduct()).items;
  } catch (err) {
    left.replaceChildren(errorBlock(err as Error));
    return;
  }

  const show = (p: ProductRead, layer: number) => {
    setActiveTab('right');
    image.setLoading(`object layer ${layer}`);
    const label = p.name ?? p.uuid.slice(0, 8);
    api
      .productObjectImage(p.uuid, layer)
      .then((img) => image.setImage(img, `${label} — object[${layer}]`))
      .catch((err: Error) => image.setError(err));
  };

  tree.setNodes(
    items.map((p): TreeNode => {
      const layers = p.object_layers ?? 0;
      const common = [
        p.object_dtype ?? '—',
        fmt(p.object_width_px),
        fmt(p.object_height_px),
        fmt(scale(p.object_pixel_width_m, 1e9)),
        fmt(scale(p.object_pixel_height_m, 1e9)),
      ];
      return {
        id: `object:${p.uuid}`,
        label: p.name ?? p.uuid.slice(0, 8),
        cells: ['—', ...common, fmtBytes(p.object_nbytes)],
        onSelect: () => show(p, 0),
        children: Array.from({ length: layers }, (_, i) => ({
          id: `object:${p.uuid}:${i}`,
          label: `Layer ${i}`,
          // N layers have N-1 spacings, so the last layer has no distance to the next.
          cells: [fmt(p.object_layer_spacing_m[i]), ...common, '—'],
          onSelect: () => show(p, i),
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
