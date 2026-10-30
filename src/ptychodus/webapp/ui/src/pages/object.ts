import { api, type ProductRead, type RenderParams } from '../api.js';
import { createImagePanel } from '../components/image_panel.js';
import { createTree, type TreeNode } from '../components/tree.js';
import { buildPageLayout, infoLabel } from '../layout.js';
import { fmt, fmtBytes, fmtInt, ingestRowState, infoText, scale } from '../format.js';

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
  image.setEmpty('Select an object or one of its layers.');

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

  // A layer of null is the object itself: the product of its layers, as the desktop
  // shows when the object's own row is selected.
  const show = (p: ProductRead, layer: number | null) => {
    const label = p.name ?? p.uuid.slice(0, 8);
    const title = layer === null ? 'object' : `layer ${layer + 1}`;
    const draw = (params: Partial<RenderParams>): void => {
      image.setLoading(title);
      const fetch =
        layer === null
          ? api.productObjectFlattenedImage(p.uuid, params)
          : api.productObjectImage(p.uuid, layer, params);
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
      const layers = p.object_layers ?? 0;
      const state = ingestRowState(p);
      return {
        id: `object:${p.uuid}`,
        label: p.name ?? p.uuid.slice(0, 8),
        state,
        cells: [
          fmt(p.object_total_thickness_m, 6),
          p.object_dtype ?? '—',
          fmtInt(p.object_width_px),
          fmtInt(p.object_height_px),
          fmt(scale(p.object_pixel_width_m, 1e9)),
          fmt(scale(p.object_pixel_height_m, 1e9)),
          fmtBytes(p.object_nbytes),
        ],
        onSelect: () => show(p, null),
        children: Array.from({ length: layers }, (_, i) => ({
          id: `object:${p.uuid}:${i}`,
          label: `Layer ${i + 1}`,
          state,
          // N layers have N-1 spacings, so the last layer has no distance to the next
          // and the desktop leaves it blank.
          cells: [
            i < p.object_layer_spacing_m.length ? fmt(p.object_layer_spacing_m[i], 6) : '',
            '',
            '',
            '',
            '',
            '',
            '',
          ],
          onSelect: () => show(p, i),
        })),
      };
    })
  );
  left.appendChild(infoLabel(infoText('Objects', total, items.map((p) => p.object_nbytes))));
}

function errorBlock(err: Error): HTMLElement {
  const el = document.createElement('div');
  el.style.color = '#ff8080';
  el.style.padding = '1rem';
  el.textContent = err.message;
  return el;
}
