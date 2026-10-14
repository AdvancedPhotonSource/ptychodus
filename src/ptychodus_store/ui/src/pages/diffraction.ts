import { api, type DiffractionRead, type RenderParams } from '../api.js';
import { createDownloadBar } from '../components/download_bar.js';
import { createImagePanel, type ImagePanel } from '../components/image_panel.js';
import { createTree, type TreeNode } from '../components/tree.js';
import { buildPageLayout } from '../layout.js';
import { fmt, fmtBytes, scale } from '../format.js';

const COLUMNS = [
  'Name',
  'Frames',
  'Width [px]',
  'Height [px]',
  'Physical Pixel\nWidth [µm]',
  'Physical Pixel\nHeight [µm]',
  'Num Bad\nPixels',
  'Size',
];

export async function mountDiffraction(root: HTMLElement): Promise<void> {
  const { page, left, right, setActiveTab } = buildPageLayout('diffraction');
  root.replaceChildren(page);

  const tree = createTree(COLUMNS);
  left.replaceChildren(tree.el);
  const downloadHost = document.createElement('div');
  const image = createImagePanel();
  right.replaceChildren(downloadHost, image.el);
  image.setEmpty();

  let redraw: ((params: Partial<RenderParams>) => void) | null = null;
  image.enableControls((params) => redraw?.(params));

  let items: DiffractionRead[] = [];
  try {
    items = (await api.listDiffraction()).items;
  } catch (err) {
    left.replaceChildren(errorBlock(err as Error));
    return;
  }

  if (items.length === 0) {
    left.replaceChildren(emptyBlock('No diffraction datasets in store.'));
    return;
  }

  tree.setNodes(
    items.map((d) => datasetNode(d, image, downloadHost, setActiveTab, (fn) => (redraw = fn)))
  );
}

function datasetNode(
  d: DiffractionRead,
  image: ImagePanel,
  downloadHost: HTMLElement,
  setActiveTab: (which: 'left' | 'right') => void,
  remember: (fn: (params: Partial<RenderParams>) => void) => void
): TreeNode {
  const label = d.label || d.uuid.slice(0, 8);
  const total = d.num_patterns_total ?? 0;

  const show = (
    title: string,
    fetch: (params: Partial<RenderParams>) => Promise<Parameters<ImagePanel['setImage']>[0]>
  ) => {
    const draw = (params: Partial<RenderParams>): void => {
      image.setLoading(title);
      fetch(params)
        .then((img) => image.setImage(img, `${label} — ${title}`))
        .catch((err: Error) => image.setError(err));
    };
    setActiveTab('right');
    downloadHost.replaceChildren(createDownloadBar(api.diffractionFileUrl(d.uuid), `${label}.h5`));
    // A control change re-fetches whatever is on screen, not the first thing drawn.
    remember(draw);
    draw({});
  };

  return {
    id: `diff:${d.uuid}`,
    label,
    cells: [
      String(total),
      fmt(d.pattern_width_px),
      fmt(d.pattern_height_px),
      fmt(scale(d.detector_pixel_width_m, 1e6)),
      fmt(scale(d.detector_pixel_height_m, 1e6)),
      fmt(d.num_bad_pixels),
      fmtBytes(d.nbytes),
    ],
    // Selecting the dataset itself shows the mean pattern, as the desktop tree does;
    // there is no synthetic "aggregate" child.
    onSelect: () => show('mean pattern', (q) => api.diffractionAggregateImage(d.uuid, q)),
    // One node per frame, built only near the viewport: a dataset runs to tens of
    // thousands, which the DOM will not hold.
    windowedChildren: {
      count: total,
      build: (i: number) => ({
        id: `diff:${d.uuid}:${i}`,
        label: `Frame ${i}`,
        cells: ['1', fmt(d.pattern_width_px), fmt(d.pattern_height_px), '—', '—', '—', '—'],
        onSelect: () => show(`frame ${i}`, (q) => api.diffractionPatternImage(d.uuid, i, q)),
      }),
    },
  };
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
