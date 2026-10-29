import { api, type DiffractionRead, type RenderParams } from '../api.js';
import { createDownloadBar } from '../components/download_bar.js';
import { createImagePanel, type ImagePanel } from '../components/image_panel.js';
import { createTree, type TreeColumn, type TreeNode } from '../components/tree.js';
import { buildPageLayout, infoLabel } from '../layout.js';
import {
  MISSING,
  dtypeItemsize,
  fmt,
  fmtBytes,
  fmtInt,
  ingestRowState,
  infoText,
  scale,
} from '../format.js';

// The desktop's columns, less Counts and the physical pixel pair: diffraction.h5 holds
// the assembled patterns, after cropping and binning, and records neither the raw
// detector's pitch nor a per-pattern sum.
const PROCESSED = 'The assembled patterns, after cropping and binning';
const COLUMNS: TreeColumn[] = [
  'Name',
  'Frames',
  { header: 'Width\n[px]', title: PROCESSED },
  { header: 'Height\n[px]', title: PROCESSED },
  'Processed Pixel\nWidth [µm]',
  'Processed Pixel\nHeight [µm]',
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
  image.enableControls((params) => redraw?.(params), { isComplex: false });

  let items: DiffractionRead[] = [];
  let total = 0;
  try {
    const listing = await api.listDiffraction();
    items = listing.items;
    total = listing.total;
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
  left.appendChild(infoLabel(infoText('Datasets', total, items.map((d) => d.nbytes))));
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
    draw(image.params());
  };

  const itemsize = dtypeItemsize(d.pattern_dtype);
  const frameBytes =
    itemsize === null || d.pattern_width_px === null || d.pattern_height_px === null
      ? null
      : d.pattern_width_px * d.pattern_height_px * itemsize;
  const state = ingestRowState(d);

  return {
    id: `diff:${d.uuid}`,
    label,
    state,
    cells: [
      String(total),
      fmtInt(d.pattern_width_px),
      fmtInt(d.pattern_height_px),
      fmt(scale(d.detector_pixel_width_m, 1e6)),
      fmt(scale(d.detector_pixel_height_m, 1e6)),
      fmtInt(d.num_bad_pixels),
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
        state,
        // The extent, pitch and bad pixels describe the dataset, so the desktop
        // leaves them blank on a frame.
        cells: ['1', '', '', '', '', '', frameBytes === null ? MISSING : fmtBytes(frameBytes)],
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
