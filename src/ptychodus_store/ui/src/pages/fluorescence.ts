import { api, type FluorescenceRead } from '../api.js';
import { createDownloadBar } from '../components/download_bar.js';
import { createImagePanel, type ImagePanel } from '../components/image_panel.js';
import { createTree, type TreeNode } from '../components/tree.js';
import { fmt, fmtBytes } from '../format.js';
import { buildPageLayout } from '../layout.js';

export async function mountFluorescence(root: HTMLElement): Promise<void> {
  const { page, left, right, setActiveTab } = buildPageLayout('fluorescence');
  root.replaceChildren(page);

  const tree = createTree(['Name', 'Product', 'Elements', 'Counts', 'Size']);
  left.replaceChildren(tree.el);
  const downloadHost = document.createElement('div');
  const image = createImagePanel();
  right.replaceChildren(downloadHost, image.el);
  image.setEmpty();

  let items: FluorescenceRead[] = [];
  try {
    const listing = await api.listFluorescence();
    items = listing.items;
  } catch (err) {
    left.replaceChildren(errorBlock(err as Error));
    return;
  }

  if (items.length === 0) {
    left.replaceChildren(emptyBlock('No fluorescence datasets in store.'));
    return;
  }

  const nodes: TreeNode[] = items.map((f) => ({
    id: `flu:${f.uuid}`,
    label: f.label || f.uuid.slice(0, 8),
    cells: [
      productOf(f)?.slice(0, 8) ?? '—',
      String(f.element_names.length),
      fmt(f.element_counts.reduce((a, b) => a + b, 0)),
      fmtBytes(f.nbytes),
    ],
    loadChildren: () => loadElementNodes(f, image, downloadHost, setActiveTab),
  }));
  tree.setNodes(nodes);
}

async function loadElementNodes(
  f: FluorescenceRead,
  image: ImagePanel,
  downloadHost: HTMLElement,
  setActiveTab: (which: 'left' | 'right') => void
): Promise<TreeNode[]> {
  const detail = await api.getFluorescence(f.uuid);
  downloadHost.replaceChildren(
    createDownloadBar(api.fluorescenceFileUrl(f.uuid), `${detail.label || f.uuid}.h5`)
  );
  const elements = detail.element_names ?? [];
  if (elements.length === 0) {
    return [{ id: `flu:${f.uuid}:none`, label: '(no elements)' }];
  }
  // Without the product the map renders at a 1 um placeholder pitch and the caption
  // reports a physical size that is simply wrong.
  const product = productOf(detail);

  return elements.map((name, i) => ({
    id: `flu:${f.uuid}:${name}`,
    label: name,
    cells: ['—', '—', fmt(detail.element_counts[i]), '—'],
    onSelect: () => {
      setActiveTab('right');
      image.setLoading(`element ${name}`);
      api
        .fluorescenceElementImage(f.uuid, name, product)
        .then((img) => image.setImage(img, `${detail.label || f.uuid} — ${name}`))
        .catch((err: Error) => image.setError(err));
    },
  }));
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

/** UUID of the product this map derives from, if the manifest records one. */
function productOf(f: FluorescenceRead): string | undefined {
  return f.derived_from.find((e) => e.kind === 'product')?.uuid;
}
