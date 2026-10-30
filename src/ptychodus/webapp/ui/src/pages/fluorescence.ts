import { api, type FluorescenceRead, type RenderParams, type RenderedImage } from '../api.js';
import { createDownloadBar } from '../components/download_bar.js';
import { createImagePanel } from '../components/image_panel.js';
import { createTree, type TreeNode } from '../components/tree.js';
import { MISSING, fmt, fmtBytes, ingestRowState, infoText } from '../format.js';
import { buildPageLayout, infoLabel } from '../layout.js';

export async function mountFluorescence(root: HTMLElement): Promise<void> {
  const { page, left, right, setActiveTab } = buildPageLayout('fluorescence');
  root.replaceChildren(page);

  const tree = createTree(['Name', 'Product', 'Elements', 'Counts', 'Size']);
  left.replaceChildren(tree.el);
  const downloadHost = document.createElement('div');
  const image = createImagePanel();
  right.replaceChildren(downloadHost, image.el);
  image.setEmpty();

  let redraw: ((params: Partial<RenderParams>) => void) | null = null;
  image.enableControls((params) => redraw?.(params), { isComplex: false });

  let items: FluorescenceRead[] = [];
  let total = 0;
  try {
    const listing = await api.listFluorescence();
    items = listing.items;
    total = listing.total;
  } catch (err) {
    left.replaceChildren(errorBlock(err as Error));
    return;
  }

  if (items.length === 0) {
    left.replaceChildren(emptyBlock('No fluorescence datasets in store.'));
    return;
  }

  const remember = (fn: (params: Partial<RenderParams>) => void): void => {
    redraw = fn;
  };

  const nodes: TreeNode[] = items.map((f) => {
    const label = f.label || f.uuid.slice(0, 8);
    const state = ingestRowState(f);
    // Without the product the map renders at a 1 um placeholder pitch and the caption
    // reports a physical size that is simply wrong.
    const product = productOf(f);

    const show = (
      title: string,
      fetch: (params: Partial<RenderParams>) => Promise<RenderedImage>
    ): void => {
      const draw = (params: Partial<RenderParams>): void => {
        image.setLoading(title);
        fetch(params)
          .then((img) => image.setImage(img, `${label} — ${title}`))
          .catch((err: Error) => image.setError(err));
      };
      setActiveTab('right');
      downloadHost.replaceChildren(
        createDownloadBar(api.fluorescenceFileUrl(f.uuid), `${label}.h5`)
      );
      remember(draw);
      draw(image.params());
    };

    return {
      id: `flu:${f.uuid}`,
      label,
      state,
      cells: [
        productName(f),
        String(f.element_names.length),
        fmt(f.element_counts.reduce((a, b) => a + b, 0)),
        fmtBytes(f.nbytes),
      ],
      // Selecting the dataset shows every element map summed, as the desktop does.
      onSelect: () =>
        show('all elements', (q) => api.fluorescenceSummaryImage(f.uuid, product, q)),
      children: f.element_names.map((name, i) => ({
        id: `flu:${f.uuid}:${name}`,
        label: name,
        state,
        cells: ['', '', fmt(f.element_counts[i]), fmtBytes(f.element_nbytes[i])],
        onSelect: () =>
          show(name, (q) => api.fluorescenceElementImage(f.uuid, name, product, q)),
      })),
    };
  });
  tree.setNodes(nodes);
  left.appendChild(infoLabel(infoText('Datasets', total, items.map((f) => f.nbytes))));
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

/** That product by name, as the desktop's Product column shows it. */
function productName(f: FluorescenceRead): string {
  const edge = f.derived_from.find((e) => e.kind === 'product');
  if (edge === undefined) return MISSING;
  return edge.label || edge.uuid.slice(0, 8);
}
