import { api, type DiffractionRead } from '../api.js';
import { createDownloadBar } from '../components/download_bar.js';
import { createImagePanel, type ImagePanel } from '../components/image_panel.js';
import { createTree, type TreeNode } from '../components/tree.js';
import { buildPageLayout } from '../layout.js';

const PATTERN_PAGE = 200;

export async function mountDiffraction(root: HTMLElement): Promise<void> {
  const { page, left, right, setActiveTab } = buildPageLayout('diffraction');
  root.replaceChildren(page);

  const tree = createTree();
  left.replaceChildren(tree.el);
  const downloadHost = document.createElement('div');
  const image = createImagePanel();
  right.replaceChildren(downloadHost, image.el);
  image.setEmpty();

  let items: DiffractionRead[] = [];
  try {
    const listing = await api.listDiffraction();
    items = listing.items;
  } catch (err) {
    left.replaceChildren(errorBlock(err as Error));
    return;
  }

  if (items.length === 0) {
    left.replaceChildren(emptyBlock('No diffraction datasets in store.'));
    return;
  }

  const nodes: TreeNode[] = items.map((d) => ({
    id: `diff:${d.uuid}`,
    label: d.label || d.uuid.slice(0, 8),
    loadChildren: () => loadPatternNodes(d, image, downloadHost, setActiveTab),
  }));
  tree.setNodes(nodes);
}

async function loadPatternNodes(
  d: DiffractionRead,
  image: ImagePanel,
  downloadHost: HTMLElement,
  setActiveTab: (which: 'left' | 'right') => void
): Promise<TreeNode[]> {
  const detail = await api.getDiffraction(d.uuid);
  downloadHost.replaceChildren(
    createDownloadBar(api.diffractionFileUrl(d.uuid), `${detail.label || d.uuid}.h5`)
  );
  const total = detail.num_patterns_total ?? 0;
  const children: TreeNode[] = [
    {
      id: `diff:${d.uuid}:agg`,
      label: 'Aggregate (mean)',
      onSelect: () => {
        setActiveTab('right');
        image.setLoading('aggregate pattern');
        api
          .diffractionAggregateImage(d.uuid)
          .then((img) => image.setImage(img, `${detail.label || d.uuid} — aggregate`))
          .catch((err: Error) => image.setError(err));
      },
    },
  ];
  const shown = Math.min(total, PATTERN_PAGE);
  for (let i = 0; i < shown; i++) {
    children.push(patternLeaf(d, detail, i, image, setActiveTab));
  }
  if (total > PATTERN_PAGE) {
    children.push({
      id: `diff:${d.uuid}:more`,
      label: `… ${total - PATTERN_PAGE} more (open a specific index directly)`,
    });
  }
  return children;
}

function patternLeaf(
  d: DiffractionRead,
  detail: DiffractionRead,
  index: number,
  image: ImagePanel,
  setActiveTab: (which: 'left' | 'right') => void
): TreeNode {
  return {
    id: `diff:${d.uuid}:${index}`,
    label: `Pattern ${index}`,
    onSelect: () => {
      setActiveTab('right');
      image.setLoading(`pattern ${index}`);
      api
        .diffractionPatternImage(d.uuid, index)
        .then((img) => image.setImage(img, `${detail.label || d.uuid} — pattern ${index}`))
        .catch((err: Error) => image.setError(err));
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
