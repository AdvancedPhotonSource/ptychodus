import { api, type ProductRead } from '../api.js';
import { createImagePanel } from '../components/image_panel.js';
import { createTable } from '../components/table.js';
import { buildPageLayout, infoLabel } from '../layout.js';
import { fmtBytes, fmtFixed, fmtInt, ingestRowState, infoText } from '../format.js';

export async function mountPositions(root: HTMLElement): Promise<void> {
  const { page, left, right, setActiveTab } = buildPageLayout('positions');
  root.replaceChildren(page);

  const image = createImagePanel();
  right.replaceChildren(image.el);
  image.setEmpty('Tick one or more scans to plot them.');

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

  const plot = (checked: ProductRead[]): void => {
    if (checked.length === 0) {
      image.setEmpty('Tick one or more scans to plot them.');
      return;
    }

    setActiveTab('right');
    image.setLoading(checked.length === 1 ? 'scan positions' : `${checked.length} scans`);
    api
      .productPositionsImage(checked.map((p) => p.uuid))
      .then((img) => image.setPlot(img, checked.map(labelFor).join(', ')))
      .catch((err: Error) => image.setError(err));
  };

  const table = createTable<ProductRead>(
    [
      { header: 'Name', render: labelFor },
      {
        header: 'Points',
        render: (p) => fmtInt(p.num_scan_points),
        sortKey: (p) => p.num_scan_points,
        numeric: true,
      },
      {
        header: 'Length [m]',
        render: (p) => fmtFixed(p.scan_length_m, 6),
        sortKey: (p) => p.scan_length_m,
        numeric: true,
      },
      { header: 'Size', render: (p) => fmtBytes(p.scan_nbytes), sortKey: (p) => p.scan_nbytes, numeric: true },
    ],
    // As on the desktop, a click only selects; the Plot checkboxes choose what is
    // drawn, in table order.
    () => undefined,
    { checkbox: { header: 'Plot', onChange: plot, position: 1 }, rowState: ingestRowState }
  );

  left.replaceChildren(
    table.el,
    infoLabel(infoText('Positions', total, items.map((p) => p.scan_nbytes)))
  );
  table.setRows(items);
}

function labelFor(p: ProductRead): string {
  return p.name ?? p.uuid.slice(0, 8);
}

function errorBlock(err: Error): HTMLElement {
  const el = document.createElement('div');
  el.style.color = '#ff8080';
  el.style.padding = '1rem';
  el.textContent = err.message;
  return el;
}
