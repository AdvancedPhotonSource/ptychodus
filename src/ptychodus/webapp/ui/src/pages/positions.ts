import { api, type ProductRead } from '../api.js';
import { createImagePanel } from '../components/image_panel.js';
import { createMetricsPanel } from '../components/metrics_panel.js';
import { createTable } from '../components/table.js';
import { buildPageLayout, infoLabel } from '../layout.js';
import { findLinked, showLink } from '../routes.js';
import { fmtBytes, fmtFixed, fmtInt, ingestRowState, infoText } from '../format.js';

export async function mountPositions(root: HTMLElement, uuid: string | null): Promise<void> {
  const { page, left, right, setActiveTab } = buildPageLayout('positions');
  root.replaceChildren(page);

  const image = createImagePanel();
  const metrics = createMetricsPanel('positions');
  const split = document.createElement('div');
  split.className = 'with-metrics';
  split.append(image.el, metrics.el);
  right.replaceChildren(split);
  image.setEmpty('Tick one or more products to plot them.');

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

  const found = await findLinked(items, uuid, api.getProduct);
  items = found.items;

  const plot = (checked: ProductRead[]): void => {
    if (checked.length === 0) {
      image.setEmpty('Tick one or more products to plot them.');
      return;
    }

    setActiveTab('right');
    image.setLoading(checked.length === 1 ? 'probe positions' : `${checked.length} products`);
    api
      .productPositionsImage(checked.map((p) => p.uuid))
      .then((img) => image.setPlot(img, checked.map(labelFor).join(', ')))
      .catch((err: Error) => image.setError(err));
  };

  const table = createTable<ProductRead>(
    [
      { header: 'Name', render: labelFor },
      {
        header: 'Positions',
        render: (p) => fmtInt(p.num_probe_positions),
        sortKey: (p) => p.num_probe_positions,
        numeric: true,
      },
      {
        header: 'Length [m]',
        render: (p) => fmtFixed(p.probe_path_length_m, 6),
        sortKey: (p) => p.probe_path_length_m,
        numeric: true,
      },
      {
        header: 'Size',
        render: (p) => fmtBytes(p.probe_positions_nbytes),
        sortKey: (p) => p.probe_positions_nbytes,
        numeric: true,
      },
    ],
    // As in the GUI, a click only selects, links to the row and describes it; the Plot
    // checkboxes choose what is drawn, in table order.
    (row) => {
      showLink('positions', row.uuid);
      metrics.show(row, items);
    },
    { checkbox: { header: 'Plot', onChange: plot, position: 1 }, rowState: ingestRowState }
  );

  left.replaceChildren(
    table.el,
    infoLabel(infoText('Positions', total, items.map((p) => p.probe_positions_nbytes)))
  );
  table.setRows(items);

  // A deep link selects its product and plots it.
  if (found.linked !== null) {
    table.setSelected(items.indexOf(found.linked));
    table.setChecked([found.linked]);
    metrics.show(found.linked, items);
  }
  if (found.error !== null) image.setError(found.error);
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
