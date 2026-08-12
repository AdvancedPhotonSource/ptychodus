import { api, type ProductRead } from '../api.js';
import { createTable } from './table.js';

export interface ProductPicker {
  el: HTMLElement;
  load: () => Promise<ProductRead[]>;
}

export function createProductPicker(onSelect: (p: ProductRead) => void): ProductPicker {
  const host = document.createElement('div');

  const load = async (): Promise<ProductRead[]> => {
    let items: ProductRead[] = [];
    try {
      const listing = await api.listProduct();
      items = listing.items;
    } catch (err) {
      host.replaceChildren(errorBlock(err as Error));
      return [];
    }
    if (items.length === 0) {
      host.replaceChildren(emptyBlock('No products in store.'));
      return [];
    }
    const table = createTable<ProductRead>(
      [{ header: 'Name', render: (p) => p.name ?? p.uuid.slice(0, 8) }],
      (row) => onSelect(row)
    );
    host.replaceChildren(table.el);
    table.setRows(items);
    return items;
  };

  return { el: host, load };
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
