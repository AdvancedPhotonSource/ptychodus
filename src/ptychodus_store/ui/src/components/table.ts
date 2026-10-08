/** Column-bearing, sortable table with an optional checkbox column. */

export interface Column<T> {
  header: string;
  render: (row: T) => string;
  /** Sort key; omit for a column that should not sort. Defaults to render(). */
  sortKey?: (row: T) => number | string | null;
  numeric?: boolean;
}

export interface TableOptions<T> {
  /** Adds a leading checkbox column, as the desktop Positions panel has. */
  checkbox?: { header: string; onChange: (checked: T[]) => void };
}

export interface DataTable<T> {
  el: HTMLElement;
  setRows: (rows: T[]) => void;
  setSelected: (index: number | null) => void;
  getChecked: () => T[];
}

type SortState = { column: number; descending: boolean } | null;

function compare<T>(column: Column<T>, a: T, b: T): number {
  const key = column.sortKey ?? ((row: T) => column.render(row));
  const left = key(a);
  const right = key(b);

  // A missing value sorts last either way, so an em dash never leads the table.
  if (left === null) return right === null ? 0 : 1;
  if (right === null) return -1;

  if (typeof left === 'number' && typeof right === 'number') return left - right;
  return String(left).localeCompare(String(right), undefined, { numeric: true });
}

export function createTable<T>(
  columns: Column<T>[],
  onRowClick: (row: T, index: number) => void,
  options: TableOptions<T> = {}
): DataTable<T> {
  const el = document.createElement('table');
  el.className = 'data-table';
  const thead = document.createElement('thead');
  const trHead = document.createElement('tr');
  const tbody = document.createElement('tbody');
  el.append(thead, tbody);

  let rows: T[] = [];
  let sort: SortState = null;
  let selectedIndex: number | null = null;
  const checked = new Set<T>();

  if (options.checkbox) {
    const th = document.createElement('th');
    th.className = 'check-col';
    th.textContent = options.checkbox.header;
    trHead.appendChild(th);
  }

  columns.forEach((col, i) => {
    const th = document.createElement('th');
    th.textContent = col.header;
    if (col.numeric) th.classList.add('numeric');
    th.classList.add('sortable');
    th.addEventListener('click', () => {
      sort =
        sort && sort.column === i ? { column: i, descending: !sort.descending } : { column: i, descending: false };
      draw();
    });
    trHead.appendChild(th);
  });

  thead.appendChild(trHead);

  function sorted(): T[] {
    if (sort === null) return rows;
    const col = columns[sort.column]!;
    const sign = sort.descending ? -1 : 1;
    return [...rows].sort((a, b) => sign * compare(col, a, b));
  }

  function draw(): void {
    const ordered = sorted();
    tbody.replaceChildren();

    trHead.querySelectorAll('th.sortable').forEach((th, i) => {
      th.classList.toggle('sorted-asc', sort?.column === i && !sort.descending);
      th.classList.toggle('sorted-desc', sort?.column === i && sort.descending);
    });

    ordered.forEach((row, i) => {
      const tr = document.createElement('tr');

      if (options.checkbox) {
        const td = document.createElement('td');
        td.className = 'check-col';
        const box = document.createElement('input');
        box.type = 'checkbox';
        box.checked = checked.has(row);
        box.addEventListener('click', (ev) => ev.stopPropagation());
        box.addEventListener('change', () => {
          if (box.checked) checked.add(row);
          else checked.delete(row);
          options.checkbox!.onChange(rows.filter((r) => checked.has(r)));
        });
        td.appendChild(box);
        tr.appendChild(td);
      }

      for (const col of columns) {
        const td = document.createElement('td');
        if (col.numeric) td.classList.add('numeric');
        td.textContent = col.render(row);
        tr.appendChild(td);
      }

      if (i === selectedIndex) tr.classList.add('selected');
      tr.addEventListener('click', () => {
        setSelected(i);
        onRowClick(row, i);
      });
      tbody.appendChild(tr);
    });
  }

  function setSelected(idx: number | null): void {
    selectedIndex = idx;
    tbody.querySelectorAll('tr.selected').forEach((n) => n.classList.remove('selected'));
    if (idx !== null) (tbody.children[idx] as HTMLElement | undefined)?.classList.add('selected');
  }

  function setRows(next: T[]): void {
    rows = next;
    for (const row of [...checked]) if (!rows.includes(row)) checked.delete(row);
    if (selectedIndex !== null && selectedIndex >= rows.length) selectedIndex = null;
    draw();
  }

  return { el, setRows, setSelected, getChecked: () => rows.filter((r) => checked.has(r)) };
}
