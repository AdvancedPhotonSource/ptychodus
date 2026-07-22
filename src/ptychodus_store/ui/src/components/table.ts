export interface Column<T> {
  header: string;
  render: (row: T) => string;
}

export interface DataTable<T> {
  el: HTMLElement;
  setRows: (rows: T[]) => void;
  setSelected: (index: number | null) => void;
}

export function createTable<T>(columns: Column<T>[], onRowClick: (row: T, index: number) => void): DataTable<T> {
  const el = document.createElement('table');
  el.className = 'data-table';
  const thead = document.createElement('thead');
  const trHead = document.createElement('tr');
  for (const col of columns) {
    const th = document.createElement('th');
    th.textContent = col.header;
    trHead.appendChild(th);
  }
  thead.appendChild(trHead);
  const tbody = document.createElement('tbody');
  el.append(thead, tbody);

  let selectedIndex: number | null = null;

  function setSelected(idx: number | null): void {
    selectedIndex = idx;
    tbody.querySelectorAll('tr.selected').forEach((n) => n.classList.remove('selected'));
    if (idx !== null) {
      const tr = tbody.children[idx] as HTMLElement | undefined;
      tr?.classList.add('selected');
    }
  }

  function setRows(rows: T[]): void {
    tbody.replaceChildren();
    rows.forEach((row, i) => {
      const tr = document.createElement('tr');
      for (const col of columns) {
        const td = document.createElement('td');
        td.textContent = col.render(row);
        tr.appendChild(td);
      }
      tr.addEventListener('click', () => {
        setSelected(i);
        onRowClick(row, i);
      });
      tbody.appendChild(tr);
    });
    if (selectedIndex !== null && selectedIndex < rows.length) setSelected(selectedIndex);
    else selectedIndex = null;
  }

  return { el, setRows, setSelected };
}
