export interface TreeNode {
  id: string;
  label: string;
  /** Values for the columns after the name, in order. */
  cells?: string[];
  loadChildren?: () => Promise<TreeNode[]>;
  children?: TreeNode[];
  /**
   * A child count with a factory, for a parent whose children are too numerous to
   * build eagerly. Only the rows near the viewport are materialised.
   */
  windowedChildren?: { count: number; build: (index: number) => TreeNode };
  onSelect?: () => void;
}

export interface TreeRoot {
  el: HTMLElement;
  setSelected: (id: string | null) => void;
  setNodes: (nodes: TreeNode[]) => void;
}

/** Rows rendered around the viewport by a windowed parent. */
const WINDOW_ROWS = 60;
const ROW_PX = 22;

export function createTree(columns: string[] = []): TreeRoot {
  const el = document.createElement('div');
  el.className = 'tree';
  let selectedId: string | null = null;

  const header = document.createElement('div');
  header.className = 'tree-header';
  const list = document.createElement('ul');
  list.className = 'tree-list';

  if (columns.length > 0) {
    for (const [i, name] of columns.entries()) {
      const cell = document.createElement('span');
      cell.textContent = name;
      if (i > 0) cell.classList.add('numeric');
      header.appendChild(cell);
    }
    el.appendChild(header);
  }

  el.appendChild(list);

  function rowContent(node: TreeNode): DocumentFragment {
    const frag = document.createDocumentFragment();
    const name = document.createElement('span');
    name.textContent = node.label;
    frag.appendChild(name);
    for (const value of node.cells ?? []) {
      const cell = document.createElement('span');
      cell.className = 'numeric';
      cell.textContent = value;
      frag.appendChild(cell);
    }
    return frag;
  }

  function render(nodes: TreeNode[]): void {
    list.replaceChildren();
    for (const node of nodes) list.appendChild(nodeToLi(node));
  }

  /** A scroll-driven window over children that are too many to build eagerly. */
  function windowedList(spec: { count: number; build: (index: number) => TreeNode }): HTMLElement {
    const inner = document.createElement('ul');
    inner.className = 'tree-window';
    inner.style.height = `${Math.min(spec.count, WINDOW_ROWS) * ROW_PX}px`;

    const spacerTop = document.createElement('li');
    const spacerBottom = document.createElement('li');
    let first = -1;

    const paint = (): void => {
      const start = Math.max(0, Math.floor(inner.scrollTop / ROW_PX) - 5);
      if (start === first) return;
      first = start;
      const end = Math.min(spec.count, start + WINDOW_ROWS + 10);

      spacerTop.style.height = `${start * ROW_PX}px`;
      spacerBottom.style.height = `${(spec.count - end) * ROW_PX}px`;

      const built: HTMLElement[] = [];
      for (let i = start; i < end; i++) built.push(nodeToLi(spec.build(i)));
      inner.replaceChildren(spacerTop, ...built, spacerBottom);
    };

    inner.addEventListener('scroll', paint, { passive: true });
    paint();
    return inner;
  }

  function nodeToLi(node: TreeNode): HTMLElement {
    const li = document.createElement('li');
    const expandable =
      node.loadChildren !== undefined ||
      node.windowedChildren !== undefined ||
      (node.children !== undefined && node.children.length > 0);

    if (expandable) {
      const details = document.createElement('details');
      const summary = document.createElement('summary');
      summary.dataset.nodeId = node.id;
      summary.appendChild(rowContent(node));
      if (node.id === selectedId) summary.classList.add('selected');
      if (node.onSelect) {
        summary.addEventListener('click', () => {
          setSelected(node.id);
          node.onSelect?.();
        });
      }
      details.appendChild(summary);
      const inner = node.windowedChildren
        ? windowedList(node.windowedChildren)
        : document.createElement('ul');
      details.appendChild(inner);
      if (node.children) {
        for (const child of node.children) inner.appendChild(nodeToLi(child));
      }
      let loaded = !node.loadChildren;
      details.addEventListener(
        'toggle',
        () => {
          if (details.open && !loaded && node.loadChildren) {
            loaded = true;
            inner.replaceChildren(makeLoading());
            node
              .loadChildren()
              .then((kids) => {
                inner.replaceChildren();
                for (const child of kids) inner.appendChild(nodeToLi(child));
              })
              .catch((err: Error) => {
                inner.replaceChildren(makeError(err));
              });
          }
        },
        { passive: true }
      );
      li.appendChild(details);
    } else {
      const leaf = document.createElement('div');
      leaf.className = 'leaf';
      leaf.dataset.nodeId = node.id;
      leaf.appendChild(rowContent(node));
      if (node.id === selectedId) leaf.classList.add('selected');
      leaf.addEventListener('click', () => {
        setSelected(node.id);
        node.onSelect?.();
      });
      li.appendChild(leaf);
    }
    return li;
  }

  function setSelected(id: string | null): void {
    selectedId = id;
    el.querySelectorAll('.selected').forEach((n) => n.classList.remove('selected'));
    if (id !== null) {
      const found = el.querySelector<HTMLElement>(`[data-node-id="${cssEscape(id)}"]`);
      found?.classList.add('selected');
    }
  }

  return { el, setSelected, setNodes: render };
}

function makeLoading(): HTMLElement {
  const li = document.createElement('li');
  const leaf = document.createElement('div');
  leaf.className = 'leaf';
  leaf.textContent = 'Loading…';
  li.appendChild(leaf);
  return li;
}

function makeError(err: Error): HTMLElement {
  const li = document.createElement('li');
  const leaf = document.createElement('div');
  leaf.className = 'leaf';
  leaf.style.color = '#ff8080';
  leaf.textContent = err.message;
  li.appendChild(leaf);
  return li;
}

function cssEscape(s: string): string {
  return (window as unknown as { CSS: { escape: (s: string) => string } }).CSS.escape(s);
}
