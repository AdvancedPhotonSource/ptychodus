export interface TreeNode {
  id: string;
  label: string;
  loadChildren?: () => Promise<TreeNode[]>;
  children?: TreeNode[];
  onSelect?: () => void;
}

export interface TreeRoot {
  el: HTMLElement;
  setSelected: (id: string | null) => void;
  setNodes: (nodes: TreeNode[]) => void;
}

export function createTree(): TreeRoot {
  const el = document.createElement('ul');
  el.className = 'tree';
  let selectedId: string | null = null;

  function render(nodes: TreeNode[]): void {
    el.replaceChildren();
    for (const node of nodes) el.appendChild(nodeToLi(node));
  }

  function nodeToLi(node: TreeNode): HTMLElement {
    const li = document.createElement('li');
    if (node.loadChildren || (node.children && node.children.length > 0)) {
      const details = document.createElement('details');
      const summary = document.createElement('summary');
      summary.textContent = node.label;
      details.appendChild(summary);
      const inner = document.createElement('ul');
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
      leaf.textContent = node.label;
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
    el.querySelectorAll('.leaf.selected').forEach((n) => n.classList.remove('selected'));
    if (id !== null) {
      const found = el.querySelector<HTMLElement>(`.leaf[data-node-id="${cssEscape(id)}"]`);
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
