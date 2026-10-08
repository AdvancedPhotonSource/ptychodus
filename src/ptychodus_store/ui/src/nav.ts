export interface NavEntry {
  route: string;
  label: string;
  icon: string;
  /** Indented beneath this entry, as ViewCore's add_subview_group does in the GUI. */
  children?: NavEntry[];
}

export const NAV: NavEntry[] = [
  { route: 'diffraction', label: 'Diffraction', icon: 'table-cells.svg' },
  {
    route: 'product',
    label: 'Products',
    icon: 'list.svg',
    children: [
      { route: 'positions', label: 'Positions', icon: 'route.svg' },
      { route: 'probe', label: 'Probe', icon: 'circle-radiation.svg' },
      { route: 'object', label: 'Object', icon: 'layer-group.svg' },
    ],
  },
  { route: 'fluorescence', label: 'Fluorescence', icon: 'atom.svg' },
];

/** Every entry, parents and children alike, in the order the rail presents them. */
export function flattenNav(entries: NavEntry[] = NAV): NavEntry[] {
  return entries.flatMap((entry) => [entry, ...flattenNav(entry.children ?? [])]);
}

interface AboutLink {
  href: string;
  label: string;
}

const ABOUT_LINKS: AboutLink[] = [
  { href: 'https://github.com/AdvancedPhotonSource/ptychodus', label: 'GitHub' },
  { href: 'https://ptychodus.readthedocs.io/', label: 'Documentation' },
];

export function renderNav(root: HTMLElement, current: string, onSelect: (route: string) => void): void {
  root.replaceChildren();

  const button = (entry: NavEntry): HTMLButtonElement => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.title = entry.label;
    btn.setAttribute('aria-label', entry.label);
    if (entry.route === current) btn.classList.add('active');
    const img = document.createElement('img');
    img.src = `/ui/icons/${entry.icon}`;
    img.alt = '';
    btn.appendChild(img);
    btn.addEventListener('click', () => onSelect(entry.route));
    return btn;
  };

  for (const entry of NAV) {
    root.appendChild(button(entry));

    if (entry.children === undefined) continue;

    // A rail down the left and an indent mark the child group, matching the
    // _SubviewGroupContainer the desktop nav draws.
    const group = document.createElement('div');
    group.className = 'nav-group';
    for (const child of entry.children) group.appendChild(button(child));
    root.appendChild(group);
  }

  const spacer = document.createElement('div');
  spacer.className = 'nav-spacer';
  root.appendChild(spacer);

  root.appendChild(buildAboutMenu());
}

function buildAboutMenu(): HTMLElement {
  const wrap = document.createElement('div');
  wrap.className = 'nav-about';

  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'nav-logo';
  btn.title = 'About ptychodus';
  btn.setAttribute('aria-label', 'About ptychodus');
  btn.setAttribute('aria-haspopup', 'menu');
  btn.setAttribute('aria-expanded', 'false');
  const img = document.createElement('img');
  img.src = '/ui/icons/ptychodus.svg';
  img.alt = '';
  btn.appendChild(img);

  const menu = document.createElement('div');
  menu.className = 'nav-menu';
  menu.setAttribute('role', 'menu');
  menu.hidden = true;
  for (const link of ABOUT_LINKS) {
    const a = document.createElement('a');
    a.href = link.href;
    a.textContent = link.label;
    a.target = '_blank';
    a.rel = 'noopener noreferrer';
    a.setAttribute('role', 'menuitem');
    menu.appendChild(a);
  }

  const close = () => {
    menu.hidden = true;
    btn.setAttribute('aria-expanded', 'false');
    document.removeEventListener('pointerdown', onOutside, true);
    document.removeEventListener('keydown', onKey, true);
  };
  const onOutside = (e: Event) => {
    if (!wrap.contains(e.target as Node)) close();
  };
  const onKey = (e: KeyboardEvent) => {
    if (e.key === 'Escape') { close(); btn.focus(); }
  };
  btn.addEventListener('click', () => {
    const open = menu.hidden;
    menu.hidden = !open;
    btn.setAttribute('aria-expanded', String(open));
    if (open) {
      document.addEventListener('pointerdown', onOutside, true);
      document.addEventListener('keydown', onKey, true);
    } else {
      document.removeEventListener('pointerdown', onOutside, true);
      document.removeEventListener('keydown', onKey, true);
    }
  });

  wrap.append(btn, menu);
  return wrap;
}
