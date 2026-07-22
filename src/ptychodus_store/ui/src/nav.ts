export interface NavEntry {
  route: string;
  label: string;
  icon: string;
}

export const NAV: NavEntry[] = [
  { route: 'diffraction', label: 'Patterns', icon: 'table-cells.svg' },
  { route: 'product', label: 'Products', icon: 'list.svg' },
  { route: 'fluorescence', label: 'Fluorescence', icon: 'atom.svg' },
];

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
  for (const entry of NAV) {
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
    root.appendChild(btn);
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
