import { NAV, renderNav } from './nav.js';
import { mountDiffraction } from './pages/diffraction.js';
import { mountFluorescence } from './pages/fluorescence.js';
import { mountObject } from './pages/object.js';
import { mountPositions } from './pages/positions.js';
import { mountProbe } from './pages/probe.js';
import { mountProduct } from './pages/product.js';

type Mount = (root: HTMLElement) => void | Promise<void>;

const PAGES: Record<string, Mount> = {
  diffraction: mountDiffraction,
  product: mountProduct,
  positions: mountPositions,
  probe: mountProbe,
  object: mountObject,
  fluorescence: mountFluorescence,
};

function currentRoute(): string {
  const hash = window.location.hash.replace(/^#/, '');
  if (hash && Object.hasOwn(PAGES, hash)) return hash;
  return NAV[0]!.route;
}

function navigate(route: string): void {
  if (window.location.hash !== `#${route}`) {
    window.location.hash = `#${route}`;
    return;
  }
  render(route);
}

function render(route: string): void {
  const navEl = document.getElementById('nav');
  const contentEl = document.getElementById('content');
  if (!navEl || !contentEl) return;
  renderNav(navEl, route, navigate);
  contentEl.replaceChildren();
  const mount = PAGES[route];
  if (mount) {
    const result = mount(contentEl);
    if (result instanceof Promise) result.catch((err: Error) => showFatalError(contentEl, err));
  }
}

function showFatalError(root: HTMLElement, err: Error): void {
  const div = document.createElement('div');
  div.style.padding = '1.5rem';
  div.style.color = '#ff8080';
  div.textContent = err.message;
  root.replaceChildren(div);
}

window.addEventListener('hashchange', () => render(currentRoute()));
render(currentRoute());
