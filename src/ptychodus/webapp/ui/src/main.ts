import { NAV, renderNav } from './nav.js';
import { mountDiffraction } from './pages/diffraction.js';
import { mountFluorescence } from './pages/fluorescence.js';
import { mountObject } from './pages/object.js';
import { mountPositions } from './pages/positions.js';
import { mountProbe } from './pages/probe.js';
import { mountProduct } from './pages/product.js';
import { parseRoute, type Route } from './routes.js';

/** Mounts a page, selecting the resource a deep link named, if any. */
type Mount = (root: HTMLElement, uuid: string | null) => void | Promise<void>;

const PAGES: Record<string, Mount> = {
  diffraction: mountDiffraction,
  product: mountProduct,
  positions: mountPositions,
  probe: mountProbe,
  object: mountObject,
  fluorescence: mountFluorescence,
};

function currentRoute(): Route {
  const route = parseRoute(window.location.hash);
  if (route && Object.hasOwn(PAGES, route.page)) return route;
  return { page: NAV[0]!.route, uuid: null };
}

function navigate(page: string): void {
  if (window.location.hash !== `#${page}`) {
    window.location.hash = `#${page}`;
    return;
  }
  render({ page, uuid: null });
}

function render(route: Route): void {
  const navEl = document.getElementById('nav');
  const contentEl = document.getElementById('content');
  if (!navEl || !contentEl) return;
  renderNav(navEl, route.page, navigate);
  contentEl.replaceChildren();
  const mount = PAGES[route.page];
  if (mount) {
    const result = mount(contentEl, route.uuid);
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
