/**
 * Hash routes, deep links among them, mirroring the REST paths: `#<kind>/<uuid>[/<view>]`.
 *
 * `#product/<uuid>` opens the product page with that product selected, and
 * `#product/<uuid>/probe`, `/object` or `/positions` that view of it; `#diffraction/<uuid>`
 * and `#fluorescence/<uuid>` open their pages. A bare `#<page>` selects nothing.
 */

/** The views of each kind of resource, each a page of its own beside the kind's page. */
const VIEWS_BY_KIND: Record<string, readonly string[]> = {
  diffraction: [],
  product: ['positions', 'probe', 'object'],
  fluorescence: [],
};

export interface Route {
  page: string;
  uuid: string | null;
}

/** The page and resource a hash names, or null for one that names neither shape. */
export function parseRoute(hash: string): Route | null {
  const [head = '', uuid, view, ...rest] = hash.replace(/^#/, '').split('/');
  if (rest.length > 0) return null;
  if (uuid === undefined) return { page: head, uuid: null };

  const views = VIEWS_BY_KIND[head];
  if (views === undefined || uuid === '') return null;
  if (view === undefined) return { page: head, uuid };
  return views.includes(view) ? { page: view, uuid } : null;
}

/** The deep link to `uuid` on `page`: `#product/<uuid>/probe` for the probe page. */
export function linkTo(page: string, uuid: string): string {
  for (const [kind, views] of Object.entries(VIEWS_BY_KIND)) {
    if (page === kind) return `#${kind}/${uuid}`;
    if (views.includes(page)) return `#${kind}/${uuid}/${page}`;
  }
  return `#${page}`;
}

/** Shows the deep link to `uuid` on `page` in the address bar, without navigating. */
export function showLink(page: string, uuid: string): void {
  history.replaceState(null, '', linkTo(page, uuid));
}

export interface Linked<T> {
  /** The listing, with the linked resource first when the listing did not hold it. */
  items: T[];
  linked: T | null;
  /** Why a linked uuid matched no resource. */
  error: Error | null;
}

/** The resource `uuid` names, from the listing or else fetched by itself. */
export async function findLinked<T extends { uuid: string }>(
  items: T[],
  uuid: string | null,
  fetch: (uuid: string) => Promise<T>
): Promise<Linked<T>> {
  if (uuid === null) return { items, linked: null, error: null };

  const listed = items.find((item) => item.uuid === uuid);
  if (listed !== undefined) return { items, linked: listed, error: null };

  try {
    const fetched = await fetch(uuid);
    return { items: [fetched, ...items], linked: fetched, error: null };
  } catch (err) {
    return { items, linked: null, error: err as Error };
  }
}
