/**
 * A product's metrics report beneath its image, and its comparison with a reference
 * product chosen from the others. Both are computed on request by the service, so each
 * shows a status while it runs and only the latest request may fill the panel.
 */

import {
  api,
  type ComparisonRead,
  type MetricsRead,
  type ProductRead,
  type ScalarValue,
  type Scalars,
} from '../api.js';
import { fmt, MISSING } from '../format.js';

export type MetricsView = 'probe' | 'object' | 'positions';

interface ViewReports {
  title: string;
  metrics: (uuid: string) => Promise<MetricsRead>;
  comparison: (uuid: string, referenceUuid: string) => Promise<ComparisonRead>;
}

const REPORTS: Record<MetricsView, ViewReports> = {
  probe: {
    title: 'Probe Metrics',
    metrics: api.productProbeMetrics,
    comparison: api.productProbeComparison,
  },
  object: {
    title: 'Object Metrics',
    metrics: api.productObjectMetrics,
    comparison: api.productObjectComparison,
  },
  positions: {
    title: 'Probe Position Metrics',
    metrics: api.productPositionsMetrics,
    comparison: api.productPositionsComparison,
  },
};

export interface MetricsPanel {
  el: HTMLElement;
  /** Describe `product`, offering every other product in `products` as a reference. */
  show: (product: ProductRead, products: ProductRead[]) => void;
}

export function createMetricsPanel(view: MetricsView): MetricsPanel {
  const reports = REPORTS[view];
  const el = document.createElement('section');
  el.className = 'metrics-panel';
  el.hidden = true;

  const metricsHeading = document.createElement('h3');
  metricsHeading.textContent = reports.title;
  const metricsBody = document.createElement('div');

  const compareHeader = document.createElement('div');
  compareHeader.className = 'metrics-compare';
  const compareLabel = document.createElement('label');
  compareLabel.textContent = 'Compare with ';
  const select = document.createElement('select');
  compareLabel.appendChild(select);
  compareHeader.appendChild(compareLabel);
  const comparisonBody = document.createElement('div');

  el.append(metricsHeading, metricsBody, compareHeader, comparisonBody);

  let current: ProductRead | null = null;
  // Each request takes a ticket; a response whose ticket is no longer the latest is
  // dropped, so a slow report never overwrites the one for a newer selection.
  let metricsTicket = 0;
  let comparisonTicket = 0;

  const compare = (): void => {
    const ticket = ++comparisonTicket;
    const referenceUuid = select.value;

    if (current === null || referenceUuid === '') {
      comparisonBody.replaceChildren();
      return;
    }

    comparisonBody.replaceChildren(statusBlock('Computing…'));
    reports
      .comparison(current.uuid, referenceUuid)
      .then((read) => {
        if (ticket === comparisonTicket) comparisonBody.replaceChildren(scalarsList(read.scalars));
      })
      .catch((err: Error) => {
        if (ticket === comparisonTicket) comparisonBody.replaceChildren(errorBlock(err));
      });
  };
  select.addEventListener('change', compare);

  const show = (product: ProductRead, products: ProductRead[]): void => {
    // Choosing another mode or layer of the same product leaves its reports as they are.
    if (current !== null && current.uuid === product.uuid) return;

    current = product;
    el.hidden = false;

    const none = new Option(MISSING, '');
    const others = products
      .filter((p) => p.uuid !== product.uuid && p.ingest_state === 'VALID')
      .map((p) => new Option(p.name ?? p.uuid.slice(0, 8), p.uuid));
    const previous = select.value;
    select.replaceChildren(none, ...others);
    select.value = others.some((o) => o.value === previous) ? previous : '';

    const ticket = ++metricsTicket;
    metricsBody.replaceChildren(statusBlock('Computing…'));
    reports
      .metrics(product.uuid)
      .then((read) => {
        if (ticket === metricsTicket) metricsBody.replaceChildren(scalarsList(read.scalars));
      })
      .catch((err: Error) => {
        if (ticket === metricsTicket) metricsBody.replaceChildren(errorBlock(err));
      });

    compare();
  };

  return { el, show };
}

// Words a label spells in capitals, and abbreviations it spells out.
const WORDS: Record<string, string> = {
  num: 'number of',
  fwhm: 'FWHM',
  rms: 'RMS',
  snr: 'SNR',
  i0: 'I0',
  opr: 'OPR',
  frc: 'FRC',
  psd: 'PSD',
  ssim: 'SSIM',
  x: 'x',
  y: 'y',
};

interface Unit {
  suffix: string;
  render: (x: number) => string;
}

/** A length with the SI prefix that keeps its mantissa readable. */
function fmtLength(x: number, power = 1): string {
  const prefixes: [number, string][] = [
    [1, 'm'],
    [1e-3, 'mm'],
    [1e-6, 'µm'],
    [1e-9, 'nm'],
  ];
  const magnitude = Math.abs(x) ** (1 / power);
  const [factor, symbol] =
    prefixes.find(([f]) => magnitude >= f) ?? prefixes[prefixes.length - 1]!;
  const exponent = power === 1 ? '' : power === 2 ? '²' : `^${power}`;
  return `${fmt(x / factor ** power)} ${symbol}${exponent}`;
}

// Longest suffix first, so `_rad_per_m` is not read as `_m`.
const UNITS: Unit[] = [
  { suffix: '_rad_per_m', render: (x) => `${fmt(x)} rad/m` },
  { suffix: '_per_m2', render: (x) => `${fmt(x)} /m²` },
  { suffix: '_m2', render: (x) => (x === 0 ? '0 m²' : fmtLength(x, 2)) },
  { suffix: '_m', render: (x) => (x === 0 ? '0 m' : fmtLength(x)) },
  { suffix: '_rad', render: (x) => `${fmt(x)} rad` },
  { suffix: '_deg', render: (x) => `${fmt(x)}°` },
  { suffix: '_px', render: (x) => `${fmt(x)} px` },
];

function unitOf(key: string): Unit | undefined {
  return UNITS.find((u) => key.endsWith(u.suffix));
}

function labelFor(key: string): string {
  const unit = unitOf(key);
  const stem = unit === undefined ? key : key.slice(0, -unit.suffix.length);
  const text = stem
    .split('_')
    .map((word) => WORDS[word] ?? word)
    .join(' ');
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function formatValue(key: string, value: ScalarValue): string {
  if (value === null) return MISSING;
  if (typeof value === 'boolean') return value ? 'Yes' : 'No';
  if (typeof value === 'string') return value;
  if (typeof value === 'number') {
    if (Number.isInteger(value) && unitOf(key) === undefined) return String(value);
    return unitOf(key)?.render(value) ?? fmt(value);
  }
  if (Array.isArray(value)) return value.map((v) => formatValue(key, v)).join(', ');
  return MISSING;
}

function isGroup(value: ScalarValue): value is Scalars {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

/** Every scalar as a labelled row, with nested groups headed and indented beneath. */
function appendScalars(dl: HTMLElement, scalars: Scalars, depth: number): void {
  for (const [key, value] of Object.entries(scalars)) {
    if (isGroup(value)) {
      dl.appendChild(groupHeading(labelFor(key), depth));
      appendScalars(dl, value, depth + 1);
    } else if (Array.isArray(value) && value.some(isGroup)) {
      value.forEach((item, i) => {
        dl.appendChild(groupHeading(`${labelFor(key)} ${i + 1}`, depth));
        if (isGroup(item)) appendScalars(dl, item, depth + 1);
      });
    } else {
      const dt = document.createElement('dt');
      dt.textContent = labelFor(key);
      dt.style.paddingLeft = `${depth}rem`;
      const dd = document.createElement('dd');
      dd.textContent = formatValue(key, value);
      dl.append(dt, dd);
    }
  }
}

function groupHeading(text: string, depth: number): HTMLElement {
  const dt = document.createElement('dt');
  dt.className = 'metrics-group';
  dt.style.paddingLeft = `${depth}rem`;
  dt.textContent = text;
  return dt;
}

function scalarsList(scalars: Scalars): HTMLElement {
  const dl = document.createElement('dl');
  dl.className = 'detail-list';
  appendScalars(dl, scalars, 0);
  return dl;
}

function statusBlock(text: string): HTMLElement {
  const el = document.createElement('div');
  el.className = 'metrics-status';
  el.textContent = text;
  return el;
}

function errorBlock(err: Error): HTMLElement {
  const el = document.createElement('div');
  el.className = 'metrics-error';
  // The service states why a report could not be made in its 400 detail.
  const detail = /"detail":\s*"([^"]*)"/.exec(err.message);
  el.textContent = detail?.[1] ?? err.message;
  return el;
}
