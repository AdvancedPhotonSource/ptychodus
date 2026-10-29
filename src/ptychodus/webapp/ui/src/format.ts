/** Shared cell formatting, so every table renders a missing value the same way. */

import type { IngestState } from './api.js';

export const MISSING = '—';

function isMissing(x: number | null | undefined): x is null | undefined {
  return x === null || x === undefined || !Number.isFinite(x);
}

/**
 * Python's `format(x, '.<precision>g')`, which the desktop applies to every derived
 * quantity: significant digits without trailing zeros, switching to an exponent with
 * at least two digits below 1e-4 or at 10**precision and above.
 */
export function formatG(x: number, precision = 4): string {
  if (x === 0) return '0';

  const [mantissa, exponentText] = x.toExponential(precision - 1).split('e');
  const exponent = Number(exponentText);

  if (exponent < -4 || exponent >= precision) {
    const digits = mantissa!.includes('.') ? mantissa!.replace(/\.?0+$/, '') : mantissa!;
    const sign = exponent < 0 ? '-' : '+';
    return `${digits}e${sign}${String(Math.abs(exponent)).padStart(2, '0')}`;
  }

  const fixed = x.toFixed(Math.max(0, precision - 1 - exponent));
  return fixed.includes('.') ? fixed.replace(/\.?0+$/, '') : fixed;
}

/**
 * A derived quantity, as the desktop's `:.4g`. A precision of 6 matches a cell the
 * desktop hands Qt as a bare float, which Qt renders as `%g`.
 */
export function fmt(x: number | null | undefined, precision = 4): string {
  return isMissing(x) ? MISSING : formatG(x, precision);
}

/** A count the desktop shows exactly: pixels, frames, bad pixels. */
export function fmtInt(x: number | null | undefined): string {
  return isMissing(x) ? MISSING : String(x);
}

/** A fixed number of decimals, as the desktop's `:.<digits>f`. */
export function fmtFixed(x: number | null | undefined, digits: number): string {
  return isMissing(x) ? MISSING : x.toFixed(digits);
}

export function scale(x: number | null | undefined, factor: number): number | null {
  return x === null || x === undefined ? null : x * factor;
}

export function shape(h: number | null, w: number | null): string {
  return h === null || w === null ? MISSING : `${h} × ${w}`;
}

/** Decimal byte units, as ptychodus.api.constants.ByteUnit renders them. */
const BYTE_UNITS: [number, string][] = [
  [0, 'B'],
  [3, 'kB'],
  [6, 'MB'],
  [9, 'GB'],
  [12, 'TB'],
  [15, 'PB'],
];

export function fmtBytes(n: number | null | undefined): string {
  if (isMissing(n)) return MISSING;

  let [power, label] = BYTE_UNITS[0]!;
  for (const [candidatePower, candidateLabel] of BYTE_UNITS) {
    if (n < 10 ** candidatePower) break;
    [power, label] = [candidatePower, candidateLabel];
  }

  return power === 0 ? `${n} ${label}` : `${(n / 10 ** power).toFixed(2)} ${label}`;
}

/** A fraction as a whole percentage, truncated as the desktop's `int(100 * x)` is. */
export function fmtPercent(x: number | null | undefined): string {
  return isMissing(x) ? MISSING : `${Math.trunc(100 * x)}%`;
}

/** Bytes per element of a numpy dtype name such as `uint16` or `complex64`. */
export function dtypeItemsize(dtype: string | null): number | null {
  if (dtype === null) return null;
  if (dtype === 'bool' || dtype === 'int8' || dtype === 'uint8') return 1;

  const bits = /(\d+)$/.exec(dtype);
  return bits === null ? null : Number(bits[1]) / 8;
}

/** Sum of the byte counts that are known, or null when none are. */
export function sumBytes(parts: (number | null | undefined)[]): number | null {
  const known = parts.filter((n): n is number => n !== null && n !== undefined);
  return known.length === 0 ? null : known.reduce((a, b) => a + b, 0);
}

/** A repository's totals, as the desktop's info label reads, e.g. "Products: 3 [1.20 GB]". */
export function infoText(noun: string, count: number, nbytes: (number | null)[]): string {
  return `${noun}: ${count} [${fmtBytes(sumBytes(nbytes) ?? 0)}]`;
}

/** How a row reads when its files are not, or not yet, usable. */
export interface RowState {
  className: 'row-pending' | 'row-failed';
  title: string;
}

/**
 * The desktop greys a failed item out and italicises one still loading; the store's
 * counterparts are a manifest that failed to ingest and one not yet ingested.
 */
export function ingestRowState(row: {
  ingest_state: IngestState;
  error_message: string | null;
}): RowState | null {
  switch (row.ingest_state) {
    case 'VALID':
      return null;
    case 'DISCOVERED':
      return { className: 'row-pending', title: 'Not yet ingested' };
    default:
      return { className: 'row-failed', title: row.error_message ?? row.ingest_state };
  }
}
