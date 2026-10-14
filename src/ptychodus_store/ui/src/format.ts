/** Shared cell formatting, so every table renders a missing value the same way. */

export const MISSING = '—';

export function fmt(x: number | null | undefined): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return MISSING;
  return Number.isInteger(x) ? String(x) : Number(x).toPrecision(4);
}

export function scale(x: number | null | undefined, factor: number): number | null {
  return x === null || x === undefined ? null : x * factor;
}

export function shape(h: number | null, w: number | null): string {
  return h === null || w === null ? MISSING : `${h} × ${w}`;
}

export function fmtBytes(n: number | null | undefined): string {
  if (n === null || n === undefined) return MISSING;

  let value = n;
  for (const unit of ['B', 'kB', 'MB', 'GB']) {
    if (value < 1024 || unit === 'GB') return `${unit === 'B' ? value : value.toFixed(1)} ${unit}`;
    value /= 1024;
  }
  return `${value.toFixed(1)} GB`;
}

export function fmtPercent(x: number | null | undefined): string {
  return x === null || x === undefined ? MISSING : `${(100 * x).toFixed(1)}%`;
}
