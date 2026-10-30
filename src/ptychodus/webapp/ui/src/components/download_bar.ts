export function createDownloadBar(href: string, filename: string): HTMLElement {
  const bar = document.createElement('div');
  bar.className = 'download-bar';
  const a = document.createElement('a');
  a.className = 'download-btn';
  a.href = href;
  a.setAttribute('download', filename);
  a.textContent = 'Download .h5';
  bar.appendChild(a);
  return bar;
}
