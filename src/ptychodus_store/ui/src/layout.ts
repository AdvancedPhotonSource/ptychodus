const NARROW_QUERY = '(max-width: 56.25rem)';
const REM_PX = () => parseFloat(getComputedStyle(document.documentElement).fontSize) || 16;

export interface PageLayout {
  page: HTMLElement;
  left: HTMLElement;
  right: HTMLElement;
  setActiveTab: (which: 'left' | 'right') => void;
}

const activeTab = new Map<string, 'left' | 'right'>();

export function buildPageLayout(pageKey: string): PageLayout {
  const page = document.createElement('div');
  page.className = 'page';

  const tabs = document.createElement('div');
  tabs.className = 'tabs';
  const tabLeft = document.createElement('button');
  tabLeft.type = 'button';
  tabLeft.textContent = 'Browse';
  const tabRight = document.createElement('button');
  tabRight.type = 'button';
  tabRight.textContent = 'View';
  tabs.append(tabLeft, tabRight);

  const left = document.createElement('div');
  left.className = 'panel-left';
  const divider = document.createElement('div');
  divider.className = 'divider';
  divider.setAttribute('role', 'separator');
  divider.setAttribute('aria-orientation', 'vertical');
  const right = document.createElement('div');
  right.className = 'panel-right';

  page.append(tabs, left, divider, right);

  const mq = window.matchMedia(NARROW_QUERY);
  const apply = () => {
    const isNarrow = mq.matches;
    if (isNarrow) {
      const active = activeTab.get(pageKey) ?? 'left';
      left.classList.toggle('hidden', active !== 'left');
      right.classList.toggle('hidden', active !== 'right');
      tabLeft.classList.toggle('active', active === 'left');
      tabRight.classList.toggle('active', active === 'right');
    } else {
      left.classList.remove('hidden');
      right.classList.remove('hidden');
    }
  };
  mq.addEventListener('change', apply);
  apply();

  const setActiveTab = (which: 'left' | 'right') => {
    activeTab.set(pageKey, which);
    apply();
  };
  tabLeft.addEventListener('click', () => setActiveTab('left'));
  tabRight.addEventListener('click', () => setActiveTab('right'));

  wireSplitterDrag(divider);

  return { page, left, right, setActiveTab };
}

function wireSplitterDrag(divider: HTMLElement): void {
  divider.addEventListener('pointerdown', (ev) => {
    ev.preventDefault();
    divider.classList.add('dragging');
    divider.setPointerCapture(ev.pointerId);
    const rem = REM_PX();
    const contentRect = divider.parentElement!.getBoundingClientRect();
    const navWidth = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--nav-w')) * rem;

    const move = (e: PointerEvent) => {
      const xInContent = e.clientX - contentRect.left - navWidth;
      const clampedPx = Math.max(rem * 12, Math.min(rem * 60, xInContent));
      document.documentElement.style.setProperty('--left-w', `${(clampedPx / rem).toFixed(3)}rem`);
    };
    const up = (e: PointerEvent) => {
      divider.classList.remove('dragging');
      divider.releasePointerCapture(e.pointerId);
      divider.removeEventListener('pointermove', move);
      divider.removeEventListener('pointerup', up);
      divider.removeEventListener('pointercancel', up);
    };
    divider.addEventListener('pointermove', move);
    divider.addEventListener('pointerup', up);
    divider.addEventListener('pointercancel', up);
  });
}
