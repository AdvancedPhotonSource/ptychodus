import { api, type ProductRead } from '../api.js';
import { createImagePanel } from '../components/image_panel.js';
import { createProductPicker } from '../components/product_picker.js';
import { buildPageLayout } from '../layout.js';

export async function mountProbe(root: HTMLElement): Promise<void> {
  const { page, left, right, setActiveTab } = buildPageLayout('probe');
  root.replaceChildren(page);

  const image = createImagePanel();
  right.replaceChildren(image.el);
  image.setEmpty('Select a product to view its probe modes.');

  const picker = createProductPicker((product) => selectProduct(product));
  left.replaceChildren(picker.el);
  await picker.load();

  function selectProduct(product: ProductRead): void {
    setActiveTab('right');
    image.setLoading('probe modes');
    api
      .productProbeModesImage(product.uuid)
      .then((img) => image.setImage(img, `${labelFor(product)} — probe modes`))
      .catch((err: Error) => image.setError(err));
  }
}

function labelFor(p: ProductRead): string {
  return p.name ?? p.uuid.slice(0, 8);
}
