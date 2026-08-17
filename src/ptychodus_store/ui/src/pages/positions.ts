import { api, type ProductRead } from '../api.js';
import { createImagePanel } from '../components/image_panel.js';
import { createProductPicker } from '../components/product_picker.js';
import { buildPageLayout } from '../layout.js';

export async function mountPositions(root: HTMLElement): Promise<void> {
  const { page, left, right, setActiveTab } = buildPageLayout('positions');
  root.replaceChildren(page);

  const image = createImagePanel();
  right.replaceChildren(image.el);
  image.setEmpty('Select a product to view its scan positions.');

  const picker = createProductPicker((product) => selectProduct(product));
  left.replaceChildren(picker.el);
  await picker.load();

  function selectProduct(product: ProductRead): void {
    setActiveTab('right');
    image.setLoading('probe positions');
    api
      .productPositionsImage(product.uuid)
      .then((img) => image.setImage(img, `${labelFor(product)} — positions`))
      .catch((err: Error) => image.setError(err));
  }
}

function labelFor(p: ProductRead): string {
  return p.name ?? p.uuid.slice(0, 8);
}
