const API = '/api/v1';

export type IngestState = 'valid' | 'invalid' | 'pending';

export interface DerivedFromEdge {
  kind: 'diffraction' | 'product' | 'fluorescence';
  uuid: string;
}

export interface Page<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
}

export interface DiffractionRead {
  uuid: string;
  label: string;
  comments: string;
  ingest_state: IngestState;
  campaign_uuid: string | null;
  derived_from: DerivedFromEdge[];
  probe_energy_eV: number | null;
  probe_photon_count: number | null;
  tomography_angle_deg: number | null;
  pattern_dtype: string | null;
  pattern_height_px: number | null;
  pattern_width_px: number | null;
  num_patterns_total: number | null;
  detector_pixel_width_m: number | null;
  detector_pixel_height_m: number | null;
}

export interface ProductRead {
  uuid: string;
  ingest_state: IngestState;
  derived_from: DerivedFromEdge[];
  name: string | null;
  comments: string | null;
  probe_energy_eV: number | null;
  object_layers: number | null;
  object_height_px: number | null;
  object_width_px: number | null;
  probe_modes: number | null;
  probe_height_px: number | null;
  probe_width_px: number | null;
  num_scan_points: number | null;
}

export interface FluorescenceRead {
  uuid: string;
  label: string;
  comments: string;
  ingest_state: IngestState;
  derived_from: DerivedFromEdge[];
  element_names: string[];
  map_height_px: number | null;
  map_width_px: number | null;
}

export interface RenderedImage {
  png_base64: string;
  mime_type: 'image/png';
  value_label: string;
  color_value_min: number;
  color_value_max: number;
  pixel_width_m: number;
  pixel_height_m: number;
  shape_h_px: number;
  shape_w_px: number;
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${API}${path}`, { headers: { accept: 'application/json' } });
  if (!res.ok) {
    let body = '';
    try {
      body = await res.text();
    } catch {
      // ignore
    }
    throw new Error(`GET ${path} → ${res.status} ${res.statusText}: ${body}`);
  }
  return (await res.json()) as T;
}

export const api = {
  listDiffraction: (limit = 200, offset = 0) =>
    get<Page<DiffractionRead>>(`/diffraction?limit=${limit}&offset=${offset}`),
  getDiffraction: (uuid: string) => get<DiffractionRead>(`/diffraction/${uuid}`),
  diffractionAggregateImage: (uuid: string) =>
    get<RenderedImage>(`/diffraction/${uuid}/patterns/aggregate/image`),
  diffractionPatternImage: (uuid: string, index: number) =>
    get<RenderedImage>(`/diffraction/${uuid}/patterns/${index}/image`),

  listProduct: (limit = 200, offset = 0) =>
    get<Page<ProductRead>>(`/product?limit=${limit}&offset=${offset}`),
  getProduct: (uuid: string) => get<ProductRead>(`/product/${uuid}`),
  productObjectImage: (uuid: string, layer: number) =>
    get<RenderedImage>(`/product/${uuid}/object/${layer}/image`),
  productProbeImage: (uuid: string, incoherent = 0) =>
    get<RenderedImage>(`/product/${uuid}/probe/image?incoherent=${incoherent}`),
  productPositionsImage: (uuid: string, canvasPx = 512, connectPath = true) =>
    get<RenderedImage>(
      `/product/${uuid}/positions/image?canvas_px=${canvasPx}&connect_path=${connectPath}`
    ),

  listFluorescence: (limit = 200, offset = 0) =>
    get<Page<FluorescenceRead>>(`/fluorescence?limit=${limit}&offset=${offset}`),
  getFluorescence: (uuid: string) => get<FluorescenceRead>(`/fluorescence/${uuid}`),
  fluorescenceElementImage: (uuid: string, name: string) =>
    get<RenderedImage>(`/fluorescence/${uuid}/elements/${encodeURIComponent(name)}/image`),
};
