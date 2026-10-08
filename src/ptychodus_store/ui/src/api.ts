const API = '/api/v1';

export type IngestState = 'valid' | 'invalid' | 'pending';

export type Polarization = 'left_circular' | 'right_circular';

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

/** Bookkeeping every indexed row carries; see _RowBase in routers/schemas.py. */
export interface RowBase {
  uuid: string;
  folder_path: string;
  manifest_mtime: string | null;
  created_from_manifest_at: string | null;
  ingest_state: IngestState;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

export interface DiffractionRead extends RowBase {
  label: string;
  comments: string;
  campaign_uuid: string | null;
  derived_from: DerivedFromEdge[];

  detector_distance_m: number | null;
  photon_energy_eV: number | null;
  probe_photon_count: number | null;
  exposure_time_s: number | null;
  tomography_angle_deg: number | null;
  tilt_angle_deg: number | null;
  polarization: Polarization | null;
  crop_center_x_px: number | null;
  crop_center_y_px: number | null;

  pattern_dtype: string | null;
  pattern_height_px: number | null;
  pattern_width_px: number | null;
  num_patterns_total: number | null;
  detector_pixel_width_m: number | null;
  detector_pixel_height_m: number | null;
  num_bad_pixels: number | null;
  nbytes: number | null;
}

export interface ProductRead extends RowBase {
  derived_from: DerivedFromEdge[];

  name: string | null;
  comments: string | null;
  detector_distance_m: number | null;
  focus_object_distance_m: number | null;
  far_field: boolean | null;
  photon_energy_eV: number | null;
  probe_photon_count: number | null;
  exposure_time_s: number | null;
  mass_attenuation_m2_per_kg: number | null;
  tomography_angle_deg: number | null;
  tilt_angle_deg: number | null;
  polarization: Polarization | null;

  object_layers: number | null;
  object_height_px: number | null;
  object_width_px: number | null;
  object_pixel_width_m: number | null;
  object_pixel_height_m: number | null;
  object_dtype: string | null;
  object_nbytes: number | null;
  object_layer_spacing_m: number[];

  probe_modes: number | null;
  probe_height_px: number | null;
  probe_width_px: number | null;
  probe_dtype: string | null;
  probe_nbytes: number | null;
  probe_mode_relative_power: number[];

  num_scan_points: number | null;
  num_loss_epochs: number | null;
  scan_length_m: number | null;
  scan_nbytes: number | null;

  // Derived on read by compute_product_geometry. The last three additionally need
  // the detector, so they are null unless derived_from names an indexed diffraction
  // dataset. Non-finite results also arrive as null -- JSON has no infinity.
  probe_wavenumber_per_m: number | null;
  probe_angular_wavenumber_rad_per_m: number | null;
  probe_photon_flux_per_s: number | null;
  probe_power_W: number | null;
  object_plane_propagation_distance_m: number | null;
  fresnel_number: number | null;
  detector_numerical_aperture: number | null;
  depth_of_field_m: number | null;
}


export interface FluorescenceRead extends RowBase {
  label: string;
  comments: string;
  derived_from: DerivedFromEdge[];

  element_names: string[];
  element_counts: number[];
  map_height_px: number | null;
  map_width_px: number | null;
  nbytes: number | null;
}

export interface PlotImage {
  png_base64: string;
  mime_type: string;
  shape_h_px: number;
  shape_w_px: number;
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
  diffractionFileUrl: (uuid: string) => `${API}/diffraction/${uuid}/files/diffraction`,

  listProduct: (limit = 200, offset = 0) =>
    get<Page<ProductRead>>(`/product?limit=${limit}&offset=${offset}`),
  getProduct: (uuid: string) => get<ProductRead>(`/product/${uuid}`),
  productObjectImage: (uuid: string, layer: number, colorModel = 'hsv_value') =>
    get<RenderedImage>(
      `/product/${uuid}/object/${layer}/image?color_model=${encodeURIComponent(colorModel)}`
    ),
  productProbeModesImage: (uuid: string, colorModel = 'hsv_value') =>
    get<RenderedImage>(
      `/product/${uuid}/probe/modes/image?color_model=${encodeURIComponent(colorModel)}`
    ),
  // Takes a list so the Positions page can overlay checked scans on shared axes.
  productPositionsImage: (uuids: string[], connectPath = true) =>
    get<PlotImage>(
      `/product/positions/image?${uuids
        .map((u) => `uuid=${encodeURIComponent(u)}`)
        .join('&')}&connect_path=${connectPath}`
    ),
  productFileUrl: (uuid: string) => `${API}/product/${uuid}/files/product`,

  listFluorescence: (limit = 200, offset = 0) =>
    get<Page<FluorescenceRead>>(`/fluorescence?limit=${limit}&offset=${offset}`),
  getFluorescence: (uuid: string) => get<FluorescenceRead>(`/fluorescence/${uuid}`),
  fluorescenceElementImage: (uuid: string, name: string) =>
    get<RenderedImage>(`/fluorescence/${uuid}/elements/${encodeURIComponent(name)}/image`),
  fluorescenceFileUrl: (uuid: string) => `${API}/fluorescence/${uuid}/files/fluorescence`,
};
