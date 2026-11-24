const API = '/api/v1';

/** Mirrors IngestState in db/base.py. */
export type IngestState =
  | 'DISCOVERED'
  | 'VALID'
  | 'INVALID'
  | 'MISSING_FILES'
  | 'ORPHANED'
  | 'AWAITING_DATA';

/** Mirrors RunState in ptychodus.api.processing. */
export type RunState = 'pending' | 'running' | 'succeeded' | 'failed' | 'cancelled';

/** How a processed resource came to be; see RunRead in service/schemas.py. */
export interface RunRead {
  state: RunState;
  reason: string | null;
  progress: number;
  /** The progress a finished run reaches; 0 if unknown. */
  progress_goal: number;
  command: string;
  updated_at: string;
}

export type Polarization = 'left_circular' | 'right_circular';
export type ScatteringRegime = 'transmission' | 'bragg' | 'laue';

export interface DerivedFromEdge {
  kind: 'diffraction' | 'product' | 'fluorescence';
  uuid: string;
  /** The target's display name; empty when the target is not indexed. */
  label: string;
}

export interface Page<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
}

/**
 * Valid choices for the render parameters; enumerated live by the service, each list
 * in the order the GUI presents it. The label maps give the GUI's names.
 */
export interface VisualizationOptions {
  colormaps_linear: string[];
  colormaps_cyclic: string[];
  transforms: string[];
  components: string[];
  color_models: string[];
  transform_labels: Record<string, string>;
  component_labels: Record<string, string>;
  color_model_labels: Record<string, string>;
}

/** What the viewer ribbon currently asks for. */
export interface RenderParams {
  colormap: string;
  transform: string;
  component: string | null;
  color_model: string | null;
  value_min: number | null;
  value_max: number | null;
  clip: boolean;
}

export function renderQuery(params: Partial<RenderParams>): string {
  const parts: string[] = [];
  for (const [key, value] of Object.entries(params)) {
    if (value === null || value === undefined || value === '') continue;
    parts.push(`${key}=${encodeURIComponent(String(value))}`);
  }
  return parts.join('&');
}

/** Bookkeeping every indexed row carries; see _RowBase in service/schemas.py. */
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
  /** Null for imported data. */
  run: RunRead | null;

  detector_distance_m: number | null;
  photon_energy_eV: number | null;
  probe_photon_count: number | null;
  exposure_time_s: number | null;
  tomography_angle_deg: number | null;
  tilt_angle_deg: number | null;
  polarization: Polarization | null;
  crop_center_x_px: number | null;
  crop_center_y_px: number | null;

  // The assembled patterns, so the extent and pixel pitch are the processed ones.
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
  /** Null for imported data. */
  run: RunRead | null;

  name: string | null;
  comments: string | null;
  detector_distance_m: number | null;
  focus_object_distance_m: number | null;
  far_field: boolean | null;
  photon_energy_eV: number | null;
  probe_photon_count: number | null;
  exposure_time_s: number | null;
  i0_photons_per_count: number | null;
  mass_attenuation_m2_per_kg: number | null;
  tomography_angle_deg: number | null;
  tilt_angle_deg: number | null;
  polarization: Polarization | null;
  two_theta_deg: number | null;
  detector_gamma_deg: number | null;
  incidence_angle_deg: number | null;
  scattering_regime: ScatteringRegime | null;

  object_layers: number | null;
  object_height_px: number | null;
  object_width_px: number | null;
  object_pixel_width_m: number | null;
  object_pixel_height_m: number | null;
  object_dtype: string | null;
  object_nbytes: number | null;
  object_layer_spacing_m: number[];
  object_total_thickness_m: number | null;

  probe_modes: number | null;
  probe_height_px: number | null;
  probe_width_px: number | null;
  probe_pixel_width_m: number | null;
  probe_pixel_height_m: number | null;
  probe_coherence: number | null;
  probe_dtype: string | null;
  probe_nbytes: number | null;
  probe_mode_relative_power: number[];

  num_probe_positions: number | null;
  num_loss_epochs: number | null;
  probe_path_length_m: number | null;
  probe_positions_nbytes: number | null;

  // Derived on read by compute_product_geometry. The last five additionally need
  // the detector, so they are null unless derived_from names an indexed diffraction
  // dataset. Non-finite results also arrive as null -- JSON has no infinity.
  photon_wavelength_m: number | null;
  probe_wavenumber_per_m: number | null;
  probe_angular_wavenumber_rad_per_m: number | null;
  probe_photon_flux_per_s: number | null;
  probe_power_W: number | null;
  object_plane_propagation_distance_m: number | null;
  magnification: number | null;
  object_plane_pixel_width_m: number | null;
  object_plane_pixel_height_m: number | null;
  fresnel_number: number | null;
  detector_numerical_aperture: number | null;
  depth_of_field_m: number | null;
}


export interface FluorescenceRead extends RowBase {
  label: string;
  comments: string;
  derived_from: DerivedFromEdge[];
  /** Null for imported data. */
  run: RunRead | null;

  element_names: string[];
  element_counts: number[];
  element_nbytes: number[];
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
  /** RGBA stops in [0, 1] from color_value_min to color_value_max, or around the wheel. */
  legend_rgba: [number, number, number, number][];
  legend_cyclic: boolean;
}

/**
 * A report's scalars, in SI units: numbers, booleans, nested groups and lists of them.
 * A statistic the report cannot define, such as I0 on a product without it, is null.
 */
export type ScalarValue = number | boolean | string | null | ScalarValue[] | Scalars;
export interface Scalars {
  [key: string]: ScalarValue;
}

/** One product's report; see _MetricsRead in service/schemas.py. */
export interface MetricsRead {
  uuid: string;
  run: RunRead | null;
  scalars: Scalars;
}

/** A product's report against a reference product; see _ComparisonRead. */
export interface ComparisonRead extends MetricsRead {
  reference_uuid: string;
  reference_run: RunRead | null;
}

export interface ProbeMetricsRead extends MetricsRead {}
export interface ObjectMetricsRead extends MetricsRead {}
export interface ProbePositionsMetricsRead extends MetricsRead {}
export interface ProbeComparisonRead extends ComparisonRead {}
export interface ObjectComparisonRead extends ComparisonRead {}
export interface ProbePositionsComparisonRead extends ComparisonRead {}

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

/** Every row of a listing, fetched a page at a time up to the routes' cap on `limit`. */
async function getAll<T>(path: string): Promise<Page<T>> {
  const limit = 200;
  const items: T[] = [];
  let total = 0;
  do {
    const page = await get<Page<T>>(`${path}?limit=${limit}&offset=${items.length}`);
    items.push(...page.items);
    total = page.total;
    // A listing that shrank while it was read ends here rather than looping.
    if (page.items.length === 0) break;
  } while (items.length < total);
  return { items, total, limit: items.length, offset: 0 };
}

export const api = {
  listDiffraction: () => getAll<DiffractionRead>('/diffraction'),
  getDiffraction: (uuid: string) => get<DiffractionRead>(`/diffraction/${uuid}`),
  diffractionAggregateImage: (uuid: string, params: Partial<RenderParams> = {}) =>
    get<RenderedImage>(`/diffraction/${uuid}/patterns/aggregate/image?${renderQuery(params)}`),
  diffractionPatternImage: (uuid: string, index: number, params: Partial<RenderParams> = {}) =>
    get<RenderedImage>(`/diffraction/${uuid}/patterns/${index}/image?${renderQuery(params)}`),
  diffractionFileUrl: (uuid: string) => `${API}/diffraction/${uuid}/files/diffraction`,

  listProduct: () => getAll<ProductRead>('/product'),
  getProduct: (uuid: string) => get<ProductRead>(`/product/${uuid}`),
  // The product of every layer, which the GUI shows for the object itself.
  productObjectFlattenedImage: (uuid: string, params: Partial<RenderParams> = {}) =>
    get<RenderedImage>(`/product/${uuid}/object/image?${renderQuery(params)}`),
  productObjectImage: (uuid: string, layer: number, params: Partial<RenderParams> = {}) =>
    get<RenderedImage>(`/product/${uuid}/object/${layer}/image?${renderQuery(params)}`),
  // The per-mode render: a route the service has always had and the UI never called.
  productProbeImage: (uuid: string, incoherent: number, params: Partial<RenderParams> = {}) =>
    get<RenderedImage>(
      `/product/${uuid}/probe/image?incoherent=${incoherent}&${renderQuery(params)}`
    ),
  productProbeModesImage: (uuid: string, params: Partial<RenderParams> = {}) =>
    get<RenderedImage>(`/product/${uuid}/probe/modes/image?${renderQuery(params)}`),
  // Takes a list so the Positions page can overlay checked products on shared axes.
  productPositionsImage: (
    uuids: string[],
    connectPath = true,
    sizePx?: { width: number; height: number }
  ) =>
    get<PlotImage>(
      `/product/positions/image?${uuids
        .map((u) => `uuid=${encodeURIComponent(u)}`)
        .join('&')}&connect_path=${connectPath}` +
        (sizePx ? `&width_px=${sizePx.width}&height_px=${sizePx.height}` : '')
    ),
  productFileUrl: (uuid: string) => `${API}/product/${uuid}/files/product`,
  productProbeMetrics: (uuid: string) =>
    get<ProbeMetricsRead>(`/product/${uuid}/probe/metrics`),
  productObjectMetrics: (uuid: string) =>
    get<ObjectMetricsRead>(`/product/${uuid}/object/metrics`),
  productPositionsMetrics: (uuid: string) =>
    get<ProbePositionsMetricsRead>(`/product/${uuid}/positions/metrics`),
  productProbeComparison: (uuid: string, referenceUuid: string) =>
    get<ProbeComparisonRead>(
      `/product/${uuid}/probe/comparison?reference_uuid=${encodeURIComponent(referenceUuid)}`
    ),
  productObjectComparison: (uuid: string, referenceUuid: string) =>
    get<ObjectComparisonRead>(
      `/product/${uuid}/object/comparison?reference_uuid=${encodeURIComponent(referenceUuid)}`
    ),
  productPositionsComparison: (uuid: string, referenceUuid: string) =>
    get<ProbePositionsComparisonRead>(
      `/product/${uuid}/positions/comparison?reference_uuid=${encodeURIComponent(referenceUuid)}`
    ),

  visualizationOptions: () => get<VisualizationOptions>('/visualization/options'),
  listFluorescence: () => getAll<FluorescenceRead>('/fluorescence'),
  getFluorescence: (uuid: string) => get<FluorescenceRead>(`/fluorescence/${uuid}`),
  // product_uuid supplies the pixel geometry; without it the service falls back to a
  // 1 um placeholder and the caption's physical size is wrong.
  fluorescenceSummaryImage: (
    uuid: string,
    productUuid?: string,
    params: Partial<RenderParams> = {}
  ) =>
    get<RenderedImage>(
      `/fluorescence/${uuid}/image?` +
        renderQuery({ ...params, ...(productUuid ? { product_uuid: productUuid } : {}) })
    ),
  fluorescenceElementImage: (
    uuid: string,
    name: string,
    productUuid?: string,
    params: Partial<RenderParams> = {}
  ) =>
    get<RenderedImage>(
      `/fluorescence/${uuid}/elements/${encodeURIComponent(name)}/image?` +
        renderQuery({ ...params, ...(productUuid ? { product_uuid: productUuid } : {}) })
    ),
  fluorescenceFileUrl: (uuid: string) => `${API}/fluorescence/${uuid}/files/fluorescence`,
};
