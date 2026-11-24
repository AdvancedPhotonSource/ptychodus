/**
 * The webapp's MCP render tools, each drawn in the chat panel through the REST route that
 * renders the same image: the agent service sends no image data over its stream.
 */

import { api, type PlotImage, type RenderParams, type RenderedImage } from '../api.js';
import { linkTo } from '../routes.js';

type Args = Record<string, unknown>;

/** A render tool call, as the panel draws it. */
export interface RenderCall {
  image: () => Promise<RenderedImage | PlotImage>;
  /** The web UI's deep link to the rendered resource. */
  link: string;
}

function text(args: Args, name: string): string {
  const value = args[name];
  return typeof value === 'string' ? value : String(value ?? '');
}

function integer(args: Args, name: string, fallback: number): number {
  const value = args[name];
  return typeof value === 'number' && Number.isInteger(value) ? value : fallback;
}

function optionalText(args: Args, name: string): string | undefined {
  const value = args[name];
  return typeof value === 'string' && value !== '' ? value : undefined;
}

/** The render parameters among a tool call's arguments. */
function renderParams(args: Args): Partial<RenderParams> {
  const params: Partial<RenderParams> = {};
  if (typeof args.colormap === 'string') params.colormap = args.colormap;
  if (typeof args.transform === 'string') params.transform = args.transform;
  if (typeof args.component === 'string') params.component = args.component;
  if (typeof args.color_model === 'string') params.color_model = args.color_model;
  if (typeof args.value_min === 'number') params.value_min = args.value_min;
  if (typeof args.value_max === 'number') params.value_max = args.value_max;
  if (typeof args.clip === 'boolean') params.clip = args.clip;
  return params;
}

export const RENDER_CALLS: Record<string, (args: Args) => RenderCall> = {
  render_diffraction_pattern: (args) => ({
    image: () =>
      api.diffractionPatternImage(text(args, 'uuid'), integer(args, 'index', 0), renderParams(args)),
    link: linkTo('diffraction', text(args, 'uuid')),
  }),
  render_diffraction_aggregate: (args) => ({
    image: () => api.diffractionAggregateImage(text(args, 'uuid'), renderParams(args)),
    link: linkTo('diffraction', text(args, 'uuid')),
  }),
  render_probe: (args) => ({
    image: () =>
      api.productProbeImage(text(args, 'uuid'), integer(args, 'incoherent', 0), renderParams(args)),
    link: linkTo('probe', text(args, 'uuid')),
  }),
  render_probe_modes: (args) => ({
    image: () => api.productProbeModesImage(text(args, 'uuid'), renderParams(args)),
    link: linkTo('probe', text(args, 'uuid')),
  }),
  render_object: (args) => ({
    image: () => api.productObjectFlattenedImage(text(args, 'uuid'), renderParams(args)),
    link: linkTo('object', text(args, 'uuid')),
  }),
  render_object_layer: (args) => ({
    image: () =>
      api.productObjectImage(text(args, 'uuid'), integer(args, 'layer', 0), renderParams(args)),
    link: linkTo('object', text(args, 'uuid')),
  }),
  render_probe_positions: (args) => {
    const uuids = Array.isArray(args.uuid) ? args.uuid.map(String) : [text(args, 'uuid')];
    const sizePx = {
      width: integer(args, 'width_px', 640),
      height: integer(args, 'height_px', 640),
    };
    return {
      image: () => api.productPositionsImage(uuids, args.connect_path !== false, sizePx),
      link: linkTo('positions', uuids[0] ?? ''),
    };
  },
  render_fluorescence_summary: (args) => ({
    image: () =>
      api.fluorescenceSummaryImage(
        text(args, 'uuid'),
        optionalText(args, 'product_uuid'),
        renderParams(args)
      ),
    link: linkTo('fluorescence', text(args, 'uuid')),
  }),
  render_fluorescence_element: (args) => ({
    image: () =>
      api.fluorescenceElementImage(
        text(args, 'uuid'),
        text(args, 'name'),
        optionalText(args, 'product_uuid'),
        renderParams(args)
      ),
    link: linkTo('fluorescence', text(args, 'uuid')),
  }),
};
