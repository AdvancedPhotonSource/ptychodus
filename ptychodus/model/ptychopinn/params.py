"""
Stores global variables for data generation and model configuration
"""
# TODO naming convention for different types of parameters
# TODO what default value and initialization for the probe scale?
from ptychodus.api.settings import SettingsRegistry

cfg = {
    # Shared settings with model/ptychopinn/settings.py
    'N': 64, 'offset': 4, 'gridsize': 2,
    'outer_offset_train': None, 'outer_offset_test': None, 'batch_size': 16,
    'nepochs': 60, 'n_filters_scale': 2, 'output_prefix': 'outputs',
    'big_gridsize': 10, 'max_position_jitter': 10, 'sim_jitter_scale': 0.,
    'default_probe_scale': 0.7, 'mae_weight': 0., 'nll_weight': 1., 'tv_weight': 0.,
    'realspace_mae_weight': 0., 'realspace_weight': 0., 'nphotons': 1e9,
    'nimgs_train': 9, 'nimgs_test': 3,
    'data_source': 'lines', 'probe.trainable': False,
    'intensity_scale.trainable': False, 'positions.provided': False,
    'object.big': True, 'probe.big': False, 'probe_scale': 10., 'set_phi': False,
    'probe.mask': True, 'model_type': 'pinn', 'label': '', 'size': 392,
    'amp_activation': 'sigmoid',
    }

# TODO parameter description
# probe.big: if True, increase the real space solution from 32x32 to 64x64

# TODO bigoffset should be a derived quantity, at least for simulation
def get_bigN():
    N = cfg['N']
    gridsize = cfg['gridsize']
    offset = cfg['offset']
    return N + (gridsize - 1) * offset

def get_padding_size():
    buffer = cfg['max_position_jitter']
    gridsize = cfg['gridsize']
    offset = cfg['offset']
    return (gridsize - 1) * offset + buffer

def get_padded_size():
    bigN = get_bigN()
    buffer = cfg['max_position_jitter']
    return bigN + buffer

def params():
    d = {k:v for k, v in cfg.items()}
    d['bigN'] = get_bigN()
    return d

def update_cfg_from_settings(settings_registry: SettingsRegistry):
    settings_dict = settings_registry.to_dict()
    ptychopinn_settings = settings_dict.get('PtychoPINN', {})
    ptychopinn_training_settings = settings_dict.get('PtychoPINNTraining', {})

    # Update shared settings
    cfg['N'] = int(ptychopinn_settings.get('N', cfg['N']))
    cfg['gridsize'] = int(ptychopinn_settings.get('Gridsize', cfg['gridsize']))
    cfg['batch_size'] = int(ptychopinn_settings.get('BatchSize', cfg['batch_size']))
    cfg['n_filters_scale'] = int(ptychopinn_settings.get('NFiltersScale', cfg['n_filters_scale']))
    cfg['output_prefix'] = str(ptychopinn_settings.get('OutputPrefix', cfg['output_prefix']))
    cfg['default_probe_scale'] = float(ptychopinn_settings.get('DefaultProbeScale', cfg['default_probe_scale']))
    cfg['nphotons'] = float(ptychopinn_settings.get('NPhotons', cfg['nphotons']))
    cfg['object.big'] = bool(ptychopinn_settings.get('ObjectBig', cfg['object.big']))
    cfg['probe.big'] = bool(ptychopinn_settings.get('ProbeBig', cfg['probe.big']))
    cfg['probe_scale'] = float(ptychopinn_settings.get('ProbeScale', cfg['probe_scale']))
    cfg['probe.mask'] = bool(ptychopinn_settings.get('ProbeMask', cfg['probe.mask']))
    cfg['model_type'] = str(ptychopinn_settings.get('ModelType', cfg['model_type']))
    cfg['amp_activation'] = str(ptychopinn_settings.get('AmpActivation', cfg['amp_activation']))

    # Settings specific to training
    # Note: Some settings may not be directly transferable and may require additional logic

# TODO refactor
def validate():
    valid_data_sources = ['lines', 'grf', 'experimental', 'points',
        'testimg', 'diagonals', 'xpp', 'V', 'generic']
    assert cfg['data_source'] in valid_data_sources, \
        f"Invalid data source: {cfg['data_source']}. Must be one of {valid_data_sources}."
    if cfg['realspace_mae_weight'] > 0.:
        assert cfg['realspace_weight'] > 0
    #assert cfg['bigoffset'] % 4 == 0
    # TODO
    return True

def set(key, value):
    cfg[key] = value
    assert validate()

def get(key):
    if key == 'bigN':
        cfg['bigN'] = get_bigN()
        return cfg['bigN']
    return cfg[key]
