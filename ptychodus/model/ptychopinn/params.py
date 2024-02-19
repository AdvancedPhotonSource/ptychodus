"""
Stores global variables for data generation and model configuration
"""
from .settings import PtychoPINNModelSettings
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

def update_cfg_from_settings(model_settings: PtychoPINNModelSettings):
    settings_registry = model_settings._settingsGroup._settingsRegistry
    if settings_registry is not None:
        settings_dict = settings_registry.to_dict()
    else:
        settings_dict = {}
    ptychopinn_settings = settings_dict.get('PtychoPINN', {})
    ptychopinn_training_settings = settings_dict.get('PtychoPINNTraining', {})

    # Define a mapping from settings registry keys to cfg keys
    key_mapping = {
        'LearningRate': 'learning_rate',
        'N': 'N',
        'Offset': 'offset',
        'Gridsize': 'gridsize',
        'BatchSize': 'batch_size',
        'NFiltersScale': 'n_filters_scale',
        'NPhotons': 'nphotons',
        'ProbeTrainable': 'probe.trainable',
        'IntensityScaleTrainable': 'intensity_scale.trainable',
        'ObjectBig': 'object.big',
        'ProbeBig': 'probe.big',
        'ProbeScale': 'probe_scale',
        'ProbeMask': 'probe.mask',
        'ModelType': 'model_type',
        'Size': 'size',
        'AmpActivation': 'amp_activation',
        'MAEWeight': 'mae_weight',
        'NLLWeight': 'nll_weight',
        'TVWeight': 'tv_weight',
        'RealspaceMAEWeight': 'realspace_mae_weight',
        'RealspaceWeight': 'realspace_weight',
    }

    # Update shared settings
    for registry_key, cfg_key in key_mapping.items():
        if registry_key in ptychopinn_settings or registry_key in ptychopinn_training_settings:
            cfg[cfg_key] = ptychopinn_settings.get(registry_key, ptychopinn_training_settings.get(registry_key, cfg[cfg_key]))

    # Update derived values
    cfg['bigN'] = get_bigN()
    cfg['padded_size'] = get_padded_size()
    cfg['padding_size'] = get_padding_size()

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
