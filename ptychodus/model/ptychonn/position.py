from importlib.metadata import version
import logging

import numpy

import config_temp 
import .settings import PtychoNNPositionPredictionSettings

logger = logging.getLogger(__name__)

class PositionPredictionWorker:

    def __init__(self, positionPredictionSettings: PtychoNNPositionPredictionSettings, *, enableAmplitude: bool) -> None:
        self._positionPredictionSettings = positionPredictionSettings
        self._enableAmplitude = enableAmplitude

        ptychonnVersion = version('ptychonn')
        logger.info(f'\tPtychoNN {ptychonnVersion}')

    @property
    def name(self) -> str:
        return 'AmplitudePhase' if self._enableAmplitude else 'PhaseOnly'

    def _createModel(self) ->  ______:
        logger.debug('Building model...')
        return ______(
            ______,
            ______,
            enable_amplitude=self._enableAmplitude,
        )

    def run(self):
        scan_idx = 235

        configs = config_temp.InferenceConfig(
            reconstruction_image_path=os.path.join(
                "data", "position", "pred_test{}".format(scan_idx), "pred_phase.tiff"
            ),
            random_seed = self._positionPredictionSettings.randomSeed,
            debug=False,
            probe_position_list=None,
            central_crop=None,
            num_neighbors_collective=4,
            registration_params=config_temp.RegistrationConfig(
                registration_method="hybrid"
            ),
        )

        configs.load_from_toml(
            os.path.join("data", "position", "config_{}.toml".format(scan_idx))
        )
        print(configs)

#        corrector_chain = ptychonn.position.ProbePositionCorrectorChain(configs)
#        corrector_chain.verbose = False
#        corrector_chain.build()
#        corrector_chain.run()

        calc_pos_list = corrector_chain.corrector_list[-1].new_probe_positions.array

        gold_pos_list = np.genfromtxt(
            os.path.join("data_gold", "position", "calc_pos_235.csv"), delimiter=","
        )
        gold_pos_list = gold_pos_list / 8e-9
        calc_pos_list -= np.mean(calc_pos_list, axis=0)
        gold_pos_list -= np.mean(gold_pos_list, axis=0)
        print(gold_pos_list, calc_pos_list)
        assert np.allclose(calc_pos_list, gold_pos_list, atol=1e-1)
                
