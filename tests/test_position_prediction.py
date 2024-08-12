import argparse
from pathlib import Path

import numpy as np

from ptychodus.model.ptychonn.position import PositionPredictionWorker
from ptychodus.model.ptychonn.settings import PtychoNNPositionPredictionSettings
from ptychodus.api.settings import SettingsRegistry


def test_position_prediction(generate_gold: bool = False, debug: bool = False) -> None:
    gold_dir = "gold_data/test_position_prediction"
    
    settings = PtychoNNPositionPredictionSettings(SettingsRegistry())
    
    settings.reconstructorImagePath.value = Path('data') / 'position_prediction' / 'pred_test235' / 'pred_phase.tiff'
    settings.probePositionListPath.value = Path('data') / 'position_prediction' / 'pred_test235' / 'nominal_pos.csv'
    settings.probePositionDataUnit.value = 'm'
    settings.pixelSizeNM.value = '8.0'
    settings.baselinePositionListPath.value = Path('')
    settings.centralCrop.value = 0
    settings.method.value = 'collective'
    settings.numberNeighborsCollective.value = 6
    settings.offsetEstimatorOrder.value = 1
    settings.offsetEstimatorBeta.value = '0.5'
    settings.smoothConstraintWeight.value = '1e-2'
    settings.rectangularGrid.value = False
    settings.randomSeed.value = 123
    settings.debug.value = False
    settings.registrationMethod.value = 'hybrid'
    settings.hybridRegistrationTols.value = '0.15,0.3'
    settings.nonHybridRegistrationTol.value = 0.15
    settings.maxShift.value = 40
    
    worker = PositionPredictionWorker(settings)
    worker.run()
    
    predicted_pos = worker.getPredictedPositions()
    
    if debug:
        worker._corrector.new_probe_positions.plot()
    
    if generate_gold:
        np.savetxt(Path(gold_dir) / 'predicted_pos.csv', predicted_pos, delimiter=',')
        
    if not generate_gold:
        gold_pos = np.genfromtxt(Path(gold_dir) / 'predicted_pos.csv', delimiter=',')
        assert np.allclose(predicted_pos, gold_pos)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-gold", action="store_true")
    args = parser.parse_args()
    
    test_position_prediction(generate_gold=args.generate_gold, debug=True)
    