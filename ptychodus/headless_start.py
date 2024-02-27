import sys
from pathlib import Path
from model.data.api import DiffractionDataAPI
from model.automation.workflow import PtychoNNTrainingAutomationDatasetWorkflow
from api.plugins import PluginChooser
from api.data import DiffractionFileReader

def main(hdf5_file_path: str):
    # Initialize necessary components
    fileReaderChooser = PluginChooser[DiffractionFileReader]()
    dataAPI = DiffractionDataAPI(fileReaderChooser=fileReaderChooser)
    # Assuming PtychoNNTrainingAutomationDatasetWorkflow is the correct workflow for ptychopinn
    workflow = PtychoNNTrainingAutomationDatasetWorkflow(dataAPI=dataAPI)

    # Load HDF5 diffraction data
    dataAPI.loadDiffractionDataset(Path(hdf5_file_path), fileType='HDF5')

    # Start training
    workflow.execute(Path(hdf5_file_path))

if __name__ == "__main__":
    main(sys.argv[1])
