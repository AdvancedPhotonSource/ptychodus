from dataclasses import dataclass
from pathlib import Path
import csv
import logging
import re

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.workflow import FileBasedWorkflow, WorkflowAPI

logger = logging.getLogger(__name__)


class PtychodusAutoloadProductFileBasedWorkflow(FileBasedWorkflow):
    @property
    def isWatchRecursive(self) -> bool:
        return True

    def getWatchFilePattern(self) -> str:
        return "product-out.npz"

    def execute(self, workflowAPI: WorkflowAPI, filePath: Path) -> None:
        workflowAPI.openProduct(filePath, fileType="NPZ")


class APS2IDFileBasedWorkflow(FileBasedWorkflow):
    @property
    def isWatchRecursive(self) -> bool:
        return False

    def getWatchFilePattern(self) -> str:
        return "*.csv"

    def execute(self, workflowAPI: WorkflowAPI, filePath: Path) -> None:
        scanName = filePath.stem
        scanID = int(re.findall(r"\d+", scanName)[-1])

        diffractionFilePath = filePath.parents[1] / "raw_data" / f"scan{scanID}_master.h5"
        workflowAPI.openPatterns(diffractionFilePath, fileType="NeXus")
        productAPI = workflowAPI.createProduct(f"scan{scanID}")
        productAPI.openScan(filePath, fileType="CSV")
        productAPI.buildProbe()
        productAPI.buildObject()
        productAPI.reconstructRemote()


class APS26IDFileBasedWorkflow(FileBasedWorkflow):
    @property
    def isWatchRecursive(self) -> bool:
        return False

    def getWatchFilePattern(self) -> str:
        return "*.mda"

    def execute(self, workflowAPI: WorkflowAPI, filePath: Path) -> None:
        scanName = filePath.stem
        scanID = int(re.findall(r"\d+", scanName)[-1])

        diffractionDirPath = filePath.parents[1] / "h5"

        for diffractionFilePath in diffractionDirPath.glob(f"scan_{scanID}_*.h5"):
            digits = int(re.findall(r"\d+", diffractionFilePath.stem)[-1])

            if digits != 0:
                break

        workflowAPI.openPatterns(diffractionFilePath, fileType="HDF5")
        productAPI = workflowAPI.createProduct(f"scan_{scanID}")
        productAPI.openScan(filePath, fileType="MDA")
        productAPI.buildProbe()
        productAPI.buildObject()
        productAPI.reconstructRemote()


@dataclass(frozen=True)
class APS31IDEMetadata:
    scan_no: int
    golden_angle: str
    encoder_angle: str
    measurement_id: str
    subtomo_no: str
    detector_position: str
    label: str

    def __str__(self) -> str:
        return f"""scan_no={self.scan_no}
        golden_angle={self.golden_angle}
        encoder_angle={self.encoder_angle}
        measurement_id={self.measurement_id}
        subtomo_no={self.subtomo_no}
        detector_position={self.detector_position}
        label={self.label}
        """


class APS31IDEFileBasedWorkflow(FileBasedWorkflow):
    @property
    def isWatchRecursive(self) -> bool:
        return True

    def getWatchFilePattern(self) -> str:
        return "*.h5"

    def execute(self, workflowAPI: WorkflowAPI, filePath: Path) -> None:
        experimentDir = filePath.parents[3]
        scan_no = int(re.findall(r"\d+", filePath.stem)[0])
        scanFile = experimentDir / "scan_positions" / f"scan_{scan_no:05d}.dat"
        scanNumbersFile = experimentDir / "dat-files" / "tomography_scannumbers.txt"
        metadata: APS31IDEMetadata | None = None

        with scanNumbersFile.open(newline="") as csvFile:
            csvReader = csv.reader(csvFile, delimiter=" ")

            for row in csvReader:
                if row[0].startswith("#"):
                    continue

                if len(row) != 7:
                    logger.warning("Unexpected row in tomography_scannumbers.txt!")
                    logger.debug(row)
                    continue

                try:
                    row_no = int(row[0])
                except ValueError:
                    logger.warning("Failed to parse row ID in tomography_scannumbers.txt!")
                    logger.debug(row[0])
                    continue

                if row_no == scan_no:
                    metadata = APS31IDEMetadata(
                        scan_no=scan_no,
                        golden_angle=str(row[1]),
                        encoder_angle=str(row[2]),
                        measurement_id=str(row[3]),
                        subtomo_no=str(row[4]),
                        detector_position=str(row[5]),
                        label=str(row[6]),
                    )
                    break

        if metadata is None:
            logger.warning(f"Failed to locate label for {row_no}!")
        else:
            productName = f"scan{scan_no:05d}_" + metadata.label
            workflowAPI.openPatterns(filePath, fileType="LYNX")
            inputProductAPI = workflowAPI.createProduct(productName, comments=str(metadata))
            inputProductAPI.openScan(scanFile, fileType="LYNXOrchestra")
            inputProductAPI.buildProbe()
            inputProductAPI.buildObject()
            # TODO would prefer to write instructions and submit to queue
            outputProductAPI = inputProductAPI.reconstructLocal(f"{productName}_out")
            outputProductAPI.saveProduct(
                experimentDir / "ptychodus" / f"{productName}.h5", fileType="HDF5"
            )


def registerPlugins(registry: PluginRegistry) -> None:
    registry.fileBasedWorkflows.registerPlugin(
        PtychodusAutoloadProductFileBasedWorkflow(),
        simpleName="Autoload_Product",
        displayName="Autoload Product",
    )
    registry.fileBasedWorkflows.registerPlugin(
        APS2IDFileBasedWorkflow(),
        simpleName="APS_2ID",
        displayName="APS 2-ID",
    )
    registry.fileBasedWorkflows.registerPlugin(
        APS26IDFileBasedWorkflow(),
        simpleName="APS_26ID",
        displayName="APS 26-ID",
    )
    registry.fileBasedWorkflows.registerPlugin(
        APS31IDEFileBasedWorkflow(),
        simpleName="APS_31IDE",
        displayName="APS 31-ID-E",
    )
