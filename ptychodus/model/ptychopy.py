from __future__ import annotations
from pathlib import Path
from typing import Final
import logging

try:
    import ptychopy
except ModuleNotFoundError:
    ptychopy = None

from ..api.observer import Observable, Observer
from ..api.settings import SettingsRegistry, SettingsGroup
from .reconstructor import Reconstructor

logger = logging.getLogger(__name__)


class PtychoPySettings(Observable, Observer):

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.probeModes = settingsGroup.createIntegerEntry('ProbeModes', 1)
        self.threshold = settingsGroup.createIntegerEntry('Threshold', 0)
        self.reconstructionIterations = settingsGroup.createIntegerEntry(
            'ReconstructionIterations', 100)
        self.reconstructionTimeInSeconds = settingsGroup.createIntegerEntry(
            'ReconstructionTimeInSeconds', 0)
        self.calculateRMS = settingsGroup.createBooleanEntry('RMS', False)
        self.updateProbe = settingsGroup.createIntegerEntry('UpdateProbe', 10)
        self.updateModes = settingsGroup.createIntegerEntry('UpdateModes', 20)
        self.phaseConstraint = settingsGroup.createIntegerEntry('PhaseConstraint', 1)

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> PtychoPySettings:
        settings = cls(settingsRegistry.createGroup('PtychoPy'))
        settings._settingsGroup.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class CommandStringBuilder:

    def __init__(self, jobID: str, algorithm: str) -> None:
        self._sb = [f'./ptycho -jobID={jobID} -algorithm={algorithm}']

    @classmethod
    def createEPIEString(cls, jobID: str) -> CommandStringBuilder:
        return cls(jobID, 'ePIE')

    @classmethod
    def createDMString(cls, jobID: str) -> CommandStringBuilder:
        return cls(jobID, 'DM')

    @classmethod
    def createMLsString(cls, jobID: str) -> CommandStringBuilder:
        return cls(jobID, 'MLs')

    def build(self) -> str:
        return ' '.join(self._sb)

    def withNumberOfReconconstructionIterations(self,
                                                iterations: int = 100) -> CommandStringBuilder:
        self._sb.append(f'-i={iterations}')
        return self

    def withMaximumAllowedReconstructionTime(self,
                                             timeInSeconds: int = 10) -> CommandStringBuilder:
        # Maximum allowed reconstruction time (in sec). Overrides iterations.
        self._sb.append(f'-T={timeInSeconds}')
        return self

    def withProbePositionCSVFile(self, filePath: Path) -> CommandStringBuilder:
        # The file can be saved with python using np.savetxt(numpyProbePositions, delimiter=', ') and it has to be arranged with position y, x in each row, the unit is m
        self._sb.append(f'-lf={filePath}')
        return self

    def withObjectGuessCSVFile(self, filePath: Path) -> CommandStringBuilder:
        # The location of a CSV complex valued file used as the initial object guess for reconstruction
        self._sb.append(f'-objectGuess={filePath}')
        return self

    def withProbeGuessCSVFile(self, filePath: Path) -> CommandStringBuilder:
        # The location of a CSV complex valued file used as the initial probe guess for reconstruction
        # The file can be saved with python using np.savetxt(numpyProbe, delimiter=', ') and it has to have the same dimensions as the size parameter
        self._sb.append(f'-probeGuess={filePath}')
        return self

    def withIncidentBeamWavelengthInMeters(self,
                                           wavelengthInMeters: float = 2.3843e-10
                                           ) -> CommandStringBuilder:
        self._sb.append(f'-lambda={wavelengthInMeters}')
        return self

    def withIncidentBeamSizeInMeters(self,
                                     beamSizeInMeters: float = 400.e-9) -> CommandStringBuilder:
        self._sb.append(f'-beamSize={beamSizeInMeters}')
        return self

    def withBitDepth(self, bitDepth: int = 32) -> CommandStringBuilder:
        self._sb.append(f'-bitDepth={bitDepth}')
        return self

    def withNumberOfDiffractionPatternsPerFile(self,
                                               patternsPerFile: int = 1) -> CommandStringBuilder:
        self._sb.append(f'-dpf={patternsPerFile}')
        return self

    def withDetectorPixelSizeInMeters(self,
                                      pixelSizeInMeters: float = 75.e-6) -> CommandStringBuilder:
        self._sb.append(f'-dx_d={pixelSizeInMeters}')
        return self

    def withDetectorDistanceInMeters(self, distanceInMeters: float = 2.2) -> CommandStringBuilder:
        # Distance between sample and detector in meters
        self._sb.append(f'-z={distanceInMeters}')
        return self

    def withDiffractionPatternGeometryInPixels(self,
                                               centerX: int = 256,
                                               centerY: int = 256,
                                               cropSize: int = 512) -> CommandStringBuilder:
        # The center of the diffraction pattern in pixels (image pixel location Y, image pixel location X).
        self._sb.append(f'-qxy={centerX},{centerY}')
        # The desired size for cropping the diffraction patterns and probe size.
        self._sb.append(f'-size={cropSize}')
        # Diffraction patterns will be cropped to a square image of sizeXsize pixels around qxy.
        return self

    def withDiffractionPatternData(self,
                                   pathFormat: str,
                                   firstFileIndex: int = 0) -> CommandStringBuilder:
        # A c-style formatted string for the location of the HDF5 files. For file name string substitution starting at fs .
        self._sb.append(f'-fp={pathFormat}')
        # The file index of the file containing the first diffraction pattern (top left corner for Cartesian scans)
        self._sb.append(f'-fs={firstFileIndex}')
        return self

    def withDatasetName(self, name: str = '/entry/data/data') -> CommandStringBuilder:
        # Diffraction data HDF5 dataset name
        self._sb.append(f'-hdf5path={name}')
        return self

    def withNumberOfOrthogonalProbeModes(self, count: int = 1) -> CommandStringBuilder:
        # Number of orthogonal probe modes to simulate partial incoherence of the beam
        self._sb.append(f'-probeModes={count}')
        return self

    def withDiffractionCountFloor(self, count: int = 0) -> CommandStringBuilder:
        # To remove noise from the diffraction patterns. Any count below this number will be set to zero in the diffraction data.
        self._sb.append(f'-threshold={count}')
        return self

    def withRootMeanSquareCalculation(self, rms: bool = False) -> CommandStringBuilder:
        self._sb.append(f'-RMS={1 if rms else 0}')
        return self

    def withPhaseConstraint(self, iterations: int = 1) -> CommandStringBuilder:
        # The number of iterations to keep applying a phase constraint (forcing the reconstructed phase in the range [-2pi, 0])
        self._sb.append(f'-phaseConstraint={iterations}')
        return self

    def withProbeUpdateWait(self,
                            primaryModeWait: int = 10,
                            allModesWait: int = 20) -> CommandStringBuilder:
        # The number of iterations after which to start updating the primary probe mode
        self._sb.append(f'-updateProbe={primaryModeWait}')
        # The number of iterations after which to start updating all probe modes
        self._sb.append(f'-updateModes={allModesWait}')
        return self

    def withProbePositionSearchWait(self, numberOfIterations: int = 20) -> CommandStringBuilder:
        # The number of iterations after which to start probe position search
        assert 'MLs' in self._sb[0]
        self._sb.append(f'-PPS={numberOfIterations}')
        return self

    def withLeastSquaresDampingConstant(self,
                                        dampingConstant: float = 0.1) -> CommandStringBuilder:
        assert 'MLs' in self._sb[0]
        self._sb.append(f'-delta_p={dampingConstant}')
        return self

    def withSquareRootTransformedDiffractionPatternMagnitudes(self,
                                                              sqrtData: bool = False
                                                              ) -> CommandStringBuilder:
        self._sb.append(f'-sqrtData={1 if sqrtData else 0}')
        return self

    def withFFTShiftedDiffractionPatterns(self,
                                          fftShiftData: bool = False) -> CommandStringBuilder:
        self._sb.append(f'-fftShiftData={1 if fftShiftData else 0}')
        return self

    def withBeamstopAreaFourierModulusConstraint(self,
                                                 beamstopMask: bool = False
                                                 ) -> CommandStringBuilder:
        # Determine whether the beamstop area (0 values, set by a binary array "beamstopMask.h5" which is put in the data directory) is applied with Fourier modulus constraint or not
        self._sb.append(f'-beamstopMask={1 if beamstopMask else 0}')
        return self

    def withBinaryOutput(self, binaryOutput: bool = False) -> CommandStringBuilder:
        # Write results in binary format (1), or CSV format (0)
        self._sb.append(f'-binaryOutput={1 if binaryOutput else 0}')
        return self

    def withDrift(self, driftX: float, driftY: float) -> CommandStringBuilder:
        self._sb.append(f'-drift={driftX},{driftY}')
        return self

    def withNumberOfIOThreads(self, count: int) -> CommandStringBuilder:
        self._sb.append(f'-threads={count}')
        return self


class PtychoPyPresenter(Observable, Observer):
    MAX_INT: Final[int] = 0x7FFFFFFF

    def __init__(self, settings: PtychoPySettings) -> None:
        super().__init__()
        self._settings = settings

    @classmethod
    def createInstance(cls, settings: PtychoPySettings) -> PtychoPyPresenter:
        presenter = cls(settings)
        settings.addObserver(presenter)
        return presenter

    def getMinProbeModes(self) -> int:
        return 1

    def getMaxProbeModes(self) -> int:
        return self.MAX_INT

    def getProbeModes(self) -> int:
        return self._settings.probeModes.value

    def setProbeModes(self, value: int) -> None:
        self._settings.probeModes.value = value

    def getMinThreshold(self) -> int:
        return 0

    def getMaxThreshold(self) -> int:
        return self.MAX_INT

    def getThreshold(self) -> int:
        return self._settings.threshold.value

    def setThreshold(self, value: int) -> None:
        self._settings.threshold.value = value

    def getMinReconstructionIterations(self) -> int:
        return 0

    def getMaxReconstructionIterations(self) -> int:
        return self.MAX_INT

    def getReconstructionIterations(self) -> int:
        return self._settings.reconstructionIterations.value

    def setReconstructionIterations(self, value: int) -> None:
        self._settings.reconstructionIterations.value = value

    def getMinReconstructionTimeInSeconds(self) -> int:
        return 0

    def getMaxReconstructionTimeInSeconds(self) -> int:
        return self.MAX_INT

    def getReconstructionTimeInSeconds(self) -> int:
        return self._settings.reconstructionTimeInSeconds.value

    def setReconstructionTimeInSeconds(self, value: int) -> None:
        self._settings.reconstructionTimeInSeconds.value = value

    def isCalculateRMSEnabled(self) -> bool:
        return self._settings.calculateRMS.value

    def setCalculateRMSEnabled(self, enabled: bool) -> None:
        self._settings.calculateRMS.value = enabled

    def getMinUpdateProbe(self) -> int:
        return 0

    def getMaxUpdateProbe(self) -> int:
        return self.MAX_INT

    def getUpdateProbe(self) -> int:
        return self._settings.updateProbe.value

    def setUpdateProbe(self, value: int) -> None:
        self._settings.updateProbe.value = value

    def getMinUpdateModes(self) -> int:
        return 0

    def getMaxUpdateModes(self) -> int:
        return self.MAX_INT

    def getUpdateModes(self) -> int:
        return self._settings.updateModes.value

    def setUpdateModes(self, value: int) -> None:
        self._settings.updateModes.value = value

    def getMinPhaseConstraint(self) -> int:
        return 0

    def getMaxPhaseConstraint(self) -> int:
        return self.MAX_INT

    def getPhaseConstraint(self) -> int:
        return self._settings.phaseConstraint.value

    def setPhaseConstraint(self, value: int) -> None:
        self._settings.phaseConstraint.value = value

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()


class ExtendedPIEReconstructor(Reconstructor):

    @property
    def name(self) -> str:
        return 'ePIE'

    @property
    def backendName(self) -> str:
        return 'PtychoPy'

    def reconstruct(self) -> int:
        return 0  # TODO


# {"epie", (PyCFunction)ptycholib_epie, METH_VARARGS|METH_KEYWORDS, epie_docstring},
# {"epienp", (PyCFunction)ptycholib_epienp, METH_VARARGS|METH_KEYWORDS, epie_docstring},
# {"epiecmdstr", (PyCFunction)ptycholib_epiecmdstr, METH_VARARGS|METH_KEYWORDS, epiecmdstr_docstring},
# {"epienpinit", (PyCFunction)ptycholib_epienpinit, METH_VARARGS|METH_KEYWORDS, epieinit_docstring},
# # ptychopy.epiecmdstr(epiestr)
# ptychopy.epieinit(simstr)
# # BEGIN REPEAT
# ptychopy.epiestep()
# reconstructed_object = ptychopy.epieresobj()
# reconstructed_probe = ptychopy.epieresprobe()
# # END REPEAT
# ptychopy.epiepost()


class DifferenceMapReconstructor(Reconstructor):

    @property
    def name(self) -> str:
        return 'DM'

    @property
    def backendName(self) -> str:
        return 'PtychoPy'

    def reconstruct(self) -> int:
        return 0  # TODO


# {"dm", (PyCFunction)ptycholib_dm, METH_VARARGS|METH_KEYWORDS, dm_docstring},
# {"dmnp", (PyCFunction)ptycholib_dmnp, METH_VARARGS|METH_KEYWORDS, dm_docstring},
# {"dmcmdstr", (PyCFunction)ptycholib_dmcmdstr, METH_VARARGS|METH_KEYWORDS, dmcmdstr_docstring},


class LeastSquaresMaximumLikelihoodReconstructor(Reconstructor):

    @property
    def name(self) -> str:
        return 'MLs'

    @property
    def backendName(self) -> str:
        return 'PtychoPy'

    def reconstruct(self) -> int:
        return 0  # TODO


# {"mls", (PyCFunction)ptycholib_mls, METH_VARARGS|METH_KEYWORDS, mls_docstring},
# {"mlsnp", (PyCFunction)ptycholib_mlsnp, METH_VARARGS|METH_KEYWORDS, mls_docstring},
# {"mlscmdstr", (PyCFunction)ptycholib_mlscmdstr, METH_VARARGS|METH_KEYWORDS, mlscmdstr_docstring},
# # ptychopy.mlscmdstr(mlsstr)
# ptychopy.mlsinit(mlsstr)
# # BEGIN REPEAT
# ptychopy.mlsstep()
# reconstructed_object = ptychopy.mlsresobj()
# reconstructed_probe = ptychopy.mlsresprobe()
# # END REPEAT
# ptychopy.mlspost()

# TODO diffractionNP
# TODO objectNP
# TODO probeNP


class PtychoPyBackend:

    def __init__(self, settingsRegistry: SettingsRegistry) -> None:
        self._settings = PtychoPySettings.createInstance(settingsRegistry)
        self.presenter = PtychoPyPresenter.createInstance(self._settings)
        self.reconstructorList: list[Reconstructor] = list()

    @classmethod
    def createInstance(cls,
                       settingsRegistry: SettingsRegistry,
                       isDeveloperModeEnabled: bool = False) -> PtychoPyBackend:
        core = cls(settingsRegistry)

        if ptychopy or isDeveloperModeEnabled:
            core.reconstructorList.append(ExtendedPIEReconstructor())
            core.reconstructorList.append(DifferenceMapReconstructor())
            core.reconstructorList.append(LeastSquaresMaximumLikelihoodReconstructor())
        else:
            logger.info('ptychopy not found.')

        return core
