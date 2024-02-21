from ..product import ObjectRepository
from .frc import FourierRingCorrelator


class AnalysisCore:

    def __init__(self, repository: ObjectRepository) -> None:
        self.fourierRingCorrelator = FourierRingCorrelator(repository)
