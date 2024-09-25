from __future__ import annotations
import logging

from ptychodus.api.observer import Observable, Observer

from ...model.ptychopack import PtychoPackPresenter, PtychoPackReconstructorLibrary
from ...view.ptychopack import PtychoPackParametersView, PtychoPackView
from .exit_wave import PtychoPackExitWaveCorrectionController
from .object import PtychoPackObjectCorrectionController
from .position import PtychoPackPositionCorrectionController
from .probe import PtychoPackProbeCorrectionController

logger = logging.getLogger(__name__)


class PtychoPackParametersController(Observer):
    def __init__(self, presenter: PtychoPackPresenter, view: PtychoPackParametersView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

        for device in presenter.get_available_devices():
            view.device_combobox.addItem(device)

        view.device_combobox.textActivated.connect(presenter.set_device)

        self._sync_model_to_view()
        presenter.addObserver(self)

    def _sync_model_to_view(self) -> None:
        self._view.device_combobox.setCurrentText(self._presenter.get_device())
        self._view.plan_label.setText(self._presenter.get_plan())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._sync_model_to_view()


class PtychoPackController:
    def __init__(self, model: PtychoPackReconstructorLibrary, view: PtychoPackView) -> None:
        self.parameters_controller = PtychoPackParametersController(
            model.presenter, view.parameters_view
        )
        self.exit_wave_controller = PtychoPackExitWaveCorrectionController(
            model.presenter, view.exit_wave_view
        )
        self.object_controller = PtychoPackObjectCorrectionController(
            model.presenter, view.object_view
        )
        self.probe_controller = PtychoPackProbeCorrectionController(
            model.presenter, view.probe_view
        )
        self.position_controller = PtychoPackPositionCorrectionController(
            model.presenter, view.position_view
        )
