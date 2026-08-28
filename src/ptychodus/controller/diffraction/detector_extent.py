from ptychodus.api.geometry import ImageExtent
from ptychodus.api.observer import Observable


class DetectorExtentSource(Observable):
    """Controller-owned observable holder for the current detector's pixel extent.

    Purely a UI-side signalling channel for the diffraction wizard's crop/bin spin
    boxes, which need to redraw their bounds when a new dataset's extent becomes
    known. Populated by the diffraction controller in response to dataset repository
    inserts and by the wizard when it opens a new file. Not persisted; a freshly
    launched session starts with ``None`` until data supplies an extent.

    Not accessed from the model layer — dataset processing derives the extent from
    each dataset's own metadata (see ``AssembledDiffractionDataset.reload`` and
    ``build_prep_pipeline``), and product geometry derives it from the dataset
    paired via ``ProductRepositoryItem.bind_dataset`` / ``unbind_dataset``.
    """

    def __init__(self) -> None:
        super().__init__()
        self._extent: ImageExtent | None = None

    def get_extent(self) -> ImageExtent | None:
        return self._extent

    def set_extent(self, extent: ImageExtent | None) -> None:
        if extent == self._extent:
            return
        self._extent = extent
        self.notify_observers()
