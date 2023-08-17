from .active import ActiveReconstructor


class ReconstructorAPI:

    def __init__(self, activeReconstructor: ActiveReconstructor) -> None:
        super().__init__()
        self._activeReconstructor = activeReconstructor

    def ingest(self) -> None:
        self._activeReconstructor.ingest()

    def train(self) -> None:
        self._activeReconstructor.train()

    def reset(self) -> None:
        self._activeReconstructor.reset()
