from ...api.scan import Scan, ScanPoint
from .builder import ScanBuilder


class CartesianScanBuilder(ScanBuilder):

    def __init__(self, *, snake: bool, centered: bool) -> None:
        super().__init__('Cartesian')  # FIXME hex scan too
        nameList: list[str] = list()

        if centered:
            nameList.append('Centered')

        if snake:
            nameList.append('Snake')
        else:
            nameList.append('Raster')

        super().__init__(' '.join(nameList))
        self.stepSizeXInMeters = self._registerRealParameter('StepSizeXInMeters', 1e-6, minimum=0.)
        self.stepSizeYInMeters = self._registerRealParameter('StepSizeYInMeters', 1e-6, minimum=0.)
        self.numberOfPointsX = self._registerIntegerParameter('NumberOfPointsX', 10, minimum=0)
        self.numberOfPointsY = self._registerIntegerParameter('NumberOfPointsY', 10, minimum=0)
        self._snake = self._registerBooleanParameter('Snake', snake)
        self._centered = self._registerBooleanParameter('Centered', centered)

    def build(self) -> Scan:
        nx = self.numberOfPointsX.getValue()
        ny = self.numberOfPointsY.getValue()
        dx = self.stepSizeXInMeters.getValue()
        dy = self.stepSizeYInMeters.getValue()
        pointList: list[ScanPoint] = list()

        for index in range(nx * ny):
            y, x = divmod(index, nx)

            if self._snake:
                if y & 1:
                    x = nx - 1 - x

            cx = (nx - 1) / 2
            cy = (ny - 1) / 2

            xf = (x - cx) * dx
            yf = (y - cy) * dy

            if self._centered:
                if y & 1:
                    xf += dx / 4
                else:
                    xf -= dx / 4

            point = ScanPoint(
                index=index,
                positionXInMeters=xf,
                positionYInMeters=yf,
            )
            pointList.append(point)

        return Scan(pointList)
