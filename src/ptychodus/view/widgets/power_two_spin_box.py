from PyQt5.QtGui import QValidator
from PyQt5.QtWidgets import QSpinBox, QWidget


class PowerTwoSpinBox(QSpinBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

    def stepBy(self, steps: int) -> None:  # noqa: N802
        if steps < 0:
            self.setValue(self.value() // (1 << -steps))
        elif steps > 0:
            self.setValue(self.value() * (1 << steps))

    def validate(self, input: str, pos: int) -> tuple[QValidator.State, str, int]:
        try:
            value = int(input)
        except ValueError:
            pass
        else:
            if value > 0:
                is_pow2 = (value & (value - 1)) == 0

                if is_pow2:
                    return QValidator.State.Acceptable, input, pos

        return QValidator.State.Intermediate, input, pos
