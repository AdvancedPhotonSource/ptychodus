try:
    from PyQt5.QtWidgets import QWidget
except ModuleNotFoundError:
    FOUND_QT = False
else:
    FOUND_QT = True


class Tester:
    def printQtStatus(self) -> None:
        if FOUND_QT:
            print("Qt FOUND!")
        else:
            print("Qt MISSING!")

    if FOUND_QT:

        def getWidget(self) -> QWidget:
            return QWidget()


if __name__ == "__main__":
    tester = Tester()
    tester.printQtStatus()
    widget = tester.getWidget()
    print(widget)
