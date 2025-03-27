try:
    from PyQt5.QtWidgets import QWidget
except ModuleNotFoundError:
    FOUND_QT = False
else:
    FOUND_QT = True


class QtTester:
    def print_qt_status(self) -> None:
        if FOUND_QT:
            print('Qt FOUND!')
        else:
            print('Qt MISSING!')

    if FOUND_QT:

        def get_widget(self) -> QWidget:
            return QWidget()


if __name__ == '__main__':
    tester = QtTester()
    tester.print_qt_status()
    widget = tester.get_widget()
    print(widget)
