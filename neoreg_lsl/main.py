from interface import EEGInterface
from PyQt5 import QtWidgets

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = EEGInterface()
    window.show()
    sys.exit(app.exec_())