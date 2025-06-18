import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
import numpy as np
from lsl_listener import connect_to_lsl
from model import predict_mood

# === Настройки интерфейса ===
CHANNEL_COUNT = 16
BUFFER_SIZE = 500
UPDATE_INTERVAL_MS = 10

class EEGInterface(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        # Инициализация интерфейса
        self.setWindowTitle("Трекер ЭЭГ")
        self.layout = QtWidgets.QVBoxLayout(self)

        # Выбор режима
        self.mode_selector = QtWidgets.QComboBox()
        self.mode_selector.addItem("Медитация")
        self.mode_selector.addItem("Настроение")
        self.mode_selector.currentIndexChanged.connect(self.switch_mode)
        self.layout.addWidget(self.mode_selector)

        # Метка для вывода информации о настроении
        self.mood_label = QtWidgets.QLabel("Настроение: Ожидание...")
        self.layout.addWidget(self.mood_label)

        # Графики ЭЭГ
        self.plot_widget = pg.GraphicsLayoutWidget(self)
        self.layout.addWidget(self.plot_widget)
        self.plots = []
        self.curves = []
        for i in range(CHANNEL_COUNT):
            p = self.plot_widget.addPlot(row=i, col=0)
            p.setYRange(-200, 200)
            p.showAxis('left', False)
            p.showAxis('bottom', False)
            curve = p.plot(pen=pg.intColor(i, hues=CHANNEL_COUNT))
            self.plots.append(p)
            self.curves.append(curve)

        self.setLayout(self.layout)

        # Подключение к LSL
        self.fs = 128
        self.buffer_duration = 6
        self.buffer_size_samples = self.fs * self.buffer_duration
        self.data_buffer = np.zeros((CHANNEL_COUNT, 0))

        self.inlet = connect_to_lsl("NBEEG16_Data")
        if not self.inlet:
            print("❌ Не удалось подключиться к потоку LSL.")
            exit()

        # Таймер обновления графиков
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(UPDATE_INTERVAL_MS)

        # Таймер анализа настроения (каждые 3 секунды)
        self.analysis_timer = QtCore.QTimer()
        self.analysis_timer.timeout.connect(self.run_analysis)
        self.analysis_timer.start(3000)  # 3000 мс = 3 секунды

    def update_plot(self):
        """Обновление графиков."""
        sample, _ = self.inlet.pull_sample(timeout=0.0)
        if sample:
            new_data = np.array(sample).reshape(-1, 1)  # [channel x 1]
            self.data_buffer = np.hstack([self.data_buffer, new_data])

            # Ограничиваем размер буфера
            if self.data_buffer.shape[1] > self.buffer_size_samples:
                self.data_buffer = self.data_buffer[:, -self.buffer_size_samples:]

            for i in range(CHANNEL_COUNT):
                if self.data_buffer.shape[1] < BUFFER_SIZE:
                    continue
                self.curves[i].setData(self.data_buffer[i, -BUFFER_SIZE:])

    def run_analysis(self):
        """Анализ данных каждые 3 секунды"""
        if self.data_buffer.shape[1] < self.fs * 6:
            print("⏳ Недостаточно данных для анализа")
            return

        segment = self.data_buffer[:32, :]  # Убедись, что у тебя 32 канала
        mood = predict_mood(segment)
        print(f"😊 Распознано настроение: {mood}")
        self.mood_label.setText(f"Настроение: {mood}")

    def switch_mode(self):
        mode = self.mode_selector.currentText()
        if mode == "Медитация":
            self.mood_label.setText("")
        elif mode == "Настроение":
            self.mood_label.setText(f"Настроение: {self.mood_label.text().split(': ')[-1]}")