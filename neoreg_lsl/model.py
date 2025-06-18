import torch
import numpy as np
from scipy.signal import butter, sosfiltfilt
from cdcn_hslt import R2G_CDCN_Hybrid  # замени 'your_model_file' на имя файла с моделью

# === Функции предобработки сигнала ===
def bandpass_filter(signal, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sosfiltfilt(sos, signal)

def compute_de(signal):
    variance = np.var(signal, axis=1, ddof=1)
    return 0.5 * np.log(2 * np.pi * np.e * (variance + 1e-8))

def compute_de_features(eeg_segment, fs=128):
    assert eeg_segment.ndim == 2, "Ожидается 2D массив [каналы, время]"
    assert eeg_segment.shape[0] == 32, f"Ожидается 32 канала, получено {eeg_segment.shape[0]}"

    bands = {
        'delta': (1, 3),
        'theta': (4, 7),
        'alpha': (8, 13),
        'beta': (14, 30),
        'gamma': (31, 50)
    }

    features = []
    for band in bands.values():
        filtered = np.array([
            bandpass_filter(channel, band[0], band[1], fs)
            for channel in eeg_segment
        ])
        de = compute_de(filtered)
        features.append(de)

    return np.array(features).T

# === Загрузка модели ===
try:
    model = R2G_CDCN_Hybrid(input_shape=(32, 5), tasks=["valence", "dominance"])
    model.load_state_dict(torch.load("model.pth"))  # или путь к модели
    model.eval()
except Exception as e:
    print("❌ Ошибка загрузки модели:", e)
    model = None

# === Предсказание настроения ===

import numpy as np
from scipy.interpolate import interp1d


def interpolate_channels(eeg_segment, target_channels=32):
    """
    Интерполирует сигнал ЭЭГ с текущего числа каналов до target_channels.

    :param eeg_segment: np.array, shape [current_channels, time]
    :param target_channels: int, желаемое число каналов (например, 32)
    :return: np.array, shape [target_channels, time]
    """
    current_channels = eeg_segment.shape[0]

    if current_channels == target_channels:
        return eeg_segment

    print(f"🔄 Интерполируем с {current_channels} до {target_channels} каналов...")

    # Создаём координаты для исходного и целевого пространства
    x_old = np.linspace(0, 1, current_channels)
    x_new = np.linspace(0, 1, target_channels)

    # Результирующий массив
    interpolated_data = np.zeros((target_channels, eeg_segment.shape[1]))

    for i in range(eeg_segment.shape[1]):
        f = interp1d(x_old, eeg_segment[:, i], kind='linear', fill_value="extrapolate")
        interpolated_data[:, i] = f(x_new)

    return interpolated_data
def predict_mood(eeg_segment):
    if model is None:
        return "модель не загружена"
    try:
        # Проверяем размерность
        if eeg_segment.shape[0] != 32:
            print(f"⚠️ Обнаружено {eeg_segment.shape[0]} каналов — выполняем интерполяцию до 32...")
            eeg_segment = interpolate_channels(eeg_segment, target_channels=32)

        features = compute_de_features(eeg_segment)
        features = torch.tensor(features[np.newaxis], dtype=torch.float32)  # [1, E, F]

        with torch.no_grad():
            outputs = model(features)

        valence = outputs["valence"].argmax().item()
        arousal = outputs["dominance"].argmax().item()

        mood_map = {
            (0, 0): "грустное",
            (0, 1): "спокойное",
            (1, 0): "раздражённое",
            (1, 1): "весёлое"
        }

        return mood_map.get((valence, arousal), "неопределённое")
    except Exception as e:
        print("❌ Ошибка при анализе:", e)
        return "ошибка анализа"