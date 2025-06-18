import numpy as np
from scipy.signal import welch


def bandpower(data, sf, band, window_sec=2):
    """
    Вычисляет мощность сигнала в заданном частотном диапазоне.

    :param data: np.array, сигнал одного канала
    :param sf: частота дискретизации
    :param band: tuple (low_freq, high_freq)
    :param window_sec: длительность окна в секундах для welch
    """
    low, high = band
    nperseg = int(window_sec * sf)

    freqs, psd = welch(data, sf, nperseg=nperseg)
    freq_res = freqs[1] - freqs[0]

    idx_band = np.logical_and(freqs >= low, freqs <= high)
    band_power = np.trapz(psd[idx_band], dx=freq_res)
    return band_power


def calculate_meditation_level(data, sf=250):
    """
    Рассчитывает уровень медитации на основе соотношения мощностей в диапазонах тета, альфа и бета.

    :param data: np.array формы (каналы, буфер)
    :param sf: частота дискретизации
    :return: float, уровень медитации от 0 до 1
    """
    theta_band = (4, 8)
    alpha_band = (8, 13)
    beta_band = (13, 30)

    theta_power = 0
    alpha_power = 0
    beta_power = 0

    # Усредняем по всем каналам
    for channel in data:
        theta_power += bandpower(channel, sf, theta_band)
        alpha_power += bandpower(channel, sf, alpha_band)
        beta_power += bandpower(channel, sf, beta_band)

    theta_power /= data.shape[0]
    alpha_power /= data.shape[0]
    beta_power /= data.shape[0]

    # Уровень медитации — доля мощности тета + альфа относительно всех
    total = theta_power + alpha_power + beta_power + 1e-6  # избегаем деления на 0
    meditation_ratio = (theta_power + alpha_power) / total

    # Нормируем от 0 до 1
    return np.clip(meditation_ratio, 0, 1)
