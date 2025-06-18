import pandas as pd
import numpy as np
from scipy.signal import welch

# === Загрузка данных ===
df = pd.read_csv("raw_eeg_data.txt", sep="\t")

# Проверим, все ли каналы — числовые
channel_cols = [f'ch{i}' for i in range(1, 17)]
for ch in channel_cols:
    df[ch] = pd.to_numeric(df[ch], errors='coerce')  # превращаем строки в float (если были)

# Проверка масштаба — посмотрим диапазон значений для всех каналов
print(df[channel_cols].describe())

# === Подсчёт RMS активности ===
df['rms_activity_v'] = df[channel_cols].apply(lambda row: np.sqrt(np.mean(row**2)), axis=1)

# 👉 Проверка масштаба данных:
sample_val = df[channel_cols].iloc[0].abs().max()

if sample_val < 1e-3:  # Если значения меньше 1e-3 Вольта, то это Вольты
    print("📏 Обнаружены значения в Вольтах — переводим в µV")
    df['rms_activity_uv'] = df['rms_activity_v'] * 1e6  # Перевод в микровольты
else:
    print("📏 Обнаружены значения в µV — не переводим")
    df['rms_activity_uv'] = df['rms_activity_v']  # Оставляем как есть (в µV)

# Среднее значение RMS
mean_rms_uv = df['rms_activity_uv'].mean()
print(f"\n🔹 Средняя мозговая активность (RMS): {mean_rms_uv:.2f} µV")

# Интерпретация мозговой активности
if mean_rms_uv < 5:
    print("🧘 Состояние: расслабленное или сонливое")
elif mean_rms_uv < 15:
    print("🙂 Состояние: спокойное бодрствование")
else:
    print("⚡ Состояние: активное внимание или возбуждение")

# === Анализ Альфа-ритма (8–12 Гц) ===
fs = 250  # частота дискретизации
channel_for_alpha = 'ch8'  # Канал для альфа-ритма (например, затылочный)

# Получаем сигнал для одного канала
signal = df[channel_for_alpha].values

# Плотность мощности методом Welch
f, Pxx = welch(signal, fs=fs, nperseg=fs*2)

# Выделяем мощность в альфа-диапазоне
alpha_band = (f >= 8) & (f <= 12)
alpha_power = np.sum(Pxx[alpha_band])
total_power = np.sum(Pxx)

# Относительная мощность альфа-ритма
alpha_ratio = alpha_power / total_power

print(f"🔹 Альфа-мощность (канал {channel_for_alpha}): {alpha_power:.2e} V²/Hz")
print(f"🔹 Доля альфа-ритма: {alpha_ratio*100:.2f}%")

# Интерпретация альфа-ритма
if alpha_ratio > 0.3:
    print("😌 Альфа-ритм выражен — вероятно, расслабленное состояние, возможно закрытые глаза.")
else:
    print("👀 Альфа-ритм слабый — возможно, открытые глаза или активное внимание.")
