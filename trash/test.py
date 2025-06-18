import pickle
import logging
logging.basicConfig(level=logging.INFO)

data_path = "data/s{subject:02d}.dat"
for subject in range(1, 32):

    file_path = data_path.format(subject=subject)
    logging.info(f"Загрузка файла {file_path}...")
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        eeg_data = data['data'][:, :32, :]  # Оставляем только первые 32 канала (ЭЭГ)
        labels = data['labels'][:, :3]  # Оставляем только Valence, Arousal, Dominance


    except Exception as e:
        logging.exception(f"Ошибка при загрузке {file_path}: {e}")



file_path = data_path.format(subject=1)
logging.info(f"Загрузка файла {file_path}...")
try:
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    logging.info(f"Размерность данных {data['data'].shape}")
    eeg_data = data['data'][:, :32, :]  # Оставляем только первые 32 канала (ЭЭГ)
    labels = data['labels'][:, :3]  # Оставляем только Valence, Arousal, Dominance
    logging.info(f"Пример {data['labels']}")
except Exception as e:
    logging.exception(f"Ошибка при загрузке {file_path}: {e}")