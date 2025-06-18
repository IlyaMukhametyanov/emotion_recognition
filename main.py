from kan import *
import logging
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel
import numpy as np
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import OneCycleLR
import pickle
from sklearn.preprocessing import StandardScaler
from torch.optim.swa_utils import update_bn
from datetime import datetime

from models.cdcn_kan import CDCN_KAN_Enhanced, CDCN_KAN, HybridCDCN, CDCN_KAN2
from models.leg import CDCN

logging.basicConfig(level=logging.INFO)
from torch.utils.data import DataLoader, TensorDataset



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DEAPDataset:
    def __init__(self, data_path="data/s{subject:02d}.dat"):
        self.data_path = data_path
        self.all_eeg = []
        self.all_labels = []
        self.label_names = ['Valence', 'Arousal', 'Dominance', 'Liking']

    def load_data(self):
        """Загружает данные DEAP без метки Liking."""
        for subject in range(1, 32):

            file_path = self.data_path.format(subject=subject)
            logging.info(f"Загрузка файла {file_path}...")
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f, encoding='latin1')

                eeg_data = data['data'][:, :32, :]  # Оставляем только первые 32 канала (ЭЭГ)
                labels = data['labels'][:, :3]  # Оставляем только Valence, Arousal, Dominance

                self.all_eeg.append(eeg_data)
                self.all_labels.append(labels)
            except Exception as e:
                logging.exception(f"Ошибка при загрузке {file_path}: {e}")
                return

        self.all_eeg = np.vstack(self.all_eeg)
        self.all_labels = np.vstack(self.all_labels)

        # Применяем стандартизацию по каждому каналу отдельно
        self.standardize_eeg()
        self.split_data()
        #self.augment_data()

    def standardize_eeg(self):
        """Применяет стандартизацию к ЭЭГ данным."""
        num_samples, num_channels, num_timepoints = self.all_eeg.shape
        self.all_eeg = self.all_eeg.reshape(num_samples, num_channels * num_timepoints)
        scaler = StandardScaler()
        self.all_eeg = scaler.fit_transform(self.all_eeg)
        self.all_eeg = self.all_eeg.reshape(num_samples, num_channels, num_timepoints)

        # Разделение Valence, Arousal, Dominance на высокий (>=5) и низкий (<5) уровни
        valence_high = self.all_labels[:, 0] >= 5
        arousal_high = self.all_labels[:, 1] >= 5
        dominance_high = self.all_labels[:, 2] >= 5

        # Кодирование 8 классов
        self.multi_labels =arousal_high.astype(int) * 1

        # (valence_high.astype(int) * 2 +
        #  arousal_high.astype(int) * 1)
        logging.info("Стандартизация данных ЭЭГ и категоризация меток (8 классов) завершена.")

    def split_data(self):
        """Разделяет данные на train (85%), val (7.5%), test (7.5%)."""
        X_train, X_temp, y_train, y_temp = train_test_split(self.all_eeg, self.multi_labels, test_size=0.15,
                                                            random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test

        logging.info(f"Данные разделены: Train={len(self.X_train)}, Val={len(self.X_val)}, Test={len(self.X_test)}")
        logging.info(f"Размерность данных: Train={self.X_train.shape}, Val={self.X_val.shape}, Test={self.X_test.shape}")

    def augment_data(self):
        """Аугментирует только обучающую выборку, увеличивая её в 5 раз."""
        original_train_eeg = self.X_train
        original_train_labels = self.y_train

        augmented_eeg = []  # Оригинальные обучающие данные
        augmented_labels = []

        for _ in range(1):  # 2 дубликатов с разной аугментацией
            noisy_eeg = np.copy(original_train_eeg)
            noise_level = np.random.uniform(0.02, 0.1)  # Разные уровни шума
            noise = np.random.normal(0, noise_level, noisy_eeg.shape)
            noisy_eeg += noise

            shift = np.random.randint(1, 15)  # Разные сдвиги по времени
            noisy_eeg = np.roll(noisy_eeg, shift, axis=2)

            augmented_eeg.append(noisy_eeg)
            augmented_labels.append(original_train_labels)

        # Объединяем аугментированные данные
        self.X_train = np.concatenate([self.X_train, *augmented_eeg], axis=0)
        self.y_train = np.concatenate([self.y_train, *augmented_labels], axis=0)

        logging.info(f"Аугментация завершена. Обучающий датасет увеличен в 2 раза. Новое число образцов: {len(self.X_train)}")

    def analyze_data(self):
        """Анализирует загруженные данные."""
        num_samples, num_channels, num_timepoints = self.all_eeg.shape
        logging.info(f"Размерность данных ЭЭГ: {self.all_eeg.shape}")
        logging.info(f"Количество участников: 32 (s32 пропущен)")
        logging.info(f"Количество записей: {num_samples}")
        logging.info(f"Частота дискретизации: 128 Гц")

        logging.info("Пример меток (первые 5 строк):")
        for row in self.all_labels[:5]:
            logging.info(f"{row}")

        for i, name in enumerate(self.label_names):
            logging.info(
                f"{name}: среднее={np.mean(self.all_labels[:, i]):.2f}, std={np.std(self.all_labels[:, i]):.2f}")

        self.binary_labels = (self.all_labels > 5).astype(int)
        for i, name in enumerate(self.label_names):
            unique, counts = np.unique(self.binary_labels[:, i], return_counts=True)
            class_distribution = dict(zip(unique, counts))
            logging.info(f"{name} - Бинарное распределение классов: {class_distribution}")

        std_per_channel = np.std(self.all_eeg, axis=(0, 2))
        logging.info("Стандартное отклонение по каналам:")
        for i, std in enumerate(std_per_channel, 1):
            logging.info(f"Канал {i}: {std:.4f}")

        logging.info("Пример данных по 5 строк для каждого испытуемого:")
        for subject in range(1, 33):
            if subject == 32:
                continue
            logging.info(f"Испытуемый {subject}:")
            logging.info(f"{self.all_eeg[(subject - 1) * 40: (subject - 1) * 40 + 5]}")

        logging.info("Анализ завершён.")

    def run_analysis(self):
        """Запускает анализ датасета."""
        self.load_data()
        if self.all_eeg is not None and len(self.all_eeg) > 0 and self.all_labels is not None and len(
                self.all_labels) > 0:
            self.analyze_data()
        else:
            logging.error("Данные не загружены!")

    def analyze_dataset_2(self):
        """
        Анализирует размерности и статистику по данным ЭЭГ и меткам.

        Parameters:
        - eeg_data: np.ndarray, форма [samples, channels, time_points]
        - labels: np.ndarray, форма [samples, 3] (valence, arousal, dominance)
        """
        eeg_data = self.all_eeg
        labels = self.all_labels
        logging.info(f"Размерность данных ЭЭГ: {eeg_data.shape}")
        num_samples, num_channels, num_time_points = eeg_data.shape

        logging.info(f"Общее количество записей: {num_samples}")
        logging.info(f"Количество каналов ЭЭГ: {num_channels}")
        logging.info(f"Длина записи (в точках): {num_time_points}")

        # Частота дискретизации DEAP — 128 Гц
        duration_sec = num_time_points / 128
        logging.info(f"Длительность одной записи: {duration_sec:.2f} сек")

        logging.info(f"Размерность меток: {labels.shape}")
        logging.info(f"Пример меток (первые 5):")
        for i in range(min(5, len(labels))):
            logging.info(f"{labels[i]}")

        for i, name in enumerate(["Valence", "Arousal", "Dominance"]):
            mean = np.mean(labels[:, i])
            std = np.std(labels[:, i])
            logging.info(f"{name}: среднее={mean:.2f}, std={std:.2f}")





def train_model(model, dataset, epochs=100, batch_size=28, learning_rate=0.0001):
    """
    Функция обучения модели EEGNet.
    :param model: Нейросетевая модель EEGNet
    :param dataset: Объект DEAPDataset с загруженными и подготовленными данными
    :param epochs: Количество эпох обучения
    :param batch_size: Размер батча
    :param learning_rate: Скорость обучения
    :param device: Устройство ('cuda' или 'cpu')
    """
    # writer = get_writer(model.__class__.__name__)  # Создаём writer для этой модели
    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    # writer = SummaryWriter('runs/eeg_experiment')  # Инициализация
    # Если устройство не передано или 'cuda' недоступен — использовать 'cpu'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Переводим модель на выбранное устройство
    model.to(device)
    swa_model = AveragedModel(model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate ,weight_decay=1e-4)

    # Создание DataLoader для train, val и test
    train_dataset = TensorDataset(torch.tensor(dataset.X_train, dtype=torch.float32).unsqueeze(1),
                                  torch.tensor(dataset.y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(dataset.X_val, dtype=torch.float32).unsqueeze(1),
                                torch.tensor(dataset.y_val, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(dataset.X_test, dtype=torch.float32).unsqueeze(1),
                                 torch.tensor(dataset.y_test, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    active_start = 0
    active_count = 1
    early_stop_training = False
    best_val_accuracy = 0.0
    log_interval = 10  # Логировать каждые 10 батчей

    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    # 3. OneCycleLR scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr = 3e-3,  # Пиковый LR (подбирается экспериментально)
        total_steps=total_steps,
        pct_start=0.3,  # 30% эпох на разогрев
        anneal_strategy='cos',  # Плавное уменьшение
        div_factor=25,  # Начальный LR = max_lr / 25 (0.01 / 25 = 0.0004)
        final_div_factor=1e3  # Финальный LR = max_lr / 1e4 (0.01 / 1e4 = 1e-6)
    )


    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0


        for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc="Training"), 1):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            _, predicted = torch.max(outputs, 1)
            running_loss += loss.item() * labels.size(0)
            correct += (predicted == labels).sum().item()



        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / len(train_loader.dataset)
        # Логируем метрики


        # Валидация
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total

        # writer.add_scalar('Loss/train', train_loss, epoch)
        # writer.add_scalar('Loss/val', val_loss, epoch)
        # writer.add_scalar('Accuracy/train', train_acc, epoch)
        # writer.add_scalar('Accuracy/val', val_acc, epoch)

        print(f"Epoch {epoch + 1}/{epochs},  LR: {scheduler.get_last_lr()[0]:.6f}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")


        # Обновляем усреднённую модель SWA
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            swa_model.update_parameters(model)
            print(f"SWA updated at epoch {epoch}, val_accuracy={val_acc:.2f}%")

        # if epoch % 5 == 0:
        #     for name, param in model.named_parameters():
        #         writer.add_histogram(f'weights/{name}', param, epoch)


    # writer.close()  # В конце обучения
    # Оценка на тестовом наборе
    update_bn(train_loader, swa_model.to(device), device=device)  # Переместите SWA-модель на GPU
    # Создаем исправленный state_dict
    swa_state_dict = swa_model.state_dict()
    fixed_state_dict = {k.replace('module.', ''): v for k, v in swa_state_dict.items() if k.startswith('module.')}
    # Загружаем исправленные параметры
    model.load_state_dict(fixed_state_dict, strict=False)  # strict=False для игнорирования несовпадающих ключей
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = test_correct / test_total
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    # Сохраняем веса модели с датой в имени файла
    torch.save(model.state_dict(), f'save_models/model_weights_{current_date}.pth')
    return model

def main():
    """
    Главная функция, загружает датасет и запускает обучение модели EEGNet.
    """
    print(torch.cuda.is_available())  # Должно быть True
    print(torch.cuda.device_count())  # Должно быть > 0

    dataset = DEAPDataset()
    dataset.load_data()
    dataset.analyze_dataset_2()
    model = CDCN(num_classes=2)  # Количество классов = 8
    trained_model = train_model(model, dataset)


if __name__ == "__main__":
    main()