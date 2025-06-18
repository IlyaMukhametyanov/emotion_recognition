import os
import pickle
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import butter, sosfiltfilt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from typing import List

# Параметры
NUM_SUBJECTS = 32
NUM_TRIALS = 40
NUM_EEG_CHANNELS = 32
SEGMENT_LENGTH = 672
BATCH_SIZE = 512
EPOCHS = 200
STRIDE = 3 * 128  # 3 секунды → 384 отсчёта


# Модули CDCN
class DenseLayer(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.GELU(),
            nn.Conv1d(in_channels, growth_rate, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, self.conv(x)], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, in_channels: int, num_layers: int, growth_rate: int):
        super().__init__()
        self.layers = nn.ModuleList()
        channels = in_channels
        for _ in range(num_layers):
            self.layers.append(DenseLayer(channels, growth_rate))
            channels += growth_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class TransitionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.GELU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.AvgPool1d(kernel_size=2, stride=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class MyCDCNPlusPlus(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int],  # (E, F): электроды × частотные полосы
        tasks: List[str] = ['valence', 'arousal', 'dominance'],
        growth_rate: int = 12,
        block_config: List[int] = [6, 6, 6],  # число слоёв в каждом DenseBlock
        compression_rate: float = 0.5,  # уменьшение каналов в TransitionBlock
    ):
        super().__init__()
        E, F = input_shape
        self.tasks = tasks

        # === Feature Extractor ===
        self.feature_extractor = nn.Sequential()

        # Первый сверточный слой: обрабатывает DE-признаки
        self.feature_extractor.add_module('conv1', nn.Conv1d(E, 2 * growth_rate, kernel_size=F))
        current_channels = 2 * growth_rate

        # Блоки DenseBlock + TransitionBlock
        for i, num_layers in enumerate(block_config):
            dblock = DenseBlock(current_channels, num_layers, growth_rate)
            self.feature_extractor.add_module(f'denseblock{i+1}', dblock)
            current_channels += num_layers * growth_rate

            if i != len(block_config) - 1:
                tblock = TransitionBlock(current_channels, int(current_channels * compression_rate))
                self.feature_extractor.add_module(f'transition{i+1}', tblock)
                current_channels = int(current_channels * compression_rate)

        # Глобальное пулинг для получения фичей
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # === Динамические головы ===
        in_features = current_channels  # после глобального пулинга
        for task in tasks:
            setattr(self, task, nn.Sequential(
                nn.Linear(in_features, 128),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(128, 2)  # бинарная задача: low/high
            ))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.feature_extractor(x)  # shape: [B, C, T]
        x = self.global_pool(x).squeeze(-1)  # [B, C]

        return {task: getattr(self, task)(x) for task in self.tasks}


class EEGDataset(Dataset):
    def __init__(self, X, y, tasks: List[str] = ['valence', 'arousal', 'dominance']):
        self.X = torch.FloatTensor(X)
        self.y = {
            'valence': torch.LongTensor(y[:, 0]),
            'arousal': torch.LongTensor(y[:, 1]),
            'dominance': torch.LongTensor(y[:, 2])
        }
        self.tasks = tasks

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], {task: self.y[task][idx] for task in self.tasks}

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


def load_and_process_data(subject_ids, data_dir="data", fs=128, use_cache=True, verbose=True):

    cache_path = os.path.join(data_dir, f"3.12de_features_fs{fs}.npz")

    if use_cache and os.path.exists(cache_path):
        data = np.load(cache_path)
        return data['X'], data['y']

    all_features = []
    all_labels = []

    for subject_id in tqdm(subject_ids, desc="Субъекты", disable=not verbose):
        file_path = os.path.join(data_dir, f"s{subject_id:02d}.dat")
        with open(file_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        eeg_data = data['data'][:, :32, :]  # shape [trials, channels, samples]
        labels = data['labels'][:, :3]      # shape [trials, 3]
        binary_labels = (labels > np.median(labels, axis=0)).astype(int)

        for trial in range(eeg_data.shape[0]):
            trial_data = eeg_data[trial]  # shape [32, samples]

            num_samples = trial_data.shape[1]
            # Сколько всего можно вырезать сегментов
            num_segments = (num_samples - SEGMENT_LENGTH) // STRIDE + 1

            # Формируем сегменты через цикл
            for i in range(num_segments):
                start = i * STRIDE
                end = start + SEGMENT_LENGTH
                segment = trial_data[:, start:end]  # shape [32, 768]

                # Вычисляем DE-фичи
                features = compute_de_features(segment, fs)
                all_features.append(features)
                all_labels.append(binary_labels[trial])

    X = np.array(all_features)
    y = np.array(all_labels)

    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X.reshape(-1, 5)).reshape(X.shape)

    if use_cache:
        np.savez(cache_path, X=X_normalized, y=y)

    return X_normalized, y


def train_model(model, train_loader, val_loader, optimizer, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    best_val_f1 = 0.0
    scheduler = OneCycleLR(optimizer, max_lr=3e-3, total_steps=EPOCHS * len(train_loader))

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []

        # Тренировка
        for X, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            X = X.to(device)
            y = {k: v.to(device) for k, v in y.items()}

            optimizer.zero_grad()
            outputs = model(X)
            loss = sum(criterion(outputs[k], y[k]) for k in outputs.keys())
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

            # Сбор предиктов для accuracy
            for task in outputs:
                preds = torch.argmax(outputs[task], dim=1)
                train_preds.extend(preds.cpu().numpy())
                train_targets.extend(y[task].cpu().numpy())

        train_acc = accuracy_score(train_targets, train_preds)

        # Валидация
        model.eval()
        val_metrics = {task: [] for task in model.tasks}
        val_loss = 0

        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                y_batch = {k: v.to(device) for k, v in y.items()}
                outputs = model(X)

                # Считаем loss
                val_loss += sum(criterion(outputs[task], y_batch[task]).item() for task in outputs)

                for task in outputs:
                    preds = torch.argmax(outputs[task], dim=1)
                    val_metrics[task].append({
                        'preds': preds.cpu(),
                        'targets': y_batch[task].cpu()
                    })

        # Расчёт метрик
        avg_f1 = 0
        all_val_preds = []
        all_val_targets = []

        for task in val_metrics:
            all_preds = torch.cat([x['preds'] for x in val_metrics[task]])
            all_targets = torch.cat([x['targets'] for x in val_metrics[task]])
            f1 = f1_score(all_targets, all_preds, average='macro')
            acc = accuracy_score(all_targets, all_preds)
            avg_f1 += f1
            all_val_preds.extend(all_preds.numpy())
            all_val_targets.extend(all_targets.numpy())
            print(f"{task} F1: {f1:.4f} | Acc: {acc:.4f}")

        avg_f1 /= len(val_metrics)
        val_acc = accuracy_score(all_val_targets, all_val_preds)

        if avg_f1 > best_val_f1:
            best_val_f1 = avg_f1
            torch.save(model.state_dict(), "../best_model.pth")

        current_lr = optimizer.param_groups[0]['lr']

        print(f"Train/val  Loss: {train_loss / len(train_loader):.4f}|{val_loss / len(val_loader):.4f}  "
              f"Acc: {train_acc:.4f}|{val_acc:.4f}  "
              f"Learning Rate: {current_lr:.10f}")

    return model

def evaluate(model, test_loader, device):
    model.eval()
    metrics = {}

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            outputs = model(X)

            for task in outputs:
                if task not in metrics:
                    metrics[task] = {'preds': [], 'targets': []}

                preds = torch.argmax(outputs[task], dim=1)
                metrics[task]['preds'].extend(preds.cpu().numpy())
                metrics[task]['targets'].extend(y[task].numpy())

    results = {}
    for task in metrics:
        acc = accuracy_score(metrics[task]['targets'], metrics[task]['preds'])
        f1 = f1_score(metrics[task]['targets'], metrics[task]['preds'], average='macro')
        results[task] = {'Accuracy': acc, 'F1': f1}

    return results


#'arousal',
def main(tasks: List[str] = ['valence',  'dominance']):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Загрузка данных
    X, y = load_and_process_data(range(1, NUM_SUBJECTS + 1))
    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    # Создание Dataset
    train_dataset = EEGDataset(X_train, y_train, tasks=tasks)
    val_dataset = EEGDataset(X_val, y_val, tasks=tasks)
    test_dataset = EEGDataset(X_test, y_test, tasks=tasks)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Инициализация модели
    model = MyCDCNPlusPlus(input_shape=(32, 5), tasks=tasks)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Обучение
    trained_model = train_model(model, train_loader, val_loader, optimizer, device)
    torch.save(trained_model.state_dict(), f'save_models/model_weights_{current_date}.pth')
    # Тестирование
    results = evaluate(trained_model, test_loader, device)
    for task in results:
        print(f"{task}:")
        print(f"  Accuracy: {results[task]['Accuracy']:.4f}")
        print(f"  F1 Score: {results[task]['F1']:.4f}")
    print("-------------------")

if __name__ == "__main__":
    main()