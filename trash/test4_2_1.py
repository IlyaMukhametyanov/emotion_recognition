import os
import pickle
from datetime import datetime
from scipy.signal import butter, filtfilt, sosfiltfilt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn, SWALR
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# Параметры
NUM_SUBJECTS = 32
NUM_TRIALS = 40
NUM_EEG_CHANNELS = 32
SEGMENT_LENGTH = 448
BATCH_SIZE = 640
EPOCHS = 10000


class ValenceHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.mlp(x)

class EEGArtifactFilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter_net = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.25),  # Добавлен Dropout
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.Dropout(0.25))  # Добавлен Dropout

    def forward(self, x):
        return x * self.filter_net(x)

class ValenceEEGModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.artifact_filter = EEGArtifactFilter()
        self.conv1 = nn.Conv1d(32, 24, kernel_size=5, padding=2)
        self.dropout1 = nn.Dropout(0.25)  # После начальной свертки
        self.dense1 = self._make_dense_block(24, 6, 12)
        self.trans1 = TransitionBlock(96, 96)
        self.dropout2 = nn.Dropout(0.3)  # После первого перехода
        self.dense2 = self._make_dense_block(96, 6, 12)
        self.trans2 = TransitionBlock(168, 168)
        self.dropout3 = nn.Dropout(0.3)  # После второго перехода
        self.dense3 = self._make_dense_block(168, 6, 12)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout4 = nn.Dropout(0.4)  # Перед финальной головой
        self.valence = ValenceHead(240)

    def _make_dense_block(self, in_channels, num_layers, growth_rate):
        layers = []
        channels = in_channels
        for _ in range(num_layers):
            layers.append(DenseLayer(channels, growth_rate))
            channels += growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.artifact_filter(x)
        x = self.conv1(x)
        x = self.dropout1(x)
        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dropout2(x)
        x = self.dense2(x)
        x = self.trans2(x)
        x = self.dropout3(x)
        x = self.dense3(x)
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout4(x)
        return self.valence(x)

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.GELU(),
            nn.Dropout(0.2),  # Добавлен Dropout в плотный блок
            nn.Conv1d(in_channels, growth_rate, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return torch.cat([x, self.conv(x)], 1)

class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.GELU(),
            nn.Dropout(0.25),  # Добавлен Dropout в переходный блок
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(self.conv(x))


# Фильтрация через SOS (более стабильная реализация)
def bandpass_filter(signal, lowcut, highcut, fs, order=5):
    """Бандпас-фильтр Баттерворта с защитой от граничных эффектов"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sosfiltfilt(sos, signal)


def compute_de(signal):
    """Вычисление дифференциальной энтропии с защитой от нулевой дисперсии"""
    variance = np.var(signal, axis=1, ddof=1)
    return 0.5 * np.log(2 * np.pi * np.e * (variance + 1e-8))  # Добавление ε к дисперсии


def compute_de_features(eeg_segment, fs=128):
    """Улучшенное вычисление DE-признаков с проверкой размерностей"""
    # Проверка входных данных
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
        # Фильтрация каждого канала
        filtered = np.array([
            bandpass_filter(channel, band[0], band[1], fs)
            for channel in eeg_segment
        ])

        # Вычисление DE
        de = compute_de(filtered)
        features.append(de)

    return np.array(features).T  # Форма [32, 5]


class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_and_process_data(subject_ids, data_dir="data", fs=128, use_cache=True, verbose=True):
    cache_path = os.path.join(data_dir, f"2de_features_fs{fs}.npz")

    if use_cache and os.path.exists(cache_path):
        if verbose:
            print(f"Загрузка из кэша: {cache_path}")
        data = np.load(cache_path)
        return data['X'], data['y']

    # Параметры с описаниями для прогресс-бара
    SEGMENT_LENGTH = fs * 3
    NUM_TRIALS = 15
    total_subjects = len(subject_ids)

    all_features = []
    all_labels = []

    # Прогресс-бар для обработки субъектов
    subject_iter = tqdm(subject_ids, desc="Субъекты", disable=not verbose)
    for subject_id in subject_iter:
        subject_iter.set_postfix_str(f"ID: {subject_id}")

        file_path = os.path.join(data_dir, f"s{subject_id:02d}.dat")
        with open(file_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        eeg_data = data['data'][:, :32, :]
        labels = data['labels'][:, 0]
        binary_labels = (labels > np.median(labels)).astype(int)

        # Прогресс-бар для обработки проб
        trial_iter = tqdm(range(NUM_TRIALS), desc="Пробы", leave=False, disable=not verbose)
        for trial in trial_iter:
            trial_iter.set_postfix_str(f"Trial: {trial + 1}/{NUM_TRIALS}")

            trial_data = eeg_data[trial]
            num_segments = trial_data.shape[1] // SEGMENT_LENGTH
            segments = np.split(
                trial_data[:, :num_segments * SEGMENT_LENGTH],
                num_segments,
                axis=1
            )

            # Прогресс-бар для сегментов
            segment_iter = tqdm(segments, desc="Сегменты", leave=False, disable=not verbose)
            for seg in segment_iter:
                de_features = compute_de_features(seg, fs)
                all_features.append(de_features)
                all_labels.append(binary_labels[trial])

    # Нормализация и сохранение в кэш
    X = np.array(all_features)
    y = np.array(all_labels)

    if verbose:
        print("Нормализация данных...")
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X.reshape(-1, 5)).reshape(X.shape)

    if use_cache:
        if verbose:
            print(f"Сохранение в кэш: {cache_path}")
        np.savez(cache_path, X=X_normalized, y=y)

    return X_normalized, y

def train_model(model, train_loader, val_loader, optimizer, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    # swa_model = AveragedModel(model).to(device)
    best_val_f1 = 0.0
    #scheduler = OneCycleLR(optimizer, max_lr=3e-3, total_steps=EPOCHS * len(train_loader))

    scheduler = OneCycleLR(
        optimizer,
        max_lr = 6e-4,
        total_steps=EPOCHS * len(train_loader),
        pct_start=0.45,
        three_phase=True,  # Трехфазный цикл
        anneal_strategy='cos',
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.90,
        div_factor=30,
        final_div_factor=1e3
    )
    # swa_start = int(EPOCHS * 0.75)
    # swa_scheduler = SWALR(optimizer, swa_lr=0.05)
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for X, y in progress_bar:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            # Сбор метрик обучения
            train_loss += loss.item() * X.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == y).sum().item()
            train_total += y.size(0)
            # Обновление прогресс-бара
            progress_bar.set_postfix({
                'train_loss': f"{train_loss / train_total:.4f}",
                'train_acc': f"{train_correct / train_total:.4f}"
            })

        # Расчет метрик обучения
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        # Валидация
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)

                val_loss += loss.item() * X.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        # Расчет метрик валидации
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        # Обновление SWA
        # if epoch >= swa_start:
        #     swa_model.update_parameters(model)
        #     swa_scheduler.step()

        # Вывод метрик
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(f"Train/val  Loss: {train_loss:.4f}|{val_loss:.4f}  Acc: {train_acc:.4f}|{val_acc:.4f} Learning Rate: {scheduler.get_last_lr()[0]:.10f}")

    #update_bn(train_loader, swa_model, device=device)
    return model


def evaluate(model, test_loader, device):
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            outputs = model(X)
            preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            targets.extend(y.numpy())

    return {
        'Accuracy': accuracy_score(targets, preds),
        'F1': f1_score(targets, preds, average='macro')
    }


def main():
    print(f'Начало загрузки')
    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y = load_and_process_data(range(1, NUM_SUBJECTS + 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    train_dataset = EEGDataset(X_train, y_train)
    val_dataset = EEGDataset(X_val, y_val)
    test_dataset = EEGDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = ValenceEEGModel()
    optimizer = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

    trained_model = train_model(model, train_loader, val_loader, optimizer, device)
    results = evaluate(trained_model, test_loader, device)

    torch.save(trained_model.state_dict(), f'save_models/model_weights_{current_date}.pth')

    print("\nFinal Test Results:")
    print(f"Accuracy: {results['Accuracy']:.4f}")
    print(f"F1 Score: {results['F1']:.4f}")


if __name__ == "__main__":
    main()