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
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# Параметры
NUM_SUBJECTS = 32
NUM_TRIALS = 40
NUM_EEG_CHANNELS = 32
SEGMENT_LENGTH = 448
BATCH_SIZE = 640
EPOCHS = 10000


class EEGArtifactFilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter_net = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.Dropout(0.25))

    def forward(self, x):
        return x * self.filter_net(x)


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.GELU(),
            nn.Dropout(0.2),
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
            nn.Dropout(0.25),
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(self.conv(x))


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


class ArousalHead(nn.Module):
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


class MultiTaskEEG(nn.Module):
    def __init__(self):
        super().__init__()
        self.artifact_filter = EEGArtifactFilter()
        self.conv1 = nn.Conv1d(32, 24, kernel_size=5, padding=2)
        self.dropout1 = nn.Dropout(0.25)
        self.dense1 = self._make_dense_block(24, 6, 12)
        self.trans1 = TransitionBlock(96, 96)
        self.dropout2 = nn.Dropout(0.3)
        self.dense2 = self._make_dense_block(96, 6, 12)
        self.trans2 = TransitionBlock(168, 168)
        self.dropout3 = nn.Dropout(0.3)
        self.dense3 = self._make_dense_block(168, 6, 12)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout4 = nn.Dropout(0.4)
        self.valence = ValenceHead(240)
        self.arousal = ArousalHead(240)

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
        return {
            'valence': self.valence(x),
            'arousal': self.arousal(x)
        }


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


class EEGDataset(Dataset):
    def __init__(self, X, y):
        # Основное исправление: преобразование меток в словарь тензоров
        self.X = torch.FloatTensor(X)
        self.y = {
            'valence': torch.LongTensor(y[:, 0]),  # Берем первый столбец
            'arousal': torch.LongTensor(y[:, 1])  # Берем второй столбец
        }

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], {
            'valence': self.y['valence'][idx],
            'arousal': self.y['arousal'][idx]
        }


def load_and_process_data(subject_ids, data_dir="data", fs=128, use_cache=True, verbose=True):
    cache_path = os.path.join(data_dir, f"3de_features_fs{fs}.npz")

    if use_cache and os.path.exists(cache_path):
        data = np.load(cache_path)
        return data['X'], data['y']

    SEGMENT_LENGTH = fs * 3
    NUM_TRIALS = 15
    all_features = []
    all_labels = []

    for subject_id in tqdm(subject_ids, desc="Субъекты", disable=not verbose):
        file_path = os.path.join(data_dir, f"s{subject_id:02d}.dat")
        with open(file_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        eeg_data = data['data'][:, :32, :]
        labels = data['labels'][:, :2]  # Берем оба признака
        binary_labels = (labels > np.median(labels, axis=0)).astype(int)  # Исправлено для каждого признака

        for trial in range(NUM_TRIALS):
            trial_data = eeg_data[trial]
            num_segments = trial_data.shape[1] // SEGMENT_LENGTH
            segments = np.split(
                trial_data[:, :num_segments * SEGMENT_LENGTH],
                num_segments,
                axis=1
            )
            # Исправлено: добавляем метки для каждого сегмента
            all_features.extend([compute_de_features(seg, fs) for seg in segments])
            all_labels.extend([binary_labels[trial]] * num_segments)

    X = np.array(all_features)
    y = np.array(all_labels)

    # Проверка формы меток
    if y.ndim != 2 or y.shape[1] != 2:
        raise ValueError(f"Некорректная форма меток: {y.shape}. Ожидается (N, 2)")

    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X.reshape(-1, 5)).reshape(X.shape)

    if use_cache:
        np.savez(cache_path, X=X_normalized, y=y)

    return X_normalized, y


def train_model(model, train_loader, val_loader, optimizer, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    scheduler = OneCycleLR(
        optimizer,
        max_lr=6e-4,
        total_steps=EPOCHS * len(train_loader),
        pct_start=0.45,
        three_phase=True,
        anneal_strategy='cos',
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.90,
        div_factor=30,
        final_div_factor=1e3
    )

    best_val_f1 = 0.0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for X, y in progress_bar:
            X = X.to(device)
            y_val = y['valence'].to(device)
            y_ars = y['arousal'].to(device)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs['valence'], y_val) + criterion(outputs['arousal'], y_ars)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * X.size(0)
            train_total += y_val.size(0)
            progress_bar.set_postfix({'loss': f"{train_loss / train_total:.4f}"})

        # Валидация
        model.eval()
        val_metrics = {'valence': [], 'arousal': []}
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                outputs = model(X)
                for task in outputs:
                    preds = torch.argmax(outputs[task], dim=1).cpu()
                    val_metrics[task].append({
                        'preds': preds,
                        'targets': y[task].cpu()
                    })

        avg_f1 = 0
        for task in val_metrics:
            all_preds = torch.cat([x['preds'] for x in val_metrics[task]]).numpy()
            all_targets = torch.cat([x['targets'] for x in val_metrics[task]]).numpy()
            f1 = f1_score(all_targets, all_preds, average='macro')
            avg_f1 += f1
            print(f"{task} F1: {f1:.4f}")
        avg_f1 /= 2
        print(f"Epoch {epoch + 1} | Val F1: {avg_f1:.4f}")

    return model


def evaluate(model, test_loader, device):
    model.eval()
    metrics = {'valence': {'preds': [], 'targets': []},
               'arousal': {'preds': [], 'targets': []}}

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            outputs = model(X)
            for task in outputs:
                preds = torch.argmax(outputs[task], dim=1).cpu().numpy()
                metrics[task]['preds'].extend(preds)
                metrics[task]['targets'].extend(y[task].numpy())

    results = {}
    for task in metrics:
        results[task] = {
            'Accuracy': accuracy_score(metrics[task]['targets'], metrics[task]['preds']),
            'F1': f1_score(metrics[task]['targets'], metrics[task]['preds'], average='macro')
        }
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")

    X, y = load_and_process_data(range(1, NUM_SUBJECTS + 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    train_dataset = EEGDataset(X_train, y_train)
    val_dataset = EEGDataset(X_val, y_val)
    test_dataset = EEGDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = MultiTaskEEG()
    optimizer = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

    trained_model = train_model(model, train_loader, val_loader, optimizer, device)
    results = evaluate(trained_model, test_loader, device)

    torch.save(trained_model.state_dict(), f'save_models/model_weights_{current_date}.pth')

    print("\nFinal Test Results:")
    for task in results:
        print(f"{task}:")
        print(f"  Accuracy: {results[task]['Accuracy']:.4f}")
        print(f"  F1 Score: {results[task]['F1']:.4f}")


if __name__ == "__main__":
    main()