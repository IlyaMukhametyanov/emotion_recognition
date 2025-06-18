from typing import List, Dict
import torch
import torch.nn as nn
from torch.nn.modules.module import Module
import os
import pickle
from datetime import datetime
from kan import KAN
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
from PyEMD.EMD import EMD
import concurrent.futures
from functools import partial

NUM_SUBJECTS = 32
NUM_TRIALS = 40
NUM_EEG_CHANNELS = 32
SEGMENT_LENGTH = 672
BATCH_SIZE = 512
EPOCHS = 200


BANDS = {
    'delta': (1, 3),
    'theta': (4, 7),
    'alpha': (8, 13),
    'beta': (14, 30),
    'gamma': (31, 50)
}
# ====================
# Базовые блоки CDCN
# ====================

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.GELU(),
            nn.Conv1d(in_channels, growth_rate, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return torch.cat([x, self.conv(x)], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, num_layers, growth_rate):
        super().__init__()
        self.layers = nn.ModuleList()
        channels = in_channels
        for _ in range(num_layers):
            self.layers.append(DenseLayer(channels, growth_rate))
            channels += growth_rate

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.GELU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(self.conv(x))


# ================
# DE-ветка
# ================

class DEBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(1, 2, 2)),
            nn.BatchNorm3d(8),
            nn.GELU(),

            nn.Conv3d(8, 16, kernel_size=(1, 2, 2)),
            nn.BatchNorm3d(16),
            nn.GELU(),

            nn.Flatten(start_dim=2),
            nn.AdaptiveAvgPool2d((32, 5)),
            nn.Flatten(start_dim=1),
            nn.Linear(32*5, 64),
            nn.GELU()
        )

    def forward(self, x):
        """
        x.shape == (B, 32, 3, 5)
        """
        B = x.size(0)
        x = x.unsqueeze(1)  # (B, 1, 32, 3, 5)
        return self.net(x)


# ================
# Механизм внимания
# ================

class AttentionFusion(nn.Module):
    def __init__(self, in_dim_eeg=240, in_dim_de=64, hidden_dim=128):
        super().__init__()
        self.fuse = nn.Linear(in_dim_eeg + in_dim_de, hidden_dim)
        self.attn = nn.Linear(hidden_dim, 2)  # веса для EEG и DE
        self.softmax = nn.Softmax(dim=1)

    def forward(self, eeg_out, de_out):
        combined = torch.cat([eeg_out, de_out], dim=1)  # (B, 304)
        weights = self.attn(self.fuse(combined))         # (B, 2)
        weights = self.softmax(weights)                  # нормализация

        # Применяем веса к исходным векторам
        weighted_eeg = weights[:, 0].unsqueeze(1) * eeg_out  # (B, 240)
        weighted_de = weights[:, 1].unsqueeze(1) * de_out    # (B, 64)

        # Возвращаем **взвешенно объединённый вектор** размером 304
        fused = torch.cat([weighted_eeg, weighted_de], dim=1)  # (B, 304)
        return fused


# ================
# Основная модель
# ================

class DualInputModel(nn.Module):
    def __init__(self, tasks: List[str] = ['valence', 'arousal', 'dominance']):
        super().__init__()
        self.tasks = tasks

        # --- EEG ветка ---
        self.eeg_branch = nn.Sequential(
            # Вход: (B, 32, 384)
            nn.Conv1d(32, 24, kernel_size=5, padding=2),
            DenseBlock(24, 6, 12),
            TransitionBlock(96, 96),
            DenseBlock(96, 6, 12),
            TransitionBlock(168, 168),
            DenseBlock(168, 6, 12),
            nn.AdaptiveAvgPool1d(1)  # (B, 240)
        )

        # --- DE ветка ---
        self.de_branch = DEBranch()  # принимает (B, 32, 3, 5)

        # --- Attention-based fusion ---
        self.fusion = AttentionFusion()

        # --- Головы задач ---
        for task in tasks:
            setattr(self, task, nn.Sequential(
                nn.Linear(304, 128),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(128, 2)
            ))

    def forward(self, eeg: torch.Tensor, de: torch.Tensor) -> Dict[str, torch.Tensor]:
        # EEG обработка
        eeg = self.eeg_branch(eeg).squeeze(-1)  # shape: [B, 240]

        # DE обработка
        de = self.de_branch(de)  # shape: [B, 64]

        # Объединение с помощью attention
        fused = self.fusion(eeg, de)  # shape: [B, 304]

        # Предикт по задачам
        return {task: getattr(self, task)(fused) for task in self.tasks}

from torch.utils.data import Dataset


class DualInputDataset(Dataset):
    def __init__(self, X_eeg, X_de, y, tasks=['valence', 'arousal', 'dominance']):
        self.X_eeg = torch.FloatTensor(X_eeg)
        self.X_de = torch.FloatTensor(X_de)
        self.y = {
            'valence': torch.LongTensor(y[:, 0]),
            'arousal': torch.LongTensor(y[:, 1]),
            'dominance': torch.LongTensor(y[:, 2])
        }
        self.tasks = tasks

    def __len__(self):
        return len(self.X_eeg)

    def __getitem__(self, idx):
        return (
            self.X_eeg[idx],
            self.X_de[idx],
            {task: self.y[task][idx] for task in self.tasks}
        )

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


def compute_memd_imfs(eeg_segment, num_imfs=3):
    imfs_list = []
    for channel in eeg_segment:
        emd = EMD()
        imfs = emd(channel)
        # Берём только первые num_imfs
        imfs_list.append(imfs[:num_imfs])
    return np.array(imfs_list)  # shape: (32, num_imfs, time_steps)


def extract_de_from_imfs(imfs):
    """
    Вычисляет DE на каждом IMF и частотном диапазоне
    imfs: (32, num_imfs, T)
    returns: (32, num_imfs, 5)
    """
    features_by_channel = []

    for ch in range(imfs.shape[0]):
        channel_imfs = imfs[ch]  # (num_imfs, T)
        de_per_imf = []

        for imf_idx in range(channel_imfs.shape[0]):
            imf = channel_imfs[imf_idx]
            de_values = []
            for band_name, (low, high) in BANDS.items():
                filtered = bandpass_filter(imf, low, high, 128)
                de = compute_de(filtered[np.newaxis, :])  # (1, 32)
                de_values.append(de[0])
            de_per_imf.append(de_values)  # (имена IMF x полосы)
        features_by_channel.append(de_per_imf)  # (каналы x IMF x полосы)

    return np.array(features_by_channel)  # (32, num_imfs, 5)



def process_trial(seg, compute_memd_imfs, extract_de_from_imfs):
    imfs = compute_memd_imfs(seg)
    de_features = extract_de_from_imfs(imfs)
    return seg, de_features


def process_subject(subject_id, data_dir, fs, SEGMENT_LENGTH, compute_memd_imfs, extract_de_from_imfs, verbose=True):
    file_path = os.path.join(data_dir, f"s{subject_id:02d}.dat")
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    eeg_data = data['data'][:, :32, :]   # (trials, channels, time)
    labels = data['labels'][:, :3]
    binary_labels = (labels > np.median(labels, axis=0)).astype(int)

    all_segments = []
    all_memd_features = []
    all_labels = []

    for trial_idx in range(len(eeg_data)):
        trial_data = eeg_data[trial_idx]  # (32, time_steps)

        num_segments = trial_data.shape[1] // SEGMENT_LENGTH
        segments = np.split(
            trial_data[:, :num_segments * SEGMENT_LENGTH],
            indices_or_sections=num_segments,
            axis=1
        )

        # Обработка всех сегментов этого trial через EMD → DE
        results = [
            extract_de_from_imfs(compute_memd_imfs(seg, 3))
            for seg in segments
        ]

        all_segments.extend(segments)
        all_memd_features.extend(results)
        all_labels.extend([binary_labels[trial_idx]] * len(segments))

    return all_segments, all_memd_features, all_labels


def load_and_process_data(subject_ids, data_dir="data", fs=128, use_cache=True, verbose=True):
    cache_path = os.path.join(data_dir, f"memd_based_features_fs{fs}.npz")

    if use_cache and os.path.exists(cache_path):
        data = np.load(cache_path)
        return data['X_eeg'], data['X_de'], data['y']

    SEGMENT_LENGTH = fs * 3  # 3 секунды
    all_segments = []
    all_memd_features = []
    all_labels = []

    # --- Правильное использование ProcessPoolExecutor ---
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_subject, subject_id, data_dir, fs, SEGMENT_LENGTH, compute_memd_imfs, extract_de_from_imfs, verbose
            )
            for subject_id in tqdm(subject_ids, desc="Подготовка субъектов", disable=not verbose)
        ]

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), disable=not verbose):
            segs, feats, lbs = future.result()
            all_segments.extend(segs)
            all_memd_features.extend(feats)
            all_labels.extend(lbs)

    X_eeg = np.array(all_segments)
    X_de = np.array(all_memd_features)
    y = np.array(all_labels)

    # Нормализация DE-признаков
    B, C, I, F = X_de.shape
    X_de_flat = X_de.reshape(-1, F)
    scaler = StandardScaler()
    X_de_normalized = scaler.fit_transform(X_de_flat).reshape(B, C, I, F)

    if use_cache:
        np.savez_compressed(cache_path, X_eeg=X_eeg, X_de=X_de_normalized, y=y)

    return X_eeg, X_de_normalized, y


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

        for eeg_batch, de_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            eeg_batch = eeg_batch.to(device)
            de_batch = de_batch.to(device)
            y_batch = {k: v.to(device) for k, v in y_batch.items()}

            optimizer.zero_grad()
            outputs = model(eeg_batch, de_batch)
            loss = sum(criterion(outputs[k], y_batch[k]) for k in outputs.keys())
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

            for task in outputs:
                preds = torch.argmax(outputs[task], dim=1)
                train_preds.extend(preds.cpu().numpy())
                train_targets.extend(y_batch[task].cpu().numpy())

        train_acc = accuracy_score(train_targets, train_preds)

        # --- Валидация ---
        model.eval()
        val_metrics = {task: [] for task in model.tasks}
        val_loss = 0

        with torch.no_grad():
            for eeg_batch, de_batch, y_batch in val_loader:
                eeg_batch = eeg_batch.to(device)
                de_batch = de_batch.to(device)
                y_true = {k: v.to(device) for k, v in y_batch.items()}
                outputs = model(eeg_batch, de_batch)

                val_loss += sum(criterion(outputs[k], y_true[k]).item() for k in outputs)

                for task in outputs:
                    preds = torch.argmax(outputs[task], dim=1)
                    val_metrics[task].append({
                        'preds': preds.cpu(),
                        'targets': y_true[task].cpu()
                    })

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
        for eeg_batch, de_batch, y_batch in test_loader:
            eeg_batch = eeg_batch.to(device)
            de_batch = de_batch.to(device)
            outputs = model(eeg_batch, de_batch)

            for task in outputs:
                if task not in metrics:
                    metrics[task] = {'preds': [], 'targets': []}

                preds = torch.argmax(outputs[task], dim=1)
                metrics[task]['preds'].extend(preds.cpu().numpy())
                metrics[task]['targets'].extend(y_batch[task].numpy())

    results = {}
    for task in metrics:
        acc = accuracy_score(metrics[task]['targets'], metrics[task]['preds'])
        f1 = f1_score(metrics[task]['targets'], metrics[task]['preds'], average='macro')
        results[task] = {'Accuracy': acc, 'F1': f1}

    return results


#'arousal','dominance'
def main(tasks: List[str] = ['valence' ]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")


    subject_ids = [i for i in range(1, NUM_SUBJECTS + 1) if i != 22]
    X_eeg, X_de, y = load_and_process_data(subject_ids)

    # Разделение
    X_train_eeg, X_test_eeg, X_train_de, X_test_de, y_train, y_test = train_test_split(
        X_eeg, X_de, y, test_size=0.2, random_state=42
    )
    X_val_eeg, X_test_eeg, X_val_de, X_test_de, y_val, y_test = train_test_split(
        X_test_eeg, X_test_de, y_test, test_size=0.5, random_state=42
    )

    # Dataset
    train_dataset = DualInputDataset(X_train_eeg, X_train_de, y_train, tasks=tasks)
    val_dataset = DualInputDataset(X_val_eeg, X_val_de, y_val, tasks=tasks)
    test_dataset = DualInputDataset(X_test_eeg, X_test_de, y_test, tasks=tasks)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = DualInputModel(tasks=tasks)
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