import os
import pickle
from datetime import datetime
from functools import partial
from multiprocessing import Pool

from kan import KAN
from scipy.signal import butter, sosfiltfilt
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
BATCH_SIZE = 64  # Уменьшено для стабильности
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class KANConv1d(nn.Module):
    """Сплайновая 1D свертка с 3D-адаптацией"""

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.kan = KAN(width=[out_channels, out_channels], grid=3)

    def forward(self, x):
        # x: [batch, in_channels, time]
        conv_out = self.conv(x)  # [batch, out_channels, time]

        # Вычисление KAN-коэффициентов
        kan_weights = self.kan(conv_out.mean(dim=-1))  # [batch, out_channels]

        return conv_out * kan_weights.unsqueeze(-1)  # [batch, out_channels, time]


class KANActivation(nn.Module):
    def __init__(self, units):
        super().__init__()
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x)


class CrossAttentionFusion(nn.Module):
    def __init__(self, mlp_dim, kan_dim):
        super().__init__()
        self.query = nn.Linear(mlp_dim, mlp_dim)
        self.key = nn.Linear(kan_dim, mlp_dim)
        self.value = nn.Linear(kan_dim, mlp_dim)

    def forward(self, mlp_feat, kan_feat):
        Q = self.query(mlp_feat)
        K = self.key(kan_feat)
        V = self.value(kan_feat)
        attn = torch.softmax(Q @ K.transpose(-2, -1) / (mlp_feat.size(-1) ** 0.5), dim=-1)
        return attn @ V


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            KANActivation(in_channels),
            KANConv1d(in_channels, growth_rate, 3)
        )

    def forward(self, x):
        return torch.cat([x, self.conv(x)], 1)


class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm1d(in_channels),  # Теперь принимает актуальное число каналов
            KANActivation(in_channels),
            KANConv1d(in_channels, out_channels, 1)
        )
        self.pool = nn.AdaptiveMaxPool1d(int(out_channels * 0.5))

    def forward(self, x):
        return self.pool(self.conv(x))


class MultiTaskEEG(nn.Module):
    def __init__(self):
        super().__init__()
        super().__init__()
        # Основной CNN ствол (исправлены размерности)
        self.conv1 = nn.Conv1d(32, 24, kernel_size=5, padding=2)
        self.dense1 = self._make_dense_block(24, 6, 12)  # 24 → 96 каналов
        self.trans1 = TransitionBlock(128, 64)  # 96 + 32 = 128 вход

        # Обработка DE-признаков
        self.de_processor = nn.Sequential(
            nn.Linear(32 * 5, 128),
            nn.GELU(),
            nn.Linear(128, 32)  # Соответствует размеру для конкатенации
        )

        # Механизм внимания
        self.cross_attn = CrossAttentionFusion(mlp_dim=96, kan_dim=32)

        # Последующие блоки (исправлены размерности)
        self.dense2 = self._make_dense_block(128, 6, 12)  # 128 → 200 каналов
        self.trans2 = TransitionBlock(200, 100)
        self.dense3 = self._make_dense_block(100, 6, 12)  # 100 → 172 канала

        # Головы
        self.valence_head = self._make_head(172)
        self.arousal_head = self._make_head(172)
        self.dominance_head = self._make_head(172)


    def _make_dense_block(self, in_channels, num_layers, growth_rate):
        layers = []
        for _ in range(num_layers):
            layers.append(DenseLayer(in_channels, growth_rate))
            in_channels += growth_rate
        return nn.Sequential(*layers)

    def _make_head(self, in_dim):
        return nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2)
        )

    def forward(self, raw_eeg, de_features):
        # Проверка размерностей
        print(f"Input shape: {raw_eeg.shape}")  # [B,32,448]

        # Основной поток
        x = self.conv1(raw_eeg)  # [B,24,448]
        x = self.dense1(x)  # [B,96,448]

        # Обработка DE-признаков
        de_flat = de_features.view(de_features.size(0), -1)  # [B,160]
        kan_branch = self.de_processor(de_flat)  # [B,32]

        # Механизм внимания
        mlp_feat = x.mean(dim=-1)  # [B,96]
        attn = self.cross_attn(mlp_feat, kan_branch)  # [B,32]
        x = torch.cat([
            x,
            attn.unsqueeze(-1).expand(-1, -1, x.size(-1))
        ], dim=1)  # [B,128,448]

        x = self.trans1(x)  # [B,64,224]
        x = self.dense2(x)  # [B,200,224]
        x = self.trans2(x)  # [B,100,112]
        x = self.dense3(x)  # [B,172,112]

        # Финализация
        x = x.mean(dim=-1)  # [B,172]
        return {
            'valence': self.valence_head(x),
            'arousal': self.arousal_head(x),
            'dominance': self.dominance_head(x)
        }


# Обработка данных (остаётся без изменений)
def bandpass_filter(signal, lowcut, highcut, fs, order=5):
    sos = butter(order, [lowcut, highcut], fs=fs, btype='band', output='sos')
    return sosfiltfilt(sos, signal)


def compute_de(signal):
    return 0.5 * np.log(2 * np.pi * np.e * (np.var(signal, axis=1) + 1e-8))


def compute_de_features(eeg_segment, fs=128):
    bands = {'delta': (1, 3), 'theta': (4, 7), 'alpha': (8, 13), 'beta': (14, 30), 'gamma': (31, 50)}
    return np.array([
        compute_de(np.array([bandpass_filter(ch, *band, fs) for ch in eeg_segment]))
        for band in bands.values()
    ]).T


class EEGDataset(Dataset):
    def __init__(self, raw, de, labels):
        self.raw = torch.FloatTensor(raw)
        self.de = torch.FloatTensor(de)
        self.labels = {k: torch.LongTensor(labels[:, i]) for i, k in enumerate(['valence', 'arousal', 'dominance'])}

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, idx):
        return {'raw': self.raw[idx], 'de': self.de[idx]}, {k: v[idx] for k, v in self.labels.items()}


def hybrid_collate(batch):
    inputs = {'raw': torch.stack([x[0]['raw'] for x in batch]),
              'de': torch.stack([x[0]['de'] for x in batch])}
    labels = {k: torch.stack([x[1][k] for x in batch]) for k in batch[0][1].keys()}
    return inputs, labels


# Обучение и оценка (с исправлениями)
def train_model(model, train_loader, val_loader):
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)

    best_f1 = 0
    for epoch in range(EPOCHS):
        model.train()
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            raw = inputs['raw'].to(DEVICE)
            de = inputs['de'].to(DEVICE)
            y = {k: v.to(DEVICE) for k, v in labels.items()}

            optimizer.zero_grad()
            outputs = model(raw, de)
            loss = sum(criterion(outputs[k], y[k]) for k in outputs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Валидация
        model.eval()
        all_preds = {k: [] for k in ['valence', 'arousal', 'dominance']}
        all_targets = {k: [] for k in ['valence', 'arousal', 'dominance']}

        with torch.no_grad():
            for inputs, labels in val_loader:
                raw = inputs['raw'].to(DEVICE)
                de = inputs['de'].to(DEVICE)
                outputs = model(raw, de)

                for task in outputs:
                    all_preds[task].append(torch.argmax(outputs[task], 1).cpu())
                    all_targets[task].append(labels[task].cpu())

        # Расчёт метрик
        avg_f1 = 0
        for task in all_preds:
            preds = torch.cat(all_preds[task]).numpy()
            targets = torch.cat(all_targets[task]).numpy()
            f1 = f1_score(targets, preds, average='macro')
            avg_f1 += f1
            print(f"{task:9} F1: {f1:.4f}")

        avg_f1 /= 3
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            torch.save(model.state_dict(), "../best_model.pth")
            print(f"New best model saved (F1: {avg_f1:.4f})")





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



def process_subject(subject_id, data_dir, verbose):
    file_path = os.path.join(data_dir, f"s{subject_id:02d}.dat")
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    eeg_data = data['data'][:, :32, :]  # [trials, channels, time]
    labels = data['labels'][:, :3]
    binary_labels = (labels > np.median(labels, axis=0)).astype(int)

    subject_raw = []  # Для хранения сырых сигналов
    subject_de = []  # Для DE-признаков
    subject_labels = []

    for trial in range(NUM_TRIALS):
        trial_data = eeg_data[trial]  # [32, 8064]
        num_segments = trial_data.shape[1] // SEGMENT_LENGTH

        # Сегментация без потерь
        segmented = trial_data[:, :num_segments * SEGMENT_LENGTH].reshape(
            32, num_segments, SEGMENT_LENGTH
        ).transpose(1, 0, 2)  # [num_segments, 32, SEGMENT_LENGTH]

        for seg in segmented:
            # Сохраняем сырой сигнал (нормализованный)
            raw_norm = (seg - np.mean(seg, axis=1, keepdims=True)) / (np.std(seg, axis=1, keepdims=True) + 1e-8)
            subject_raw.append(raw_norm)

            # Вычисляем DE-признаки
            subject_de.append(compute_de_features(seg, fs=128))

        subject_labels.extend([binary_labels[trial]] * num_segments)

    return (
        np.array(subject_raw),
        np.array(subject_de),
        np.array(subject_labels)
    )

def load_and_process_data(subject_ids, data_dir="data", fs=128, use_cache=True, verbose=True):
    cache_path = os.path.join(data_dir, f"hybrid_features_fs{fs}.npz")

    if use_cache and os.path.exists(cache_path):
        data = np.load(cache_path)
        return data['raw'], data['de'], data['y']

    with Pool(processes=os.cpu_count()) as pool:
        results = list(tqdm(
            pool.imap(partial(process_subject, data_dir=data_dir, verbose=verbose), subject_ids),
            total=len(subject_ids),
            desc="Субъекты",
            disable=not verbose
        ))
    # Агрегация данных
    all_raw, all_de, all_labels = [], [], []
    for raw, de, labels in results:
        all_raw.extend(raw)
        all_de.extend(de)
        all_labels.extend(labels)

    # Конвертация в numpy
    X_raw = np.array(all_raw, dtype=np.float32)
    X_de = np.array(all_de, dtype=np.float32)
    y = np.array(all_labels)

    # Нормализация DE-признаков
    de_scaler = StandardScaler()
    X_de = de_scaler.fit_transform(X_de.reshape(-1, 5)).reshape(X_de.shape)

    if use_cache:
        np.savez(cache_path, raw=X_raw, de=X_de, y=y)

    return X_raw, X_de, y


def main():
    # Загрузка данных
    X_raw, X_de, y = load_and_process_data(range(1, NUM_SUBJECTS + 1))  # Ваша реализация

    # Разделение данных
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, stratify=y[:, 0], random_state=42)
    val_idx, test_idx = train_test_split(test_idx, test_size=0.5, stratify=y[test_idx, 0], random_state=42)

    # Датасеты
    train_data = EEGDataset(X_raw[train_idx], X_de[train_idx], y[train_idx])
    val_data = EEGDataset(X_raw[val_idx], X_de[val_idx], y[val_idx])
    test_data = EEGDataset(X_raw[test_idx], X_de[test_idx], y[test_idx])

    # DataLoader
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=hybrid_collate)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, collate_fn=hybrid_collate)

    # Обучение
    model = MultiTaskEEG()
    train_model(model, train_loader, val_loader)


if __name__ == "__main__":
    main()


