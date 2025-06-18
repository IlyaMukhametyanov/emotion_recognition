import concurrent
import os
import pickle
from itertools import combinations

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt
from sklearn.neighbors import kneighbors_graph
from torch.nn import Linear, LayerNorm, MultiheadAttention
from torch.optim.lr_scheduler import OneCycleLR
from torch_geometric.data import Data
from torch_geometric.graphgym import GATConv
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import numpy as np
from torch_geometric.nn import GATConv as OldGATConv
from torch_geometric.nn import BatchNorm, global_mean_pool
from torch.nn import Linear, ReLU
import torch.nn.functional as F


NUM_SUBJECTS = 32
NUM_TRIALS = 40
NUM_EEG_CHANNELS = 32
SEGMENT_LENGTH = 768
BATCH_SIZE = 512
EPOCHS = 200



# ================== 1. Функции для обработки сигнала ==================
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band', output='sos')


def bandpass_filter(signal, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
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

    return np.array(features).T  # shape: [32, 5]


# ================== 2. Загрузка и обработка данных ==================

fs = 128


def create_fully_connected_edge_index(num_nodes=32):
    edges = list(combinations(range(num_nodes), 2))  # все пары
    edges += [(j, i) for (i, j) in edges]  # двусторонние связи
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def process_subject(subject_id, data_dir, fs, SEGMENT_LENGTH, NUM_TRIALS, compute_de_features):
    file_path = os.path.join(data_dir, f"s{subject_id:02d}.dat")
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    eeg_data = data['data'][:, :32, :]  # trials x channels x time
    labels = data['labels'][:, :1]  # берем первую метку, например валентность
    binary_labels = (labels > np.median(labels, axis=0)).astype(int)
    binary_labels = binary_labels.ravel()  # делаем одномерным

    all_segments = []
    all_de_features = []
    all_labels = []

    shift = 384  # шаг в 3 секунды

    for trial in range(NUM_TRIALS):
        trial_data = eeg_data[trial]

        total_length = trial_data.shape[1]
        num_segments = (total_length - SEGMENT_LENGTH) // shift + 1

        segments = [
            trial_data[:, i * shift: i * shift + SEGMENT_LENGTH]
            for i in range(num_segments)
        ]

        de_features = [compute_de_features(seg, fs) for seg in segments]

        all_segments.extend(segments)
        all_de_features.extend(de_features)
        all_labels.extend([int(binary_labels[trial])] * num_segments)

    return all_segments, all_de_features, all_labels


def load_and_process_data(subject_ids, data_dir="data", fs=128, use_cache=True, verbose=True):
    cache_path = os.path.join(data_dir, f"gan_v3{fs}.npz")

    if use_cache and os.path.exists(cache_path):
        data = np.load(cache_path)
        return data['X_eeg'], data['X_de'], data['y']

    all_segments = []
    all_de_features = []
    all_labels = []

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_subject,
                subject_id, data_dir, fs, SEGMENT_LENGTH, NUM_TRIALS, compute_de_features
            )
            for subject_id in tqdm(subject_ids, desc="Подготовка субъектов", disable=not verbose)
        ]

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), disable=not verbose):
            segs, feats, lbls = future.result()
            all_segments.extend(segs)
            all_de_features.extend(feats)
            all_labels.extend(lbls)

    X_eeg = np.array(all_segments)
    X_de = np.array(all_de_features)
    y = np.array(all_labels)

    # Нормализация DE-признаков
    scaler = StandardScaler()
    X_de_normalized = scaler.fit_transform(X_de.reshape(-1, 5)).reshape(X_de.shape)

    if use_cache:
        np.savez(cache_path, X_eeg=X_eeg, X_de=X_de_normalized, y=y)

    return X_eeg, X_de_normalized, y


# ================== 3. Создание edge_index по топологии 10-20 ==================
edge_pairs = [
    # Горизонтальные связи
    [0, 1], [2, 3], [4, 5], [6, 7], [8, 9],
    [10, 11], [12, 13], [14, 15], [16, 17], [18, 19],
    [20, 21], [22, 23], [24, 25], [26, 27], [28, 29], [30, 31],

    # Вертикальные связи
    [0, 2], [1, 3],
    [2, 4], [3, 5],
    [4, 6], [5, 7],
    [6, 8], [7, 9],
    [8, 10], [9, 11],
    [10, 12], [11, 13],
    [12, 14], [13, 15],
    [14, 16], [15, 17],
    [16, 18], [17, 19],
    [18, 20], [19, 21],
    [20, 22], [21, 23],
    [22, 24], [23, 25],
    [24, 26], [25, 27],
    [26, 28], [27, 29],
    [28, 30], [29, 31],
]

edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
def custom_knn_edge_index(x, k=4):
    """
    x: tensor [num_nodes, num_features]
    Возвращает edge_index в формате PyG
    """
    x_np = x.numpy()
    adj_matrix = kneighbors_graph(x_np, n_neighbors=k, mode='connectivity', include_self=False)
    rows, cols = adj_matrix.nonzero()
    edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
    return edge_index

# ================== 4. Преобразование в PyG Dataset ==================
def create_knn_dataset(X_de, y, k=4):
    """
    Создаёт датасет, где для каждого графа свой edge_index (на основе KNN по признакам)
    """
    dataset = []
    for i in range(len(X_de)):
        x = torch.tensor(X_de[i], dtype=torch.float)  # [32, 5]

        # строим рёбра на основе признакового пространства
        edge_index = custom_knn_edge_index(x, k=k)

        # если нужно двусторонние связи (bidirectional)
        # edge_index = to_undirected(edge_index)

        label = torch.tensor(int(y[i].item()), dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=label)
        dataset.append(data)

    return dataset


# ================== 5. Модель GCN ==================
class DAMGCN(torch.nn.Module):
    def __init__(self, num_features=5, hidden_dim=256, num_classes=2):
        super().__init__()

        # --- Feature Extraction ---
        self.input_proj = Linear(num_features, hidden_dim)

        # --- Graph Convolution Block ---
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)

        # --- Dual Attention Block ---
        self.channel_attn = MultiheadAttention(embed_dim=hidden_dim, num_heads=4)
        self.freq_attn = MultiheadAttention(embed_dim=hidden_dim, num_heads=4)

        self.ln1 = LayerNorm(hidden_dim)
        self.ln2 = LayerNorm(hidden_dim)

        # --- Classifier Block ---
        self.classifier = torch.nn.Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            torch.nn.GELU(),
            Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # --- Feature Extraction ---
        x = self.input_proj(x)  # shape: [32, hidden_dim]

        # --- Graph Convolution Block ---
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.gelu(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.gelu(x)

        # --- Dual Attention Block ---

        # 1. Electrode Channel Attention
        # Нужно изменить форму под [seq_len, batch_size, embed_dim]
        num_nodes = 32
        batch_size = x.size(0) // num_nodes  # например, 64
        x = x.view(num_nodes, -1, x.size(-1))  # [32, B, C]

        # Channel-wise attention
        x_ch, _ = self.channel_attn(x, x, x)  # [32, B, C]
        x = x + x_ch  # добавляем residual
        x = self.ln1(x)

        # 2. Frequency Band Attention
        x = x.permute(1, 0, 2)  # [B, 32, C] → меняем местами seq и batch
        x_freq, _ = self.freq_attn(x, x, x)
        x = x + x_freq  # residual
        x = self.ln2(x)

        # --- Global pooling + classification ---
        x = x.mean(dim=1)  # усредняем по каналам
        out = self.classifier(x)

        return F.log_softmax(out, dim=1)

def print_gradient_norms(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    print(f"Gradient norm: {total_norm:.4f}")

# ================== 6. Загрузка данных и обучение ==================
if __name__ == "__main__":
    subject_ids = [i for i in range(1, NUM_SUBJECTS + 1) if i != 22]
    data_dir = "data"

    print("Загрузка и обработка данных...")
    X_eeg, X_de, y = load_and_process_data(subject_ids, data_dir=data_dir)

    # 🔍 Проверка: форма данных
    print("\nФорма X_de:", X_de.shape)  # должно быть (N, 32, 5)
    print("Форма y:   ", y.shape)  # должно быть (N, )

    # 🔍 Проверка: первые метки
    print("\nПервые 100 меток:")
    print(y[:100])


    print("\nУникальные метки:", np.unique(y))
    print("Баланс классов:", dict(Counter(y)))

    # 🔍 Проверка: среднее и std по признакам
    print("\nПроверка нормализации признаков:")
    mean_per_feature = X_de.mean(axis=(0, 1))  # среднее по всем узлам и графам
    std_per_feature = X_de.std(axis=(0, 1))  # std по всем узлам и графам

    print("Среднее по признакам:", mean_per_feature)
    print("Std по признакам:   ", std_per_feature)

    # Если значения далеки от 0 и 1 — значит, нужно нормализовать
    if np.any(np.abs(mean_per_feature) > 0.3) or np.any(std_per_feature < 0.7) or np.any(std_per_feature > 1.3):
        print("⚠️ Признаки не нормализованы. Рекомендуется использовать StandardScaler.")

    print("\nСоздание датасета...")
    dataset = create_knn_dataset(X_de, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DAMGCN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=3e-3,  # верхний предел
        total_steps=EPOCHS * len(loader),  # общее число шагов
        pct_start=0.3,  # процент эпох на разогрев
        div_factor=10,  # делитель для начального LR
        final_div_factor=100  # делитель для конечного LR
    )

    print("Обучение модели...")
    model.train()
    for epoch in range(200):
        model.train()
        total_loss = 0
        correct = 0
        total_samples = 0  # для страховки — считаем количество графов вручную

        for data in tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)

            loss = F.nll_loss(out, data.y)
            loss.backward()

            # Градиентное ограничение (clipping) — опционально, но полезно
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            # Накопление метрик
            total_loss += loss.item()  # <-- только среднее по батчу
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
            total_samples += data.y.size(0)

        avg_loss = total_loss / len(loader)  # усреднение по количеству батчей
        accuracy = correct / total_samples  # точность по всем примерам
        print(f"Epoch {epoch + 1} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")