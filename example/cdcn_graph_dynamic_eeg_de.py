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
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import dense_to_sparse
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform

NUM_SUBJECTS = 32
NUM_TRIALS = 40
NUM_EEG_CHANNELS = 32
SEGMENT_LENGTH = 672
BATCH_SIZE = 512
EPOCHS = 200

# ================
# Базовые блоки CDCN
# ================

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
    def __init__(self, input_shape=(32, 5), hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            # Вход: (B, 32, 5)
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.5),

            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),

            nn.Flatten(),  # (B, 64)
            nn.Linear(64, hidden_dim),
            nn.GELU()
        )

    def forward(self, x):
        return self.net(x)



class AttentionFusion(nn.Module):
    def __init__(self, in_dim_eeg=240, in_dim_gcn=64, hidden_dim=128):
        super().__init__()
        self.fuse = nn.Linear(in_dim_eeg + in_dim_gcn, hidden_dim)
        self.attn = nn.Linear(hidden_dim, 2)  # два источника: eeg и gcn
        self.softmax = nn.Softmax(dim=1)

    def forward(self, eeg_out, gcn_out):
        combined = torch.cat([eeg_out, gcn_out], dim=1)  # (B, 240 + hidden_dim)
        weights = self.attn(self.fuse(combined))
        weights = self.softmax(weights)

        weighted_eeg = weights[:, 0].unsqueeze(1) * eeg_out
        weighted_gcn = weights[:, 1].unsqueeze(1) * gcn_out
        fused = torch.cat([weighted_eeg, weighted_gcn], dim=1)
        return fused

# ================
# Основная модель
# ================

class DualInputGNNModel(nn.Module):
    def __init__(self, tasks=['valence', 'arousal'], gcn_input_dim=5, hidden_dim=64):
        super().__init__()
        self.tasks = tasks

        # --- EEG Branch ---
        self.eeg_branch = nn.Sequential(
            nn.Conv1d(32, 24, kernel_size=5, padding=2),
            DenseBlock(24, 6, 12),
            TransitionBlock(96, 96),
            DenseBlock(96, 6, 12),
            TransitionBlock(168, 168),
            DenseBlock(168, 6, 12),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        # --- GNN Branch ---
        self.gcn1 = GCNConv(gcn_input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)

        # --- Attention Fusion ---
        self.fusion = AttentionFusion(in_dim_eeg=240, in_dim_gcn=hidden_dim)

        # --- Task Heads ---
        for task in tasks:
            setattr(self, task, nn.Sequential(
                nn.Linear(240 + hidden_dim, 128),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(128, 2)
            ))

    def forward(self, eeg, data):
        """
        eeg: (B, 32, 384)
        data: Batch(x, edge_index, batch, y)
        """

        # --- GCN Branch ---
        x = data.x
        edge_index = data.edge_index
        batch_idx = data.batch

        print("x.shape", x.shape)          # должно быть (B * 32, 5)
        print("edge_index.max()", edge_index.max().item())  # должно быть < B * 32
        print("batch_idx", batch_idx.shape) # (B * 32, )

        x = F.gelu(self.gcn1(x, edge_index))
        x = F.gelu(self.gcn2(x, edge_index))

        gcn_out = global_mean_pool(x, batch_idx)  # (B, H)

        # --- EEG Branch ---
        eeg_out = self.eeg_branch(eeg)  # (B, 240)

        # --- Attention-based fusion ---
        fused = self.fusion(eeg_out, gcn_out)

        # --- Task heads ---
        return {task: getattr(self, task)(fused) for task in self.tasks}


from torch.utils.data import Dataset

class GraphEEGDataset(Dataset):
    def __init__(self, X_eeg, X_de, y, edge_index, edge_weight=None, tasks=['valence', 'arousal']):
        super().__init__()
        self.X_eeg = torch.FloatTensor(X_eeg)
        self.X_de = torch.FloatTensor(X_de)

        # Метки
        self.y = {task: torch.LongTensor(y[:, i]) for i, task in enumerate(tasks)}
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.tasks = tasks

    def __len__(self):
        return len(self.X_de)

    def __getitem__(self, idx):
        return (
            self.X_eeg[idx],
            self.X_de[idx],
            {task: self.y[task][idx] for task in self.tasks}
        )

    def collate(self, batch):
        eeg_batch = torch.stack([b[0] for b in batch])
        de_batch = torch.stack([b[1] for b in batch])
        y_batch = {task: torch.stack([b[2][task] for b in batch]) for task in self.tasks}

        B, CN, FDIM = de_batch.shape  # например, (512, 32, 5)

        data_list = []
        for i in range(B):
            data = Data(
                x=de_batch[i],                      # (32, 5)
                edge_index=self.edge_index,         # (2, E)
                edge_attr=self.edge_weight,         # (E,)
                y={task: y_batch[task][i] for task in self.tasks}  # метка для этого графа
            )
            data_list.append(data)

        batch_data = Batch.from_data_list(data_list)  # автоматически создаёт .batch
        return eeg_batch, batch_data, {k: v for k, v in y_batch.items()}

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
    cache_path = os.path.join(data_dir, f"3.7de_features_fs{fs}.npz")

    if use_cache and os.path.exists(cache_path):
        data = np.load(cache_path)
        return data['X_eeg'], data['X_de'], data['y']

    SEGMENT_LENGTH = fs * 3
    NUM_TRIALS = 15
    all_segments = []
    all_de_features = []
    all_labels = []

    for subject_id in tqdm(subject_ids, desc="Субъекты", disable=not verbose):
        file_path = os.path.join(data_dir, f"s{subject_id:02d}.dat")
        with open(file_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        eeg_data = data['data'][:, :32, :]
        labels = data['labels'][:, :3]
        binary_labels = (labels > np.median(labels, axis=0)).astype(int)

        for trial in range(NUM_TRIALS):
            trial_data = eeg_data[trial]
            num_segments = trial_data.shape[1] // SEGMENT_LENGTH
            segments = np.split(
                trial_data[:, :num_segments * SEGMENT_LENGTH],
                num_segments,
                axis=1
            )

            # Сохраняем сегменты ЭЭГ и соответствующие DE-признаки
            all_segments.extend(segments)
            all_de_features.extend([compute_de_features(seg, fs) for seg in segments])
            all_labels.extend([binary_labels[trial]] * num_segments)

    X_eeg = np.array(all_segments)
    X_de = np.array(all_de_features)
    y = np.array(all_labels)

    # Нормализация DE-признаков
    scaler = StandardScaler()
    X_de_normalized = scaler.fit_transform(X_de.reshape(-1, 5)).reshape(X_de.shape)

    if use_cache:
        np.savez(cache_path, X_eeg=X_eeg, X_de=X_de_normalized, y=y)

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

        for eeg_batch, data_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            eeg_batch = eeg_batch.to(device)
            data_batch = data_batch.to(device)
            y_batch = {k: v.to(device) for k, v in y_batch.items()}

            optimizer.zero_grad()
            outputs = model(eeg=eeg_batch, data=data_batch)

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
            for eeg_batch, data_batch, y_batch in val_loader:
                eeg_batch = eeg_batch.to(device)
                data_batch = data_batch.to(device)
                y_batch = {k: v.to(device) for k, v in y_batch.items()}

                de_x = data_batch.x
                edge_index = data_batch.edge_index
                batch_idx = data_batch.batch
                outputs = model(eeg=eeg_batch, de_x=de_x, edge_index=edge_index, batch=batch_idx)

                val_loss += sum(criterion(outputs[k], y_batch[k]).item() for k in outputs)

                for task in outputs:
                    preds = torch.argmax(outputs[task], dim=1)
                    val_metrics[task].append({
                        'preds': preds.cpu(),
                        'targets': y_batch[task].cpu()
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


def build_electrode_graph(three_d_coords):
    """
    three_d_coords: numpy array of shape (32, 3)
    returns: edge_index, edge_weight
    """
    distances = squareform(pdist(three_d_coords, metric='euclidean'))

    # Преобразуем в веса рёбер
    adj_matrix = 1 / (distances + 1e-8)
    np.fill_diagonal(adj_matrix, 0)  # убираем self-loops

    # Преобразуем в PyTorch формат
    adj_tensor = torch.tensor(adj_matrix, dtype=torch.float)

    # Получаем edge_index и edge_weight
    edge_index, edge_weight = dense_to_sparse(adj_tensor)

    return edge_index.long(), edge_weight.float()

electrode_coords = np.array([
    [-0.94, 0.34, 0.0],   # FP1
    [0.0, 1.0, 0.0],     # FPZ
    [0.94, 0.34, 0.0],   # FP2
    [-0.67, 0.74, 0.0],   # AF3
    [0.67, 0.74, 0.0],    # AF4
    [-0.92, 0.39, 0.0],   # F7
    [-0.75, 0.66, 0.0],   # F5
    [-0.5, 0.87, 0.0],    # F3
    [-0.34, 0.94, 0.0],   # F1
    [0.0, 1.0, 0.0],      # FZ
    [0.34, 0.94, 0.0],    # F2
    [0.5, 0.87, 0.0],     # F4
    [0.75, 0.66, 0.0],    # F6
    [0.92, 0.39, 0.0],     # F8
    [-0.92, 0.39, 0.0],   # FT7
    [-0.75, 0.66, 0.0],    # FC5
    [-0.5, 0.87, 0.0],     # FC3
    [-0.34, 0.94, 0.0],    # FC1
    [0.0, 1.0, 0.0],       # FCZ
    [0.34, 0.94, 0.0],     # FC2
    [0.5, 0.87, 0.0],      # FC4
    [0.75, 0.66, 0.0],      # FC6
    [0.92, 0.39, 0.0],      # FT8
    [-0.92, -0.39, 0.0],     # T7
    [-0.75, -0.66, 0.0],     # C5
    [-0.5, -0.87, 0.0],      # C3
    [-0.34, -0.94, 0.0],     # C1
    [0.0, -1.0, 0.0],        # CZ
    [0.34, -0.94, 0.0],      # C2
    [0.5, -0.87, 0.0],       # C4
    [0.75, -0.66, 0.0],      # C6
    [0.92, -0.39, 0.0],      # T8
    [-0.67, -0.74, 0.0],     # TP7
    [-0.5, -0.87, 0.0],      # CP3
    [-0.34, -0.94, 0.0],     # CP1
    [0.0, -1.0, 0.0],        # CPZ
    [0.34, -0.94, 0.0],      # CP2
    [0.5, -0.87, 0.0],       # CP4
    [0.67, -0.74, 0.0],      # TP8
    [-0.92, -0.39, 0.0],     # P7
    [-0.75, -0.66, 0.0],     # P5
    [-0.5, -0.87, 0.0],       # P3
    [-0.34, -0.94, 0.0],     # P1
    [0.0, -1.0, 0.0],        # PZ
    [0.34, -0.94, 0.0],      # P2
    [0.5, -0.87, 0.0],       # P4
    [0.75, -0.66, 0.0],      # P6
    [0.92, -0.39, 0.0],      # P8
    [-0.67, -0.74, 0.0],     # O1
    [0.0, -1.0, 0.0],        # OZ
    [0.67, -0.74, 0.0]       # O2
])  # shape: (32, 3)

#edge_index, edge_weight = build_electrode_graph(electrode_coords)


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
    edge_index, edge_weight = build_electrode_graph(electrode_coords)
    # Dataset
    train_dataset = GraphEEGDataset(X_train_eeg, X_train_de, y_train, edge_index, edge_weight=None,
                                    tasks=tasks)
    val_dataset = GraphEEGDataset(X_val_eeg, X_val_de, y_val, edge_index, edge_weight=None,
                                  tasks=tasks)
    test_dataset = GraphEEGDataset(X_test_eeg, X_test_de, y_test, edge_index, edge_weight=None,
                                   tasks=tasks)

    # Используем кастомный collate
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=train_dataset.collate)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=val_dataset.collate)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=test_dataset.collate)

    model = DualInputGNNModel(tasks=tasks)
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