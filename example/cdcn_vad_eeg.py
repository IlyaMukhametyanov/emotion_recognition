import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# Параметры
NUM_SUBJECTS = 32
NUM_TRIALS = 40
NUM_EEG_CHANNELS = 32
SEGMENT_LENGTH = 672
BATCH_SIZE = 512
EPOCHS = 200


# Модули CDCN
class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.GELU(),
            nn.Conv1d(in_channels, growth_rate, kernel_size=3, padding=1, stride=1)
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
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(self.conv(x))


class MultiTaskEEG(nn.Module):
    def __init__(self):
        super().__init__()
        # Общие слои
        self.conv1 = nn.Conv1d(32, 24, kernel_size=5, padding=2)
        self.dense1 = DenseBlock(24, 6, 12)
        self.trans1 = TransitionBlock(96, 96)
        self.dense2 = DenseBlock(96, 6, 12)
        self.trans2 = TransitionBlock(168, 168)
        self.dense3 = DenseBlock(168, 6, 12)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Индивидуальные головы
        self.valence = nn.Sequential(
            nn.Linear(240, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
        self.arousal = nn.Sequential(
            nn.Linear(240, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
        self.dominance = nn.Sequential(
            nn.Linear(240, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.trans2(x)
        x = self.dense3(x)
        x = self.global_pool(x).squeeze(-1)

        return {
            'valence': self.valence(x),
            'arousal': self.arousal(x),
            'dominance': self.dominance(x)
        }


class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = {
            'valence': torch.LongTensor(y[:, 0]),
            'arousal': torch.LongTensor(y[:, 1]),
            'dominance': torch.LongTensor(y[:, 2])
        }

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], {
            'valence': self.y['valence'][idx],
            'arousal': self.y['arousal'][idx],
            'dominance': self.y['dominance'][idx]
        }


def load_and_process_data(subject_ids, data_dir="data"):
    all_segments = []
    all_labels = []

    for subject_id in subject_ids:
        file_path = os.path.join(data_dir, f"s{subject_id:02d}.dat")
        with open(file_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        eeg_data = data['data'][:, :32, :]
        labels = data['labels'][:, :3]

        thresholds = np.median(labels, axis=0)
        binary_labels = (labels > thresholds).astype(int)

        for trial in range(NUM_TRIALS):
            trial_data = eeg_data[trial]
            num_segments = trial_data.shape[1] // SEGMENT_LENGTH
            segments = np.split(trial_data[:, :num_segments * SEGMENT_LENGTH], num_segments, axis=1)
            all_segments.extend(segments)
            all_labels.extend([binary_labels[trial]] * num_segments)

    return np.array(all_segments), np.array(all_labels)


def train_model(model, train_loader, val_loader, optimizer, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    #swa_model = AveragedModel(model).to(device)
    best_val_f1 = 0.0
    scheduler = OneCycleLR(optimizer, max_lr=3e-3, total_steps=EPOCHS * len(train_loader))

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

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

        # Валидация
        model.eval()
        val_metrics = {'valence': [], 'arousal': [], 'dominance': []}
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                y = {k: v.to(device) for k, v in y.items()}
                outputs = model(X)

                for task in outputs:
                    preds = torch.argmax(outputs[task], dim=1)
                    val_metrics[task].append({
                        'preds': preds.cpu(),
                        'targets': y[task].cpu()
                    })

        # Расчет метрик
        avg_f1 = 0
        for task in val_metrics:
            all_preds = torch.cat([x['preds'] for x in val_metrics[task]])
            all_targets = torch.cat([x['targets'] for x in val_metrics[task]])
            f1 = f1_score(all_targets, all_preds, average='macro')
            avg_f1 += f1
            print(f"{task} F1: {f1:.4f}")

        avg_f1 /= 3
        if avg_f1 > best_val_f1:
            best_val_f1 = avg_f1
            torch.save(model.state_dict(), "../best_model.pth")

        print(f"Epoch {epoch + 1} | Train Loss: {train_loss / len(train_loader):.4f} | Val F1: {avg_f1:.4f}")

    # Применяем SWA
    #update_bn(train_loader, swa_model, device=device)
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загрузка данных
    X, y = load_and_process_data(range(1, NUM_SUBJECTS + 1))
    # X = (X - X.mean(axis=(0, 2), keepdims=True)) / X.std(axis=(0, 2), keepdims=True)))

     # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    # Создание Dataset
    train_dataset = EEGDataset(X_train, y_train)
    val_dataset = EEGDataset(X_val, y_val)
    test_dataset = EEGDataset(X_test, y_test)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Инициализация модели
    model = MultiTaskEEG()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Обучение
    trained_model = train_model(model, train_loader, val_loader, optimizer, device)

    # Тестирование
    results = evaluate(trained_model, test_loader, device)
    for task in results:
        print(f"{task}:")
    print(f"  Accuracy: {results[task]['Accuracy']:.4f}")
    print(f"  F1 Score: {results[task]['F1']:.4f}")
    print("-------------------")

if __name__ == "__main__":
    main()