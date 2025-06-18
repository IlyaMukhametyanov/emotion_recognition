import os
import pickle
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn, SWALR
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import torch
import torch.nn as nn
from kan import KAN

# Параметры
NUM_SUBJECTS = 32
NUM_TRIALS = 40
NUM_EEG_CHANNELS = 32
SEGMENT_LENGTH = 448
BATCH_SIZE = 512
EPOCHS = 200


class KANTransformerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn_kan = KAN(width=[dim, dim], grid=5)
        self.ffn = nn.Sequential(
            KAN(width=[dim, dim * 2], grid=3),
            nn.GELU(),
            KAN(width=[dim * 2, dim], grid=3)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        attn = torch.sigmoid(self.attn_kan(x))
        x = x * attn
        return self.norm(x + self.ffn(x))


class TransitionBlock_Hybrid(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.kan = KAN(width=[in_channels, out_channels], grid=5)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        return self.pool(self.kan(x.transpose(1, 2)).transpose(1, 2))


class MultiTaskEEG_Hybrid(nn.Module):
    def __init__(self):
        super().__init__()
        # Входные слои
        self.conv1 = nn.Conv1d(32, 24, kernel_size=5, padding=2)

        # Трансформерные блоки
        self.blocks = nn.Sequential(
            *[KANTransformerBlock(24) for _ in range(6)],
            TransitionBlock_Hybrid(24, 96),
            *[KANTransformerBlock(96) for _ in range(6)],
            TransitionBlock_Hybrid(96, 168),
            *[KANTransformerBlock(168) for _ in range(6)]
        )

        # MoE-головки
        self.moe_heads = nn.ModuleDict({
            task: nn.Sequential(
                KAN(width=[168, 64], grid=5),
                nn.Linear(64, 2)
            ) for task in ['valence', 'arousal', 'dominance']
        })

    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks(x)
        x = x.mean(dim=-1)
        return {task: head(x) for task, head in self.moe_heads.items()}


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
    swa_model = AveragedModel(model).to(device)
    best_val_f1 = 0.0
    scheduler = OneCycleLR(optimizer, max_lr=3e-3, total_steps=EPOCHS * len(train_loader))
    swa_start =  int(EPOCHS * 0.75)  # Начинаем усреднение с 75% эпох
    swa_scheduler = SWALR(optimizer, swa_lr=0.05)  # Специальный LR для фазы SWA
    active = True
    frezze = 0
    last_acc = []

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


            train_loss += loss.item()

        # Начиная с swa_start, активируем SWA
        if (epoch < swa_start and (int(epoch) % 10 == 0) or
                (epoch >= swa_start and int(epoch) % 3 == 0 and active == True)):
            swa_model.update_parameters(model)  # Обновляем усредненные веса
            swa_scheduler.step()  # Обновляем LR
            # Валидация
            swa_model.eval()
            val_metrics = {'valence': [], 'arousal': [], 'dominance': []}
            with torch.no_grad():
                for X, y in val_loader:
                    X = X.to(device)
                    y = {k: v.to(device) for k, v in y.items()}
                    outputs = swa_model(X)

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
        else:
            scheduler.step()

            if active== False and frezze < 10:
                frezze +=1
            elif active == False and frezze >= 10:
                active = True
                frezze = 0

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
    update_bn(train_loader, swa_model, device=device)
    return swa_model


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
    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
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
    model = MultiTaskEEG_Hybrid()
    optimizer = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

    # Обучение
    trained_model = train_model(model, train_loader, val_loader, optimizer, device)

    # Тестирование
    results = evaluate(trained_model, test_loader, device)
    torch.save(trained_model.state_dict(), f'save_models/model_weights_{current_date}.pth')
    for task in results:
        print(f"{task}:")
    print(f"  Accuracy: {results[task]['Accuracy']:.4f}")
    print(f"  F1 Score: {results[task]['F1']:.4f}")
    print("-------------------")

if __name__ == "__main__":
    main()