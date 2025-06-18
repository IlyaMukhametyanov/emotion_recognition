import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
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


class CDCN(nn.Module):
    def __init__(self, num_classes=2, in_channels=32):  # Всегда 2 класса для бинарной классификации
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 24, kernel_size=5, padding=2)
        self.dense1 = DenseBlock(24, 6, 12)
        self.trans1 = TransitionBlock(96, 96)
        self.dense2 = DenseBlock(96, 6, 12)
        self.trans2 = TransitionBlock(168, 168)
        self.dense3 = DenseBlock(168, 6, 12)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(240, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.trans2(x)
        x = self.dense3(x)
        x = self.global_pool(x).squeeze(-1)
        return self.fc(x)


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
        binary_labels = (labels > thresholds).astype(np.float32)

        for trial in range(NUM_TRIALS):
            trial_data = eeg_data[trial]
            num_segments = trial_data.shape[1] // SEGMENT_LENGTH
            segments = np.split(trial_data[:, :num_segments * SEGMENT_LENGTH], num_segments, axis=1)
            all_segments.extend(segments)
            all_labels.extend([binary_labels[trial]] * num_segments)

    return np.array(all_segments), np.array(all_labels)

def load_and_process_data_cross(subject_ids, data_dir="data"):
    all_segments = []
    all_labels = []

    for subject_id in subject_ids:
        file_path = os.path.join(data_dir, f"s{subject_id:02d}.dat")
        with open(file_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        eeg_data = data['data'][:, :32, :]  # (40, 32, 8064)
        labels = data['labels'][:, :3]  # (40, 3)

        thresholds = np.median(labels, axis=0)
        binary_labels = (labels > thresholds).astype(np.float32)


        # Сегментация данных
        for trial in range(NUM_TRIALS):
            trial_data = eeg_data[trial]
            num_segments = trial_data.shape[1] // SEGMENT_LENGTH
            segments = np.split(trial_data[:, :num_segments * SEGMENT_LENGTH], num_segments, axis=1)
            all_segments.extend(segments)
            all_labels.extend([binary_labels[trial]] * num_segments)

    return np.array(all_segments), np.array(all_labels)


def train_model(model, train_loader, val_loader, optimizer, criterion, device, target_dims, epochs=EPOCHS):
    model.to(device)
    swa_model = AveragedModel(model).to(device)
    best_val_accuracy = 0.0
    swa_start_epoch = 1

    total_steps = epochs * len(train_loader)

    scheduler = OneCycleLR(
        optimizer,
        max_lr = 3e-3,
        total_steps=total_steps,
        pct_start=0.3,
        three_phase=True,  # Трехфазный цикл
        anneal_strategy='cos',
        cycle_momentum=True,
        # base_momentum=0.85,
        # max_momentum=0.95,
        div_factor=25,
        final_div_factor=1e3
    )

    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0

        progress_bar = tqdm(enumerate(train_loader, 1),
                            total=len(train_loader),
                            desc=f"Epoch {epoch + 1}/{epochs}",
                            leave=False)

        for batch_idx, (X, y) in progress_bar:
            X = X.to(device)
            y_target = y[:, target_dims].long().squeeze().to(device)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y_target)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_target).sum().item()
            total += y_target.size(0)

            progress_bar.set_postfix({
                'Loss': f"{train_loss / batch_idx:.4f}",
                'Acc': f"{correct / total:.4f}",
                'LR': f"{scheduler.get_last_lr()[0]:.2e}"
            })

        progress_bar.close()

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                y_target = y[:, target_dims].long().squeeze().to(device)

                outputs = model(X)
                val_loss += criterion(outputs, y_target).item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == y_target).sum().item()
                val_total += y_target.size(0)

        val_acc = val_correct / val_total
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss / len(train_loader):.4f} | Acc: {correct / total:.4f}")
        print(f"Val Loss: {val_loss / len(val_loader):.4f} | Acc: {val_acc:.4f}")
        print("-" * 50)

        if epoch >= swa_start_epoch and val_acc >= best_val_accuracy:
            best_val_accuracy = val_acc
            swa_model.update_parameters(model)
            print(f"SWA updated at epoch {epoch + 1}, val_accuracy={val_acc:.4f}")

    # Final SWA steps
    print("\nApplying SWA and updating BatchNorm...")
    update_bn(train_loader, swa_model, device=device)

    # Fix state dict keys
    swa_state_dict = swa_model.state_dict()
    fixed_state_dict = {k.replace('module.', ''): v for k, v in swa_state_dict.items()}

    model.load_state_dict(fixed_state_dict, strict=False)
    model.eval()

    return model


def test_model(model, test_loader, device, target_dims):
    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_true.extend(y[:, target_dims].long().squeeze().cpu().numpy())

    print("\nРезультаты тестирования:")
    print(f"Accuracy: {accuracy_score(all_true, all_preds):.4f}")
    print(f"F1 Score: {f1_score(all_true, all_preds, average='binary'):.4f}")
    print(f"Precision: {precision_score(all_true, all_preds, average='binary'):.4f}")
    print(f"Recall: {recall_score(all_true, all_preds, average='binary'):.4f}")
    print("=" * 50)


def main():
    # Загрузка данных
    print("Загрузка данных...")
    X, y = load_and_process_data_cross(range(1, NUM_SUBJECTS + 1))  # Все участники

    # Нормализация
    mean = X.mean(axis=(0, 2), keepdims=True)
    std = X.std(axis=(0, 2), keepdims=True)
    X = (X - mean) / std

    # Преобразование в тензоры
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)

    # Стратифицированное разделение (сохраняем распределение классов)
    # 1. 90% train / 10% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.1,
        random_state=42,
        shuffle=True,
        stratify=y  # Для многоклассовой стратификации
    )

    # 2. Делим temp на test и val (1:1)
    X_test, y_test  =X_temp,y_temp

    X_val = X_test.clone()  # вместо .copy()
    y_val = y_test.clone()  # если y_test тоже тензор

    # Создание DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, BATCH_SIZE)
    test_loader = DataLoader(test_dataset, BATCH_SIZE)

    # Вывод информации
    print(f"Всего примеров: {len(X)}")
    print(f"Train: {len(X_train)} ({len(X_train) / len(X):.1%})")
    print(f"Val: {len(X_val)} ({len(X_val) / len(X):.1%})")
    print(f"Test: {len(X_test)} ({len(X_test) / len(X):.1%})")

    # Инициализация моделей
    models = {
        'valence': CDCN(num_classes=2),  # Классификация Valence
        'arousal': CDCN(num_classes=2),  # Классификация Arousal
        'dominance': CDCN(num_classes=2)  # Классификация Dominance
    }

    for model_name, model in models.items():
        print(f"\n{'#' * 50}")
        print(f"Обучение модели: {model_name}")
        print(f"{'#' * 50}")

        target_dims = {
            'valence': [0],
            'arousal': [1],
            'dominance': [2]
        }[model_name]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        trained_model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            target_dims=target_dims
        )

        test_model(
            model=trained_model,
            test_loader=test_loader,
            device=device,
            target_dims=target_dims
        )


if __name__ == "__main__":
    main()