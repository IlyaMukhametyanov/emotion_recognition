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

# Параметры
NUM_SUBJECTS = 32
NUM_TRIALS = 40
NUM_EEG_CHANNELS = 32
SEGMENT_LENGTH = 448
BATCH_SIZE = 512
EPOCHS = 200


class PhysioConstrainedHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim)
        )

        # Альтернативная реализация ограничений
        self.constraint_fc = nn.Linear(output_dim, 1)
        self._init_constraints()

    def _init_constraints(self):
        """Инициализация нейрофизиологических правил через веса"""
        with torch.no_grad():
            self.constraint_fc.weight.data = torch.tensor([[1.0, -1.0]])
            self.constraint_fc.bias.data = torch.tensor([-0.4])

    def forward(self, x):
        logits = self.mlp(x)
        constraints = torch.sigmoid(self.constraint_fc(logits))
        return logits * constraints


class EEGArtifactFilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter_net = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.filter_net(x)


class MultiTaskEEG(nn.Module):
    def __init__(self):
        super().__init__()
        self.artifact_filter = EEGArtifactFilter()

        self.conv1 = nn.Conv1d(32, 24, kernel_size=5, padding=2)
        self.dense1 = self._make_dense_block(24, 6, 12)
        self.trans1 = TransitionBlock(96, 96)
        self.dense2 = self._make_dense_block(96, 6, 12)
        self.trans2 = TransitionBlock(168, 168)
        self.dense3 = self._make_dense_block(168, 6, 12)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.valence = PhysioConstrainedHead(240, 2)
        self.arousal = PhysioConstrainedHead(240, 2)

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
        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.trans2(x)
        x = self.dense3(x)
        x = self.global_pool(x).squeeze(-1)

        return {
            'valence': self.valence(x),
            'arousal': self.arousal(x)
        }


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.GELU(),
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
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(self.conv(x))


class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = {
            'valence': torch.LongTensor(y[:, 0]),
            'arousal': torch.LongTensor(y[:, 1])
        }

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], {
            'valence': self.y['valence'][idx],
            'arousal': self.y['arousal'][idx]
        }


def load_and_process_data(subject_ids, data_dir="data"):
    all_segments = []
    all_labels = []

    for subject_id in subject_ids:
        file_path = os.path.join(data_dir, f"s{subject_id:02d}.dat")
        with open(file_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        eeg_data = data['data'][:, :32, :]
        labels = data['labels'][:, :2]  # Берем только valence и arousal

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
    scheduler = OneCycleLR(
        optimizer,
        max_lr = 3e-3,
        total_steps=EPOCHS * len(train_loader),
        pct_start=0.3,
        three_phase=True,  # Трехфазный цикл
        anneal_strategy='cos',
        cycle_momentum=True,
        # base_momentum=0.85,
        # max_momentum=0.95,
        div_factor=25,
        final_div_factor=1e3
    )
    swa_start = int(EPOCHS * 0.75)
    swa_scheduler = SWALR(optimizer, swa_lr=0.05)
    active = True
    frezze = 0

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

        if (epoch < swa_start and (epoch % 10 == 0) or
                (epoch >= swa_start and epoch % 3 == 0 and active)):
            swa_model.update_parameters(model)
            swa_scheduler.step()

            swa_model.eval()
            val_metrics = {'valence': [], 'arousal': []}
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

            avg_f1 = 0
            for task in val_metrics:
                all_preds = torch.cat([x['preds'] for x in val_metrics[task]])
                all_targets = torch.cat([x['targets'] for x in val_metrics[task]])
                f1 = f1_score(all_targets, all_preds, average='macro')
                avg_f1 += f1
                print(f"{task} F1: {f1:.4f}")
            avg_f1 /= 2

            if avg_f1 > best_val_f1:
                best_val_f1 = avg_f1
                torch.save(model.state_dict(), "../best_model.pth")
            print(f"Epoch {epoch + 1} | Train Loss: {train_loss / len(train_loader):.4f} | Val F1: {avg_f1:.4f}")
        else:
            scheduler.step()

            model.eval()
            val_metrics = {'valence': [], 'arousal': []}
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

            avg_f1 = 0
            for task in val_metrics:
                all_preds = torch.cat([x['preds'] for x in val_metrics[task]])
                all_targets = torch.cat([x['targets'] for x in val_metrics[task]])
                f1 = f1_score(all_targets, all_preds, average='macro')
                avg_f1 += f1
                print(f"{task} F1: {f1:.4f}")
            avg_f1 /= 2

            if avg_f1 > best_val_f1:
                best_val_f1 = avg_f1
                torch.save(model.state_dict(), "../best_model.pth")
            print(f"Epoch {epoch + 1} | Train Loss: {train_loss / len(train_loader):.4f} | Val F1: {avg_f1:.4f}")

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

    for task in results:
        print(f"{task}:")
        print(f"  Accuracy: {results[task]['Accuracy']:.4f}")
        print(f"  F1 Score: {results[task]['F1']:.4f}")
        print("-------------------")


if __name__ == "__main__":
    main()