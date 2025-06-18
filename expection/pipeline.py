import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import torch.nn.functional as F
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
from concurrent.futures import ProcessPoolExecutor, as_completed

from cdcn_hslt import R2G_CDCN_Hybrid
from cdcn_hslt_kan import R2G_CDCN_KAN_Hybrid

# Параметры
NUM_SUBJECTS = 32
NUM_TRIALS = 40
NUM_EEG_CHANNELS = 32
SEGMENT_LENGTH = 672
BATCH_SIZE = 5500
EPOCHS = 1000
STRIDE = 3 * 128  # 3 секунды → 384 отсчёта



class EEGDataset(Dataset):
    def __init__(self, X, y, tasks: List[str] = ['valence', 'arousal', 'dominance']):
        self.X = torch.FloatTensor(X)
        self.y = {
            'valence': torch.LongTensor(y[:, 0]),
            'arousal': torch.LongTensor(y[:, 1]),
            'dominance': torch.LongTensor(y[:, 2])
        }
        self.tasks = tasks

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], {task: self.y[task][idx] for task in self.tasks}

# Эти функции должны быть доступны в каждом процессе
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


# === Функция для обработки одного субъекта ===
def process_subject(subject_id, data_dir, fs, SEGMENT_LENGTH, STRIDE):
    file_path = os.path.join(data_dir, f"s{subject_id:02d}.dat")
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    eeg_data = data['data'][:, :32, :]  # shape [trials, channels, samples]
    labels = data['labels'][:, :3]      # shape [trials, 3]
    binary_labels = (labels > np.median(labels, axis=0)).astype(int)

    subject_features = []
    subject_labels = []

    # тут надо почистит сигнал

    for trial in range(eeg_data.shape[0]):
        trial_data = eeg_data[trial]  # shape [32, samples]

        num_samples = trial_data.shape[1]
        num_segments = (num_samples - SEGMENT_LENGTH) // STRIDE + 1

        for i in range(num_segments):
            start = i * STRIDE
            end = start + SEGMENT_LENGTH
            segment = trial_data[:, start:end]  # shape [32, 768]

            features = compute_de_features(segment, fs)
            subject_features.append(features)
            subject_labels.append(binary_labels[trial])

    return np.array(subject_features), np.array(subject_labels)


# === Параллельная загрузка ===
def load_and_process_data(subject_ids, data_dir="data", fs=128, use_cache=True, verbose=True):
    SEGMENT_LENGTH = 6 * fs  # 6 секунд
    STRIDE = 3 * fs          # 3 секунды

    cache_path = os.path.join(data_dir, f"de_feature_3_v2{fs}.npz")

    if use_cache and os.path.exists(cache_path):
        data = np.load(cache_path)
        return data['X'], data['y']

    all_features = []
    all_labels = []

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_subject, sid, data_dir, fs, SEGMENT_LENGTH, STRIDE)
            for sid in subject_ids
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Обработка субъектов", disable=not verbose):
            features, labels = future.result()
            all_features.extend(features)
            all_labels.extend(labels)

    X = np.array(all_features)
    y = np.array(all_labels)

    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X.reshape(-1, 5)).reshape(X.shape)

    if use_cache:
        np.savez(cache_path, X=X_normalized, y=y)

    return X_normalized, y

class LabelSmoothedFocalLoss(nn.Module):
    def __init__(self, smoothing=0.1, gamma=2):
        super().__init__()
        self.smoothing = smoothing
        self.gamma = gamma
        self.confidence = 1 - smoothing

    def forward(self, pred, target):
        """
        pred: [B, C] - логиты
        target: [B] - метки классов
        """
        log_prob = F.log_softmax(pred, dim=-1)
        n_classes = pred.size(-1)

        # Создаём "мягкие" метки
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        # Вычисляем pt (вероятность истинного класса)
        probs = torch.exp(log_prob)
        pt = probs.gather(1, target.unsqueeze(1)).squeeze()

        # Focal Loss веса
        focal_weight = (1 - pt) ** self.gamma

        # Умножаем log_prob на true_dist и применяем focal_weight
        losses = -log_prob * true_dist
        losses = losses.sum(dim=-1)  # сумма по классам
        focal_losses = focal_weight * losses  # apply focal scaling

        return focal_losses.mean()

def train_model(model, train_loader, val_loader, optimizer, device,tasks, smoothing=0.2,alpha=0.3):
    model.to(device)
    criterion =  LabelSmoothedFocalLoss(smoothing=smoothing, gamma=2)
    best_val_f1 = 0.0
    scheduler = OneCycleLR(optimizer, max_lr=3e-3, total_steps=EPOCHS * len(train_loader))
    task_weights = {task: 1.0 for task in tasks}  # начальные веса

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        all_train_preds = []
        all_train_targets = []

        # Тренировка
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for X, y in progress_bar:
            X = X.to(device)
            y = {k: v.to(device) for k, v in y.items()}

            optimizer.zero_grad()
            outputs = model(X)

            # --- Сбор потерь по каждой задаче ---
            task_losses = {}
            task_preds = []
            task_targets = []

            for task in model.tasks:
                preds = torch.argmax(outputs[task], dim=1)
                task_preds.append(preds)
                task_targets.append(y[task])

                task_losses[task] = criterion(outputs[task], y[task]).item()

            # --- Динамические веса: больший вес на сложные задачи ---
            losses_tensor = torch.tensor([task_losses[task] for task in model.tasks])
            losses_tensor = losses_tensor / losses_tensor.sum()
            weights = (1 - losses_tensor) + 0.1
            weights = weights / weights.sum()  # нормализация

            # --- Взвешенная потеря ---
            weighted_loss = torch.tensor(0.0).to(device)
            for i, task in enumerate(model.tasks):
                loss_task = criterion(outputs[task], y[task])
                weighted_loss += weights[i] * loss_task

            # --- Exact Match Loss (EML): штраф за неточное совпадение ---
            preds_array = torch.stack(task_preds, dim=1)
            targets_array = torch.stack(task_targets, dim=1)
            exact_match = (preds_array == targets_array).all(dim=1)
            eml_weight = (~exact_match).float().mean()
            loss = weighted_loss + alpha * eml_weight

            # --- Обучение ---
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

            # Сбор предсказаний по всем задачам
            task_preds = preds_array
            task_targets = targets_array

            all_train_preds.append(task_preds.cpu())
            all_train_targets.append(task_targets.cpu())

            # Для прогрессбара: точность по последнему батчу
            batch_exact_match = (task_preds.cpu().numpy() == task_targets.cpu().numpy()).all(axis=1).mean()
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'batch_acc': f"{batch_exact_match:.4f}"
            })

        all_train_preds = torch.cat(all_train_preds).numpy()
        all_train_targets = torch.cat(all_train_targets).numpy()

        # Точность только если все метки совпадают
        exact_match = (all_train_preds == all_train_targets).all(axis=1)
        train_acc = exact_match.mean()


        # ================= ВАЛИДАЦИЯ =================
        model.eval()
        val_metrics = {task: [] for task in model.tasks}
        val_loss = 0

        all_val_preds = []
        all_val_targets = []

        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                y_batch = {k: v.to(device) for k, v in y.items()}
                outputs = model(X)

                # Считаем loss
                val_loss += sum(criterion(outputs[task], y_batch[task]).item() for task in outputs)

                batch_preds = []
                batch_targets = []

                for task in outputs:
                    preds = torch.argmax(outputs[task], dim=1)
                    val_metrics[task].append({
                        'preds': preds.cpu(),
                        'targets': y_batch[task].cpu()
                    })
                    batch_preds.append(preds.cpu().numpy())
                    batch_targets.append(y_batch[task].cpu().numpy())

                # Сохраняем для Exact Match
                all_val_preds.extend(np.stack(batch_preds, axis=1))  # shape [batch_size, 3]
                all_val_targets.extend(np.stack(batch_targets, axis=1))

        # Расчёт метрик по задачам
        avg_f1 = 0
        all_val_preds_task = {task: [] for task in model.tasks}
        all_val_targets_task = {task: [] for task in model.tasks}

        for task in val_metrics:
            all_preds = torch.cat([x['preds'] for x in val_metrics[task]])
            all_targets = torch.cat([x['targets'] for x in val_metrics[task]])
            f1 = f1_score(all_targets, all_preds, average='macro')
            acc = accuracy_score(all_targets, all_preds)
            avg_f1 += f1
            all_val_preds_task[task] = all_preds.numpy()
            all_val_targets_task[task] = all_targets.numpy()
            print(f"{task} F1: {f1:.4f} | Acc: {acc:.4f}")

        avg_f1 /= len(val_metrics)

        # Точность только если все метки совпадают (Exact Match / Subset Accuracy)
        all_val_preds_np = np.array(all_val_preds)  # shape [N_samples, N_tasks]
        all_val_targets_np = np.array(all_val_targets)  # shape [N_samples, N_tasks]

        exact_match = (all_val_preds_np == all_val_targets_np).all(axis=1)
        exact_match_acc = exact_match.mean()

        # +++ Подсчёт ошибок в зависимости от количества задач +++
        correct_counts = (all_val_preds_np == all_val_targets_np).sum(axis=1)  # число правильных меток на пример
        num_tasks = all_val_preds_np.shape[1]

        # Подсчитываем статистику по числу совпадений
        acc_val = 0
        error_distribution = {}
        for i in range(num_tasks, -1, -1):  # от "все правильные" до "все неправильные"
            count = np.sum(correct_counts == i)
            percent = 100 * count / len(correct_counts)
            error_distribution[i] = (count, percent)

        print(f"\nРаспределение ошибок ({num_tasks} задач):")
        for correct, (count, percent) in sorted(error_distribution.items()):
            if correct == num_tasks:
                acc_val = percent
                print(f"✅ Все метки верны: {count} ({percent:.2f}%)")
            elif correct == 0:
                print(f"❌ Все метки неверны: {count} ({percent:.2f}%)")
            else:
                wrong = num_tasks - correct
                print(f"⚠️ Ошибок: {wrong}: {count} ({percent:.2f}%)")

        print("-" * 50)

        # Сохранение лучшей модели по avg_f1
        if avg_f1 > best_val_f1:
            best_val_f1 = avg_f1
            torch.save(model.state_dict(), "../best_model.pth")

        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch: {epoch} "
              f"Train/val  Loss: {train_loss / len(train_loader):.4f}|{val_loss / len(val_loader):.4f} "
              f"Acc: {train_acc:.4f}|{acc_val/100} "
              f"Learning Rate: {current_lr:.10f}")
    return model

def evaluate(model, test_loader, device):
    model.eval()
    metrics = {}

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            outputs = model(X)

            # Сбор предсказаний по всем задачам
            task_names = list(outputs.keys())
            preds_array = torch.stack([torch.argmax(outputs[task], dim=1) for task in task_names], dim=1).cpu().numpy()
            targets_array = torch.stack([y[task] for task in task_names], dim=1).cpu().numpy()

            # Сохраняем для оценки по отдельным задачам
            for i, task in enumerate(task_names):
                if task not in metrics:
                    metrics[task] = {'preds': [], 'targets': []}
                metrics[task]['preds'].extend(preds_array[:, i])
                metrics[task]['targets'].extend(targets_array[:, i])

            # Сохраняем для оценки по триплету
            all_preds.append(preds_array)
            all_targets.append(targets_array)

    # Точность по триплету
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    exact_match = (all_preds == all_targets).all(axis=1)
    triplet_accuracy = exact_match.mean()

    # Точность и F1 по каждой задаче
    results = {}
    for task in metrics:
        results[task] = {
            'Accuracy': accuracy_score(metrics[task]['targets'], metrics[task]['preds']),
            'F1': f1_score(metrics[task]['targets'], metrics[task]['preds'], average='macro')
        }

    # Добавляем общую метрику
    results['triplet'] = {
        'Exact Match Accuracy': triplet_accuracy
    }

    return results

#'arousal',
def main(tasks: List[str] = ['valence',  'dominance']):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Загрузка данных
    X, y = load_and_process_data(range(1, NUM_SUBJECTS + 1))
    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    # Создание Dataset
    train_dataset = EEGDataset(X_train, y_train, tasks=tasks)
    val_dataset = EEGDataset(X_val, y_val, tasks=tasks)
    test_dataset = EEGDataset(X_test, y_test, tasks=tasks)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Инициализация модели
    model = R2G_CDCN_KAN_Hybrid(input_shape=(32, 5), tasks=tasks)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Обучение
    trained_model = train_model(model, train_loader, val_loader, optimizer, device, tasks)
    torch.save(trained_model.state_dict(), f'save_models/model_weights_{current_date}.pth')
    # Тестирование
    results = evaluate(trained_model, test_loader, device)
    # for task in results:
    #     print(f"{task}:")
    #     print(f"  Accuracy: {results[task]['Accuracy']:.4f}")
    #     print(f"  F1 Score: {results[task]['F1']:.4f}")
    # print("-------------------")

if __name__ == "__main__":
    main()