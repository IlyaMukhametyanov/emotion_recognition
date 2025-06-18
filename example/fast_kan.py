import torch
import numpy as np
from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# MatrixKAN
from MatrixKAN import MatrixKAN

# Original KAN
from kan import KAN


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def create_dataset(n_samples=10000):
    X = torch.rand(n_samples, 2, dtype=torch.float32) * 4 - 2
    y = torch.sin(X[:, 0]) + torch.cos(X[:, 1])
    return X, y.unsqueeze(1)


# Данные
X_train, y_train = create_dataset()
X_train = X_train.to(device).float()
y_train = y_train.to(device).float()


def train_model(model, X, y, epochs=15, lr=0.01, batch_size=256):
    device = next(model.parameters()).device
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    total_start_time = time()

    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

        for x_batch, y_batch in progress_bar:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()

            # Для совместимости: forward возвращает (output, ...)
            output_tuple = model(x_batch)
            y_pred = output_tuple[0]  # Получаем только выход модели

            # Формы должны быть одинаковыми
            loss = loss_fn(y_pred.squeeze(), y_batch.squeeze())

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_epoch_loss = epoch_loss / len(train_loader)
        tqdm.write(f"[Epoch {epoch + 1}/{epochs}] Loss: {avg_epoch_loss:.4f}")

    total_time = time() - total_start_time
    print(f"\n✅ Обучение завершено за {total_time:.2f} секунд")
    return total_time


# Создание моделей
model_matrixkan = MatrixKAN(
    width=[2, 257, 129, 65, 33, 17, 9, 5, 1],  # 8 слоёв
    grid=5,
    k=3,
    mult_arity=2,       # параллелизуемая мультипликация
    base_fun='silu',
    device=device
).float().to(device)

model_kan = KAN(width=[2, 257, 129, 65, 33, 17, 9, 5, 1], grid=5, k=3, noise_scale=0.1)
model_kan.to(device)

# Обучение
print("Training MatrixKAN...")
time_matrixkan = train_model(model_matrixkan, X_train, y_train, epochs=20, lr=1e-2)

print("\nTraining KAN...")
time_kan = train_model(model_kan, X_train, y_train, epochs=20, lr=1e-2)

# Вывод результатов
print("\n--- Performance Comparison ---")
print(f"MatrixKAN: {time_matrixkan:.2f} sec")
print(f"KAN:       {time_kan:.2f} sec")
print(f"Speedup:   {time_kan / time_matrixkan:.2f}x")

# График
plt.figure(figsize=(6, 4))
models = ['KAN', 'MatrixKAN']
times = [time_kan, time_matrixkan]
plt.bar(models, times, color=['red', 'green'])
plt.ylabel('Time (seconds)')
plt.title('Training Time Comparison')
plt.grid(True)
plt.show()