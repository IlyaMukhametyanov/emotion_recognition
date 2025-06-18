from kan import KAN
import torch
import torch.nn as nn
import pennylane as qml
import math
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




class CDCN_KAN_Enhanced(nn.Module):
    def __init__(self, num_classes=4, in_channels=32, time_points=128):
        super().__init__()

        # Initial conv
        self.conv1 = nn.Conv1d(in_channels, 24, kernel_size=5, padding=2)

        # Dense Blocks
        self.dense1 = DenseBlock(24, 6, 12)
        self.trans1 = TransitionBlock(96, 96)

        self.dense2 = DenseBlock(96, 6, 12)
        self.trans2 = TransitionBlock(168, 168)

        self.dense3 = DenseBlock(168, 6, 12)

        # KAN ветви
        self.kan_dense1 = KAN(width=[96, 64, 96], grid=5, k=3)
        self.kan_dense2 = KAN(width=[168, 64, 168], grid=5, k=3)

        # KAN на последнем слое
        self.kan_last = KAN(width=[240, 64, 240], grid=5, k=3)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(240, num_classes)

    def forward(self, x):
        # [B, 1, 32, 128]
        x = x.squeeze(1)  # [B, 32, 128]
        B, C, T = x.shape

        # Conv
        x = self.conv1(x)  # [B, 24, 128]

        # Dense Block 1
        x = self.dense1(x)  # [B, 96, 128]
        x_kan = x.permute(0, 2, 1).reshape(B * T, 96)  # -> [B*T, C]
        x_kan = self.kan_dense1(x_kan)  # -> [B*T, 96]
        x_kan = x_kan.view(B, T, 96).permute(0, 2, 1)  # -> [B, 96, T]
        x = x + x_kan  # Сумма или concat

        x = self.trans1(x)  # [B, 96, 64]

        # Dense Block 2
        x = self.dense2(x)  # [B, 168, 64]
        B, C, T = x.shape
        x_kan = x.permute(0, 2, 1).reshape(B * T, 168)
        x_kan = self.kan_dense2(x_kan)
        x_kan = x_kan.view(B, T, 168).permute(0, 2, 1)
        x = x + x_kan

        x = self.trans2(x)  # [B, 168, 32]

        # Dense Block 3
        x = self.dense3(x)  # [B, 240, 32]

        # KAN на последнем слое
        B, C, T = x.shape
        x_kan = x.permute(0, 2, 1).reshape(B * T, C)
        x_kan = self.kan_last(x_kan)
        x = x_kan.view(B, T, C).permute(0, 2, 1)  # [B, 240, 32]

        # Классификация
        x = self.global_pool(x).squeeze(-1)  # [B, 240]
        return self.fc(x)


class CDCN_KAN(nn.Module):
    def __init__(self, num_classes=4, in_channels=32, time_points=128):
        super().__init__()

        # Initial convolution
        self.conv1 = nn.Conv1d(in_channels, 24, kernel_size=5, stride=1, padding=2)

        # Dense Block 1
        self.dense1 = DenseBlock(24, 6, 12)  # 24 + 6*12 = 96 channels
        self.trans1 = TransitionBlock(96, 96)

        # Dense Block 2
        self.dense2 = DenseBlock(96, 6, 12)  # 96 + 6*12 = 168 channels
        self.trans2 = TransitionBlock(168, 168)

        # Dense Block 3
        self.dense3 = DenseBlock(168, 6, 12)  # 168 + 6*12 = 240 channels

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # KAN вместо FC слоя
        self.kan = KAN(width=[240, num_classes], grid=3, k=3)  # Простейшая конфигурация KAN

    def forward(self, x):
        # Input: [B, 1, 32, 128]
        x = x.squeeze(1)  # [B, 32, 128]

        # Initial conv
        x = self.conv1(x)  # [B, 24, 128]

        # Dense Block 1
        x = self.dense1(x)  # [B, 96, 128]
        x = self.trans1(x)  # [B, 96, 64]

        # Dense Block 2
        x = self.dense2(x)  # [B, 168, 64]
        x = self.trans2(x)  # [B, 168, 32]

        # Dense Block 3
        x = self.dense3(x)  # [B, 240, 32]

        # Global pooling
        x = self.global_pool(x)  # [B, 240, 1]
        x = x.squeeze(-1)  # [B, 240]

        # KAN вместо FC
        x = self.kan(x)  # [B, num_classes]
        return x

class HybridCDCN(nn.Module):
    def __init__(self, num_classes=4, in_channels=32, time_points=128,
                 kan_dims=(240, 128, 64), n_qubits=6, q_depth=2):
        super().__init__()

        # Основная CNN ветвь
        self.conv1 = nn.Conv1d(in_channels, 24, kernel_size=5, padding=2)
        self.dense1 = DenseBlock(24, 6, 12)
        self.trans1 = TransitionBlock(96, 96)
        self.dense2 = DenseBlock(96, 6, 12)
        self.trans2 = TransitionBlock(168, 168)
        self.dense3 = DenseBlock(168, 6, 12)  # output: [B, 240, 32]

        # KAN ветвь
        self.kan_pool = nn.AdaptiveAvgPool1d(1)
        self.kan_flatten = nn.Flatten()
        self.kan = KAN(width=list(kan_dims), grid=3, k=3)
        self.kan_proj = nn.Linear(kan_dims[-1], 32)  # Фиксированный выход 32

        # Квантовая часть
        self.n_qubits = n_qubits  # Сохраняем оригинальное значение 6
        self.q_depth = q_depth

        # Препроцессинг для VQC
        self.vqc_preprocess = nn.Sequential(
            nn.Linear(240, 64),
            nn.GELU(),
            nn.Linear(64, self.n_qubits),  # Выход = n_qubits (6)
            nn.Tanh()
        )

        # Квантовые параметры
        self.q_weights = nn.Parameter(torch.randn(q_depth, n_qubits, 3) * 0.1)

        # Квантовое устройство
        dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(dev, interface='torch', diff_method='parameter-shift')
        def quantum_circuit(inputs, weights):
            inputs = (inputs + 1) * math.pi / 2

            # Кодирование данных
            for i in range(self.n_qubits):
                qml.RY(inputs[i], wires=i)

            # Вариационные слои
            for layer in weights:
                for qubit in range(self.n_qubits):
                    qml.Rot(*layer[qubit], wires=qubit)
                for qubit in range(self.n_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])

            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.qlayer = quantum_circuit

        # Финальный классификатор (32 + 6 = 38 входных признаков)
        self.final_fc = nn.Linear(32 + n_qubits, num_classes)

    def forward(self, x):
        # Проверка входа
        if x.dim() != 4:
            raise ValueError(f"Input must be 4D [B,1,C,T], got {x.shape}")

        # CNN обработка
        x = x.squeeze(1)
        x = self.conv1(x)
        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.trans2(x)
        x = self.dense3(x)  # [B,240,32]

        # Подготовка признаков
        cnn_features = self.kan_pool(x).squeeze(-1)  # [B,240]

        # KAN ветвь
        kan_features = self.kan(cnn_features)
        kan_features = self.kan_proj(kan_features)  # [B,32]

        # Квантовая ветвь
        vqc_input = self.vqc_preprocess(cnn_features)  # [B,6]

        # Обработка батча через VQC
        quantum_features = []
        for i in range(vqc_input.shape[0]):
            q_out = self.qlayer(vqc_input[i], self.q_weights)
            q_out = torch.tensor(q_out, dtype=torch.float32, device=x.device)
            quantum_features.append(q_out)

        quantum_features = torch.stack(quantum_features)  # [B,6]

        # Объединение и классификация
        combined = torch.cat([kan_features, quantum_features], dim=1)  # [B,38]
        return self.final_fc(combined)


class CDCN_KAN2(nn.Module):
    def __init__(self, num_classes=4, in_channels=32, time_points=128,
                 kan_dims=(240, 128, 64)):
        super().__init__()

        # Основная CNN ветвь
        self.conv1 = nn.Conv1d(in_channels, 24, kernel_size=5, padding=2)
        self.dense1 = DenseBlock(24, 6, 12)
        self.trans1 = TransitionBlock(96, 96)
        self.dense2 = DenseBlock(96, 6, 12)
        self.trans2 = TransitionBlock(168, 168)
        self.dense3 = DenseBlock(168, 6, 12)  # output: [B, 240, 32]

        # KAN ветвь
        self.kan_pool = nn.AdaptiveAvgPool1d(1)
        self.kan_flatten = nn.Flatten()

        # Инициализация KAN
        self.kan = KAN(
            width=list(kan_dims),
            grid=3,
            k=3,
            grid_range=[-1.5, 1.5],
            noise_scale=0.1,
            base_fun=torch.nn.SiLU()
        )

        # Финальный классификатор
        self.final_fc = nn.Linear(kan_dims[-1], num_classes)  # 64 -> num_classes

    def forward(self, x):
        # Проверка входа
        if x.dim() != 4:
            raise ValueError(f"Input must be 4D [B,1,C,T], got {x.shape}")

        # CNN обработка
        x = x.squeeze(1)  # [B, C, T]
        x = self.conv1(x)
        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.trans2(x)
        x = self.dense3(x)  # [B, 240, 32]

        # Подготовка признаков для KAN
        cnn_features = self.kan_pool(x).squeeze(-1)  # [B, 240]

        # KAN обработка
        kan_features = self.kan(cnn_features)  # [B, 64]

        # Классификация
        return self.final_fc(kan_features)