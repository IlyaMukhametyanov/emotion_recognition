from kan import *


class CDCN_KAN(nn.Module):
    def __init__(self, num_classes=4, in_channels=32, time_points=128):
        super().__init__()

        # Initial KAN preprocessing
        self.kan = KAN(width=[in_channels, 64, 24], grid=5, k=3)  # Обрабатывает каждый канал отдельно
        self.kan_proj = nn.Conv1d(24, 24, kernel_size=1)  # Исправлено!

        # Оригинальная CDCN архитектура
        self.conv1 = nn.Conv1d(24, 24, kernel_size=5, padding=2)

        # Dense Blocks
        self.dense1 = DenseBlock(24, 6, 12)
        self.trans1 = TransitionBlock(96, 96)

        self.dense2 = DenseBlock(96, 6, 12)
        self.trans2 = TransitionBlock(168, 168)

        self.dense3 = DenseBlock(168, 6, 12)

        # Classification
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(240, num_classes)

    def forward(self, x):
        # Input: [B, 1, 32, 128]
        x = x.squeeze(1)  # [B, 32, 128]

        B, C, T = x.shape
        # KAN processing
        x = x.permute(0, 2, 1)  # [B, T, C]
        x = x.reshape(B * T, C)  # [B*T, C]
        x = self.kan(x)  # [B*T, 24]
        x = x.view(B, T, 24)  # [B, T, 24]
        x = x.permute(0, 2, 1)  # [B, 24, T]
        x = self.kan_proj(x)  # [B, 24, T=128]

        # Оригинальный CDCN processing
        x = self.conv1(x)
        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.trans2(x)
        x = self.dense3(x)

        x = self.global_pool(x).squeeze(-1)
        return self.fc(x)


# Оригинальные компоненты CDCN без изменений
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

        # Classification
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(240, num_classes)

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

        # Global pooling and classification
        x = self.global_pool(x)  # [B, 240, 1]
        x = x.squeeze(-1)  # [B, 240]
        return self.fc(x)

class CDCN_KAN_Light(nn.Module):
    def __init__(self, num_classes=4, in_channels=32, time_points=128):
        super().__init__()

        # Initial KAN preprocessing (сохраняем оригинальный KAN блок)
        self.kan = KAN(width=[in_channels, 32, 16], grid=3, k=3)  # Уменьшил размеры
        self.kan_proj = nn.Conv1d(16, 16, kernel_size=1)  # Уменьшил выходные каналы

        # Уменьшенная CDCN архитектура
        self.conv1_light = nn.Conv1d(16, 16, kernel_size=5, padding=2)  # Добавил _light

        # Dense Blocks с меньшим количеством слоев и каналов
        self.dense1_light = DenseBlock_Light(16, 4, 8)  # Было (24,6,12)
        self.trans1_light = TransitionBlock_Light(16 + 4 * 8, 64)  # Было 96

        self.dense2_light = DenseBlock_Light(64, 4, 8)  # Было (96,6,12)
        self.trans2_light = TransitionBlock_Light(64 + 4 * 8, 96)  # Было 168

        self.dense3_light = DenseBlock_Light(96, 4, 8)  # Было (168,6,12)

        # Classification
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_light = nn.Linear(96 + 4 * 8, num_classes)  # Было 240

    def forward(self, x):
        # Input: [B, 1, 32, 128]
        x = x.squeeze(1)  # [B, 32, 128]

        B, C, T = x.shape
        # KAN processing (без изменений)
        x = x.permute(0, 2, 1)  # [B, T, C]
        x = x.reshape(B * T, C)  # [B*T, C]
        x = self.kan(x)  # [B*T, 16]
        x = x.view(B, T, 16)  # [B, T, 16]
        x = x.permute(0, 2, 1)  # [B, 16, T]
        x = self.kan_proj(x)  # [B, 16, T=128]

        # CDCN processing
        x = self.conv1_light(x)
        x = self.dense1_light(x)
        x = self.trans1_light(x)
        x = self.dense2_light(x)
        x = self.trans2_light(x)
        x = self.dense3_light(x)

        x = self.global_pool(x).squeeze(-1)
        return self.fc_light(x)


# Облегченные версии компонентов
class DenseLayer_Light(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.GELU(),
            nn.Conv1d(in_channels, growth_rate, kernel_size=3, padding=1, stride=1)
        )

    def forward(self, x):
        return torch.cat([x, self.conv(x)], dim=1)


class DenseBlock_Light(nn.Module):
    def __init__(self, in_channels, num_layers, growth_rate):
        super().__init__()
        self.layers = nn.ModuleList()
        channels = in_channels
        for _ in range(num_layers):
            self.layers.append(DenseLayer_Light(channels, growth_rate))  # Используем Light версию
            channels += growth_rate

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransitionBlock_Light(nn.Module):
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
