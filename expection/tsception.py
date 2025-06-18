import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Многошкальная временная свёртка
class DynamicTemporalLayer(nn.Module):
    def __init__(self, in_channels=1, fs=128, kernel_scales=[0.5, 0.25, 0.125]):
        super().__init__()
        self.kernels = nn.ModuleList()
        for scale in kernel_scales:
            kernel_size = int(fs * scale)
            self.kernels.append(
                nn.Conv1d(in_channels, 1, kernel_size=kernel_size, padding=0)
            )

    def forward(self, x):
        # x: [B, 1, T]
        outputs = []
        for kernel in self.kernels:
            out = kernel(x)  # [B, 1, T']
            out = F.interpolate(out, size=x.shape[-1], mode='linear', align_corners=False)  # восстановление времени
            outputs.append(out)
        return torch.cat(outputs, dim=1)  # [B, K, T], где K — число шкал


# 2. Асимметричная пространственная обработка
class AsymmetricSpatialLayer(nn.Module):
    def __init__(self, in_channels, left_indices, right_indices, central_indices=None, depth_multiplier=2):
        super().__init__()
        self.left_indices = left_indices
        self.right_indices = right_indices
        self.central_indices = central_indices if central_indices else []

        # Левые и правые ветки
        self.left_conv = self._make_branch(len(left_indices), in_channels, depth_multiplier)
        self.right_conv = self._make_branch(len(right_indices), in_channels, depth_multiplier)

        # Центральные каналы (опционально)
        if self.central_indices:
            self.central_conv = self._make_branch(len(central_indices), in_channels, depth_multiplier)

    def _make_branch(self, num_channels, in_channels, depth_multiplier):
        return nn.Sequential(
            nn.BatchNorm1d(num_channels),
            nn.Conv1d(num_channels, num_channels * depth_multiplier, kernel_size=1, groups=num_channels),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        # x: [B*K, C, T]
        B_K, C, T = x.shape
        x = x.view(-1, C, T)  # [B, K, C, T] → [B*K, C, T]

        # Разделение на левые, правые и центральные каналы
        left_x = x[:, self.left_indices, :]   # [B*K, L, T]
        right_x = x[:, self.right_indices, :] # [B*K, R, T]

        # Обработка
        left_feat = self.left_conv(left_x).squeeze(-1)  # [B*K, L*dm]
        right_feat = self.right_conv(right_x).squeeze(-1)  # [B*K, R*dm]

        feats = [left_feat, right_feat]
        if self.central_indices:
            central_x = x[:, self.central_indices, :]
            central_feat = self.central_conv(central_x).squeeze(-1)
            feats.append(central_feat)

        # Объединение
        fused = torch.cat(feats, dim=1)  # [B*K, D]
        return fused


# 3. High-Level Fusion + Classifier
class HighLevelFusionLayer(nn.Module):
    def __init__(self, in_features, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return self.fusion(x)


# 4. Полная модель TSception
class TSception(nn.Module):
    def __init__(
        self,
        input_size=(32, 600),  # (C, T)
        fs=128,
        tasks=['valence', 'arousal'],
        left_indices=None,
        right_indices=None,
        central_indices=None,
        depth_multiplier=2,
        hidden_dim=128
    ):
        super().__init__()
        C, T = input_size

        # Пример индексов для DEAP (можно заменить под SEED)
        if left_indices is None:
            left_indices = list(range(0, 16))  # первая половина — левое полушарие
        if right_indices is None:
            right_indices = list(range(16, 32))  # вторая половина — правое
        if central_indices is None:
            central_indices = []  # опционально

        # === Слои ===
        self.temporal = DynamicTemporalLayer(in_channels=1, fs=fs)
        self.K = len(self.temporal.kernels)  # Теперь K сохраняется как атрибут

        # Для каждого канала отдельно
        self.spatial_blocks = nn.ModuleList([
            AsymmetricSpatialLayer(
                self.K, left_indices, right_indices, central_indices, depth_multiplier
            ) for _ in range(C)
        ])

        spatial_in = self.K * depth_multiplier * (
            len(left_indices) + len(right_indices) + len(central_indices)
        )
        self.fusion = HighLevelFusionLayer(spatial_in, hidden_dim=hidden_dim)

        # === Мультитасковый классификатор ===
        self.tasks = tasks
        for task in tasks:
            setattr(self, task, nn.Sequential(
                nn.Linear(hidden_dim, 2)  # бинарная классификация
            ))

    def forward(self, x):
        # x: [B, C, T]
        B, C, T = x.shape
        x = x.unsqueeze(1)  # [B, 1, C, T]

        # === Динамическая временная обработка ===
        temporal_outs = []
        for i in range(C):
            xi = x[:, :, i, :]  # [B, 1, T]
            xi = self.temporal(xi)  # [B, K, T]
            temporal_outs.append(xi.unsqueeze(1))  # [B, 1, K, T]
        x = torch.cat(temporal_outs, dim=1)  # [B, C, K, T]

        # === Асимметричная пространственная обработка ===
        x = x.transpose(1, 2)  # [B, K, C, T]
        x = x.contiguous().view(B * self.K, C, T)  # [B*K, C, T]

        spatial_outs = []
        for c in range(C):
            xc = x[:, c:c+1, :]  # [B*K, 1, T]
            xc = self.spatial_blocks[c](xc)
            spatial_outs.append(xc.unsqueeze(1))  # [B*K, 1, D]
        x = torch.cat(spatial_outs, dim=1)  # [B*K, C, D]

        x = x.mean(dim=1)  # [B*K, D]
        x = x.view(B, -1)  # [B, K*D]

        # === High-level fusion ===
        x = self.fusion(x)

        # === Классификация по задачам ===
        return {task: getattr(self, task)(x) for task in self.tasks}