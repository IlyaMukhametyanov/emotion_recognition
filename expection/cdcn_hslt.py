import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict


# 1. Dense Layer и Dense Block
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
        self.down = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.GELU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.AvgPool1d(kernel_size=2, stride=2)  # Уменьшает T вдвое
        )

    def forward(self, x):
        return self.down(x)


# 2. Regional Transformer (исправленная версия)
class RegionalTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [E_region, B, F]
        residual = x
        x, _ = self.attn(x, x, x)  # [E_region, B, F]
        x = self.norm1(residual + x)
        x = self.norm2(x + self.ffn(x))
        return x  # [E_region, B, F] — не делаем mean, оставляем внимание


# 3. Hybrid Regional Block (исправленная версия)
class HybridRegionalBlock(nn.Module):
    def __init__(self, input_features, growth_rate=12, num_layers=6, embed_dim=64, num_heads=4):
        super().__init__()
        # 1. DenseBlock — усиление корреляций между электродами
        self.dense_block = DenseBlock(input_features, num_layers, growth_rate)
        current_channels = input_features + growth_rate * num_layers

        # 2. TransitionBlock — снижение размерности
        self.transition = TransitionBlock(current_channels, int(current_channels // 2))
        self.output_channels = int(current_channels // 2)

        # 3. Проекция в пространство Transformer
        self.projection = nn.Linear(self.output_channels, embed_dim)

        # 4. Regional Transformer
        self.region_transformer = RegionalTransformer(embed_dim, num_heads=num_heads)

        # 5. Итоговая проекция региональных признаков
        self.final_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        x: [B, E_region, F] → batch, число каналов в регионе, частотные полосы
        """
        B, E, F = x.shape
        x = x.transpose(1, 2)  # -> [B, F, E]

        # 1. DenseBlock
        x = self.dense_block(x)  # [B, C_new, E]

        # 2. TransitionBlock
        x = self.transition(x)  # [B, C_reduced, E_reduced]

        # 3. Проекция в пространство Transformer
        x = x.permute(2, 0, 1)  # -> [E_reduced, B, C_reduced]
        x = self.projection(x)  # -> [E_reduced, B, embed_dim]

        # 4. Transformer для региона
        x = self.region_transformer(x)  # [E_reduced, B, embed_dim]

        # 5. Global Pooling + Final Projection
        x = x.mean(dim=0)  # [B, embed_dim]
        x = self.final_proj(x)  # [B, embed_dim]
        return x


# 4. Global Transformer (остаётся как был, но лучше контролирует вход)
class GlobalTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.global_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, regional_features):
        # regional_features: list of [B, embed_dim]
        x = torch.stack(regional_features, dim=0)  # [R, B, embed_dim]
        x = x.transpose(0, 1)  # [B, R, embed_dim]
        x, _ = self.global_attn(x, x, x)  # [B, R, embed_dim]
        x = self.norm1(x)
        x = self.norm2(x + self.ffn(x))
        return x.mean(dim=1)  # [B, embed_dim]


# 5. Мультитасковая голова
class TaskHead(nn.Module):
    def __init__(self, in_features, out_classes=2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, out_classes)
        )

    def forward(self, x):
        return self.head(x)

class R2G_CDCN_Hybrid(nn.Module):
    def __init__(
        self,
        input_shape=(32, 5),  # (E, F)
        tasks=['valence', 'arousal'],
        brain_regions: Dict[str, List[int]] = None,
        embed_dim: int = 64,
        growth_rate: int = 12,
        num_heads: int = 4
    ):
        super().__init__()
        E, F = input_shape

        if brain_regions is None:
            brain_regions = {
                'frontal': list(range(0, 14)),
                'central': list(range(14, 24)),
                'temporal': list(range(24, 32)),
            }
        self.brain_regions = brain_regions
        self.region_names = list(brain_regions.keys())
        self.tasks = tasks  # <-- добавляем эту строку

        # === Гибридные региональные блоки ===
        for region in self.region_names:
            setattr(self, f'rst_{region}', HybridRegionalBlock(
                input_features=F,
                growth_rate=growth_rate,
                num_layers=6,
                embed_dim=embed_dim,
                num_heads=num_heads
            ))

        # === Global Transformer ===
        self.global_transformer = GlobalTransformer(embed_dim, num_heads=num_heads)

        # === Мультитасковые головы ===
        head_in_features = embed_dim
        for task in self.tasks:  # <-- корректное использование self.tasks
            setattr(self, task, TaskHead(head_in_features))

    def forward(self, x):
        B, E, F = x.shape

        # === Разбиение по регионам и обработка ===
        regional_features = []
        for region_name in self.region_names:
            indices = self.brain_regions[region_name]
            region_x = x[:, indices, :]  # [B, E_region, F]
            region_feat = getattr(self, f'rst_{region_name}')(region_x)  # [B, embed_dim]
            regional_features.append(region_feat)

        # === Глобальная обработка через Transformer ===
        global_feat = self.global_transformer(regional_features)  # [B, embed_dim]

        # === Классификация ===
        outputs = {}
        for task in self.tasks:  # <-- здесь всё работает, т.к. есть self.tasks
            outputs[task] = getattr(self, task)(global_feat)

        return outputs