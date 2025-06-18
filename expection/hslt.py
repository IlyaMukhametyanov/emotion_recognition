import torch
import torch.nn as nn
from typing import List, Dict

class RegionalSpatialTransformer(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4):
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
        # x: [T, E_region, F]
        residual = x
        x = x.transpose(0, 1)  # -> [E_region, T, F]
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(attn_out + residual.transpose(0, 1))
        x = self.norm2(x + self.ffn(x))
        return x.mean(dim=0)  # -> [T, F]

class RegionalTemporalTransformer(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4):
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
        # x: [T, F]
        x = x.unsqueeze(0)  # -> [1, T, F]
        x, _ = self.attn(x, x, x)
        x = self.norm1(x)
        x = self.norm2(x + self.ffn(x))
        return x.squeeze(0)  # -> [T, F]

class GlobalSpatialTemporalTransformer(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, regional_features):
        # regional_features: list of [T, F] from each region
        x = torch.stack(regional_features, dim=0)  # -> [R, T, F]
        x = x.transpose(0, 1)  # -> [T, R, F]
        x = x.reshape(x.shape[0], -1)  # -> [T, R*F]
        x = self.norm1(x)
        x = self.norm2(x + self.ffn(x))
        return x.mean(dim=0)  # -> [R*F]

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


class R2GSTLT(nn.Module):
    def __init__(
        self,
        input_shape=(32, 5),  # (E, F)
        tasks: List[str] = ['valence', 'arousal'],
        brain_regions: Dict[str, List[int]] = None,
        embed_dim: int = 64,
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
        self.embed_dim = embed_dim

        # Embedding layer
        self.embedding = nn.Linear(F, embed_dim)

        # Regional Spatial Transformers
        for region in self.region_names:
            setattr(self, f'spatial_{region}', RegionalSpatialTransformer(embed_dim, num_heads))

        # Regional Temporal Transformers
        for region in self.region_names:
            setattr(self, f'temporal_{region}', RegionalTemporalTransformer(embed_dim, num_heads))

        # Global Spatial–Temporal Transformer
        self.global_transformer = GlobalSpatialTemporalTransformer(embed_dim, num_heads)

        # Task heads
        head_input_size = len(self.region_names) * embed_dim
        for task in tasks:
            setattr(self, task, TaskHead(head_input_size, out_classes=2))

    def forward(self, x):
        B, E, F = x.shape

        # === Шаг 1: Эмбеддинг DE-признаков ===
        x_embedded = self.embedding(x)  # [B, E, embed_dim]

        # === Шаг 2: Региональная обработка (RST-Trans) ===
        regional_features = []
        for region_name in self.region_names:
            indices = self.brain_regions[region_name]
            if not indices:
                continue
            region_x = x_embedded[:, indices, :]  # [B, E_region, embed_dim]

            # Своя обработка на уровне региона
            spatial_block = getattr(self, f'spatial_{region_name}')
            temporal_block = getattr(self, f'temporal_{region_name}')

            region_x = region_x.transpose(1, 2).contiguous()  # [B, embed_dim, E_region]
            region_x = region_x.permute(2, 0, 1)  # [E_region, B, embed_dim]

            region_feat = spatial_block(region_x)  # [B, embed_dim]
            region_feat = temporal_block(region_feat)  # [B, embed_dim]
            regional_features.append(region_feat)

        # === Шаг 3: Глобальная обработка (GST-Trans) ===
        global_feat = torch.cat(regional_features, dim=1)  # [B, R * embed_dim]

        # === Шаг 4: Мультитасковые головы ===
        outputs = {}
        for task in self.tasks:
            outputs[task] = getattr(self, task)(global_feat)

        return outputs