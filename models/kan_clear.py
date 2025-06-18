
import torch.nn as nn
from kan import KAN

from torch.optim.swa_utils import update_bn, AveragedModel



class EEG_KAN(nn.Module):
    def __init__(self, input_dim=310, num_classes=3):
        super().__init__()
        self.kan = KAN(
            width=[input_dim, 168, 64, 168, num_classes],  # Архитектура сети
            grid=5,  # Количество точек сетки для базисных функций
            k=3,  # Порядок B-сплайнов
            seed=42  # Для воспроизводимости
        )

        # Инициализация весов
        self._init_weights()

    def _init_weights(self):
        for name, param in self.kan.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x):
        # Преобразование входных данных: (batch, 62, 5) -> (batch, 310)
        x = x.view(x.size(0), -1)
        x = self.kan(x)
        return x