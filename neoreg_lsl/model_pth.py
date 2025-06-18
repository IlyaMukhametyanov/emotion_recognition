import torch

# Загружаем чекпоинт
state_dict = torch.load("model.pth")

# Выводим все ключи
for key in state_dict.keys():
    print(key)