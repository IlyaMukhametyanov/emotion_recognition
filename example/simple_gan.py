import torch
from torch.nn import Linear, ReLU
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


# Рёбра графа: [2, E]
edge_index = torch.tensor([
    [0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
    [0, 2], [1, 3], [2, 4], [3, 5]
], dtype=torch.long).t().contiguous()

# Признаки узлов: [num_nodes, num_features]
x = torch.tensor([
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1],
    [1, 1, 0, 0, 0]
], dtype=torch.float)

# Целевые метки: [num_nodes]
y = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)

# Объединяем в объект Data
data = Data(x=x, edge_index=edge_index, y=y)

class GNN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


# Инициализация модели, оптимизатора
model = GNN(num_features=5, hidden_dim=16, num_classes=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

# Тренировка
model.train()
for epoch in range(150):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out, data.y)
    acc = (out.argmax(dim=1) == data.y).float().mean()

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:03d} | Loss: {loss.item():.4f} | Accuracy: {acc.item():.4f}")

model.eval()
pred = model(data).argmax(dim=1)

print("\nPredicted classes:", pred.tolist())
print("True labels:       ", data.y.tolist())