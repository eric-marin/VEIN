import torch, torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

class Iris_MLP(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )
    def forward(self, x):
        return self.layers(x)

class Iris_Linear(nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.fc = nn.Linear(4, 3)
        self.fc.weight.data = weight
        self.fc.bias.data = bias
    def forward(self, x):
        return self.fc(x)

iris = load_iris()
scaler = StandardScaler()
X = scaler.fit_transform(iris.data).astype('float32') # pyright: ignore
y = iris.target.astype('int64') # pyright: ignore

dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
trainloader = DataLoader(dataset, batch_size=16, shuffle=True)

def train_model(name: str, dim):
    net = Iris_MLP(hidden_dim=dim)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

    print(f"Training {name} ({dim} neurons)...")
    for epoch in range(100):
        global loss
        for data in trainloader:
            inputs, targets = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}, Loss: {loss.item():.4f}")
    return net

if __name__ == "__main__":
    torch_net_a = train_model("Base Network", 10).eval()

    with torch.no_grad():
        torch_net_a.layers[0].weight.fill_(0.1) # pyright: ignore
        torch_net_a.layers[0].bias.fill_(10.0) # pyright: ignore

        W1 = torch_net_a.layers[0].weight.data
        b1 = torch_net_a.layers[0].bias.data
        W2 = torch_net_a.layers[2].weight.data
        b2 = torch_net_a.layers[2].bias.data

        W_collapsed = torch.matmul(W2, W1) # pyright: ignore
        b_collapsed = torch.matmul(W2, b1) + b2 # pyright: ignore

        torch_net_b = Iris_Linear(W_collapsed, b_collapsed).eval()

    torch.onnx.export(torch_net_a, (torch.randn(1, 4),), "iris_a.onnx")
    torch.onnx.export(torch_net_b, (torch.randn(1, 4),), "iris_b.onnx")
