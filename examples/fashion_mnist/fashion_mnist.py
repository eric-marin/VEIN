import torch, torch.nn as nn
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from torchvision import transforms

class FashionMNIST_MLP(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10),
        )
    def forward(self, x):
        return self.layers(x)

train_dataset = FashionMNIST('./', download=True, transform=transforms.ToTensor(), train=True)
trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

def train_model(name: str, dim):
    net = FashionMNIST_MLP(hidden_dim=dim)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

    print(f"Training {name} ({dim} neurons)...")
    for epoch in range(10):
        global loss
        for data in trainloader:
            inputs, targets = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 1 == 0:
            print(f"  Epoch {epoch+1}, Loss: {loss.item():.4f}")
    return net

if __name__ == "__main__":
    torch_net_a = train_model("Network A", 28).eval()
    torch_net_b = train_model("Network B", 56).eval()

    torch.onnx.export(torch_net_a, (torch.randn(1, 28, 28),), "fashion_mnist_a.onnx")
    torch.onnx.export(torch_net_b, (torch.randn(1, 28, 28),), "fashion_mnist_b.onnx")
