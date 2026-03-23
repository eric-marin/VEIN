import torch, torch.nn as nn
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from torchvision import transforms

class FashionMNIST_MLP(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6 * 6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
    def forward(self, x):
        return self.layers(x)

transform = transforms.Compose([
    transforms.Resize((6, 6)),
    transforms.ToTensor(),
])

train_dataset = FashionMNIST('./', download=True, transform=transform, train=True)
tshirts_trousers = [id for id, data in enumerate(train_dataset.targets) if data.item() == 0 or data.item() == 1]
train_dataset = torch.utils.data.Subset(train_dataset, tshirts_trousers)

trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

def train_model(name: str, dim):
    net = FashionMNIST_MLP(hidden_dim=dim)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

    print(f"Training {name}...")
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
    torch_net_a = train_model("Network A", 6).eval()
    torch_net_b = train_model("Network B", 12).eval()

    torch.onnx.export(torch_net_a, (torch.randn(1, 1, 6, 6),), "fashion_mnist_a.onnx")
    torch.onnx.export(torch_net_b, (torch.randn(1, 1, 6, 6),), "fashion_mnist_b.onnx")
