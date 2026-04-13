import torch, torch.nn as nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms

class MNIST_MLP(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(784, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10),
        )
    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)

class MNIST_Linear(nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(784, 10)
        self.fc.weight.data = weight
        self.fc.bias.data = bias
    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)

train_dataset = MNIST('./', download=True, transform=transforms.ToTensor(), train=True)
trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

def train_model(name: str, dim):
    net = MNIST_MLP(hidden_dim=dim)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.5e-4)

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
        print(f"  Epoch {epoch+1}, Loss: {loss.item():.4f}")
    return net

if __name__ == "__main__":
    torch_net_a = train_model("Base Network", 6).eval()

    with torch.no_grad():
        torch_net_a.layers[0].weight.fill_(0.01) # pyright: ignore
        torch_net_a.layers[0].bias.fill_(5.0) # pyright: ignore

        W1 = torch_net_a.layers[0].weight.data
        b1 = torch_net_a.layers[0].bias.data
        W2 = torch_net_a.layers[2].weight.data
        b2 = torch_net_a.layers[2].bias.data

        W_collapsed = torch.matmul(W2, W1) # pyright: ignore
        b_collapsed = torch.matmul(W2, b1) + b2 # pyright: ignore

        torch_net_b = MNIST_Linear(W_collapsed, b_collapsed).eval()

    torch.onnx.export(torch_net_a, (torch.randn(1, 28, 28),), "mnist_a.onnx")
    torch.onnx.export(torch_net_b, (torch.randn(1, 28, 28),), "mnist_b.onnx")
