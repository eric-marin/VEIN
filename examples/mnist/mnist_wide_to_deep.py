import torch, torch.nn as nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms

class Wide_MNIST_MLP(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10),
        )
    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)

class Deep_MNIST_Block(nn.Module):
    def __init__(self, w1, b1, w2):
        super().__init__()
        self.fc1 = nn.Linear(784, 1)
        self.fc1.weight.data = w1.clone().unsqueeze(0)
        self.fc1.bias.data = b1.clone().unsqueeze(0)
        self.fc2 = nn.Linear(1, 10, bias=False)
        self.fc2.weight.data = w2.clone().unsqueeze(1)
        
    def forward(self, x_s):
        x, s = x_s
        z = torch.relu(self.fc1(x))
        ds = self.fc2(z)
        return (x, s + ds)

class Deep_MNIST_MLP(nn.Module):
    def __init__(self, wide_net):
        super().__init__()
        self.flatten = nn.Flatten()
        
        w1 = wide_net.layers[0].weight.data
        b1 = wide_net.layers[0].bias.data
        w2 = wide_net.layers[2].weight.data
        b2 = wide_net.layers[2].bias.data
        num_neurons = w1.shape[0]
        
        self.blocks = nn.Sequential(*[
            Deep_MNIST_Block(w1[j], b1[j], w2[:, j]) for j in range(num_neurons)
        ])
        self.final_bias = nn.Parameter(b2.clone())

    def forward(self, x):
        x = self.flatten(x)
        s = torch.zeros(x.shape[0], 10, device=x.device)
        _, final_s = self.blocks((x, s))
        return final_s + self.final_bias

train_dataset = MNIST('./', download=True, transform=transforms.ToTensor(), train=True)
trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

def train_model(name: str, dim):
    net = Wide_MNIST_MLP(hidden_dim=dim)
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
    torch_net_a = train_model("Wide Network", 8).eval()
    torch_net_b = Deep_MNIST_MLP(torch_net_a).eval()

    torch.onnx.export(torch_net_a, (torch.randn(1, 28, 28),), "mnist_a.onnx")
    torch.onnx.export(torch_net_b, (torch.randn(1, 28, 28),), "mnist_b.onnx")
