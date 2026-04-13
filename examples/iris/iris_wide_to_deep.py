import torch, torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

class Wide_Iris_MLP(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )
    def forward(self, x):
        return self.layers(x)

class Deep_Block(nn.Module):
    def __init__(self, w1, b1, w2):
        super().__init__()
        self.fc1 = nn.Linear(4, 1)
        self.fc1.weight.data = w1.clone().unsqueeze(0)
        self.fc1.bias.data = b1.clone().unsqueeze(0)
        self.fc2 = nn.Linear(1, 3, bias=False)
        self.fc2.weight.data = w2.clone().unsqueeze(1)
        
    def forward(self, x_s):
        x, s = x_s
        z = torch.relu(self.fc1(x))
        ds = self.fc2(z)
        return (x, s + ds)

class Deep_Iris_MLP(nn.Module):
    def __init__(self, wide_net):
        super().__init__()
        w1 = wide_net.layers[0].weight.data
        b1 = wide_net.layers[0].bias.data
        w2 = wide_net.layers[2].weight.data
        b2 = wide_net.layers[2].bias.data
        num_neurons = w1.shape[0]
        
        self.blocks = nn.Sequential(*[
            Deep_Block(w1[j], b1[j], w2[:, j]) for j in range(num_neurons)
        ])
        self.final_bias = nn.Parameter(b2.clone())

    def forward(self, x):
        s = torch.zeros(x.shape[0], 3, device=x.device)
        _, final_s = self.blocks((x, s))
        return final_s + self.final_bias

iris = load_iris()
scaler = StandardScaler()
X = scaler.fit_transform(iris.data).astype('float32') # pyright: ignore
y = iris.target.astype('int64') # pyright: ignore

dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
trainloader = DataLoader(dataset, batch_size=16, shuffle=True)

def train_model(name:str, dim):
    net = Wide_Iris_MLP(hidden_dim=dim)
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
    torch_net_a = train_model("Wide Network", 10).eval()
    torch_net_b = Deep_Iris_MLP(torch_net_a).eval()
    
    torch.onnx.export(torch_net_a, (torch.randn(1, 4),), "iris_a.onnx")
    torch.onnx.export(torch_net_b, (torch.randn(1, 4),), "iris_b.onnx")
