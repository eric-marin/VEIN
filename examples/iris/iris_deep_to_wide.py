import torch, torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

class Deep_Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 1)
        self.fc2 = nn.Linear(1, 3, bias=False)
        
    def forward(self, x_s):
        x, s = x_s
        z = torch.relu(self.fc1(x))
        ds = self.fc2(z)
        return (x, s + ds)

class Deep_Iris_MLP(nn.Module):
    def __init__(self, num_blocks):
        super().__init__()
        self.blocks = nn.Sequential(*[Deep_Block() for _ in range(num_blocks)])
        self.final_bias = nn.Parameter(torch.zeros(3))

    def forward(self, x):
        s = torch.zeros(x.shape[0], 3, device=x.device)
        _, final_s = self.blocks((x, s))
        return final_s + self.final_bias

class Wide_Iris_MLP(nn.Module):
    def __init__(self, deep_net):
        super().__init__()
        num_neurons = len(deep_net.blocks)
        self.layers = nn.Sequential(
            nn.Linear(4, num_neurons),
            nn.ReLU(),
            nn.Linear(num_neurons, 3),
        )
        with torch.no_grad():
            w1_all = []
            b1_all = []
            w2_all = []
            
            for block in deep_net.blocks:
                w1_all.append(block.fc1.weight.data)
                b1_all.append(block.fc1.bias.data)
                w2_all.append(block.fc2.weight.data)
                
            self.layers[0].weight.copy_(torch.cat(w1_all, dim=0)) # pyright: ignore
            self.layers[0].bias.copy_(torch.cat(b1_all, dim=0)) # pyright: ignore
            self.layers[2].weight.copy_(torch.cat(w2_all, dim=1)) # pyright: ignore
            self.layers[2].bias.copy_(deep_net.final_bias.data) # pyright: ignore

    def forward(self, x):
        return self.layers(x)

iris = load_iris()
scaler = StandardScaler()
X = scaler.fit_transform(iris.data).astype('float32') # pyright: ignore
y = iris.target.astype('int64') # pyright: ignore

dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
trainloader = DataLoader(dataset, batch_size=16, shuffle=True)

def train_deep_model(name: str, num_blocks):
    net = Deep_Iris_MLP(num_blocks=num_blocks)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

    print(f"Training {name} ({num_blocks} blocks)...")
    for epoch in range(100):
        global loss
        for inputs, targets in trainloader:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}, Loss: {loss.item():.4f}")
    return net

if __name__ == "__main__":
    torch_net_a = train_deep_model("Deep Network", 10).eval()
    torch_net_b = Wide_Iris_MLP(torch_net_a).eval()
    
    torch.onnx.export(torch_net_a, (torch.randn(1, 4),), "iris_a.onnx")
    torch.onnx.export(torch_net_b, (torch.randn(1, 4),), "iris_b.onnx")
