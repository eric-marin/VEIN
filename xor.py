import torch
import torch.nn as nn
import nneq

class xor_mlp(nn.Module):
    def __init__(self, hidden_dim=4):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.layers(x)

def train_model(name: str):
    X = torch.tensor([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float32)
    Y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

    net = xor_mlp()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    print(f"Training {name}...")
    for epoch in range(1000):
        optimizer.zero_grad()
        out = net(X)
        loss = loss_fn(out, Y)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 100 == 0:
            print(f"  Epoch {epoch+1}, Loss: {loss.item():.4f}")
    return net

if __name__ == "__main__":
    net_a = train_model("Network A")
    net_b = train_model("Network B")

    z3_net_a = nneq.net(net_a, (2,))
    z3_net_b = nneq.net(net_b, (2,))
    
    print("")
    nneq.strict_equivalence(z3_net_a, z3_net_b)
    print("")
    nneq.epsilon_equivalence(z3_net_a, z3_net_b, 0.1)
    print("")
    nneq.argmax_equivalence(z3_net_a, z3_net_b)

