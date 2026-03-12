import torch
import torch.nn as nn
import torch.fx as fx
import numpy as np
import os
from typing import List, Dict

class XOR_MLP(nn.Module):
    def __init__(self, hidden_dim=4):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.layers(x)

class NameGen:
    def __init__(self):
        self.counter = 0
    def next(self) -> str:
        name = f"v{self.counter}"
        self.counter += 1
        return name

def get_rules() -> str:
    rules_path = os.path.join(os.path.dirname(__file__), "rules.in")
    if not os.path.exists(rules_path):
        return "// Rules not found in rules.in\n"
    
    rules_lines = []
    with open(rules_path, "r") as f:
        for line in f:
            if "// Net testing" in line:
                break
            rules_lines.append(line)
    return "".join(rules_lines)

def export_to_inpla_wiring(model: nn.Module, input_shape: tuple) -> str:
   traced = fx.symbolic_trace(model)
   name_gen = NameGen()
   script: List[str] = []
   wire_map: Dict[str, List[str]] = {}
   
   for node in traced.graph.nodes:
       if node.op == 'placeholder':
           num_inputs = int(np.prod(input_shape))
           wire_map[node.name] = [f"Linear(Symbolic(X_{i}), 1.0, 0.0)" for i in range(num_inputs)]

       elif node.op == 'call_module':
           target_str = str(node.target)
           module = dict(model.named_modules())[target_str]
           
           input_node = node.args[0]
           if not isinstance(input_node, fx.Node):
               continue
           input_wires = wire_map[input_node.name]

           if isinstance(module, nn.Flatten):
               wire_map[node.name] = input_wires

           elif isinstance(module, nn.Linear):
               W = (module.weight.data.detach().cpu().numpy()).astype(float)
               B = (module.bias.data.detach().cpu().numpy()).astype(float)
               out_dim, in_dim = W.shape
               
               neuron_wires = [f"Concrete({B[j]})" for j in range(out_dim)]

               for i in range(in_dim):
                   in_term = input_wires[i] 
                   if out_dim == 1:
                       weight = float(W[0, i])
                       if weight == 0:
                           script.append(f"Eraser ~ {in_term};")
                       elif weight == 1:
                           new_s = name_gen.next()
                           script.append(f"Add({new_s}, {in_term}) ~ {neuron_wires[0]};")
                           neuron_wires[0] = new_s
                       else:
                           mul_out = name_gen.next()
                           new_s = name_gen.next()
                           script.append(f"Mul({mul_out}, Concrete({weight})) ~ {in_term};")
                           script.append(f"Add({new_s}, {mul_out}) ~ {neuron_wires[0]};")
                           neuron_wires[0] = new_s
                   else:
                       branch_wires = [name_gen.next() for _ in range(out_dim)]
                       
                       def nest_dups(names: List[str]) -> str:
                           if len(names) == 1: return names[0]
                           if len(names) == 2: return f"Dup({names[0]}, {names[1]})"
                           return f"Dup({names[0]}, {nest_dups(names[1:])})"
                       
                       script.append(f"{nest_dups(branch_wires)} ~ {in_term};")
                       
                       for j in range(out_dim):
                           weight = float(W[j, i])
                           if weight == 0:
                               script.append(f"Eraser ~ {branch_wires[j]};")
                           elif weight == 1:
                               new_s = name_gen.next()
                               script.append(f"Add({new_s}, {branch_wires[j]}) ~ {neuron_wires[j]};")
                               neuron_wires[j] = new_s
                           else:
                               mul_out = name_gen.next()
                               new_s = name_gen.next()
                               script.append(f"Mul({mul_out}, Concrete({weight})) ~ {branch_wires[j]};")
                               script.append(f"Add({new_s}, {mul_out}) ~ {neuron_wires[j]};")
                               neuron_wires[j] = new_s

               wire_map[node.name] = neuron_wires

           elif isinstance(module, nn.ReLU):
               output_wires = []
               for i, w in enumerate(input_wires):
                   r_out = name_gen.next()
                   script.append(f"ReLU({r_out}) ~ {w};")
                   output_wires.append(r_out)
               wire_map[node.name] = output_wires

       elif node.op == 'output':
           output_node = node.args[0]
           if isinstance(output_node, fx.Node):
               final_wires = wire_map[output_node.name]
               for i, w in enumerate(final_wires):
                   res_name = f"result{i}"
                   script.append(f"Materialize({res_name}) ~ {w};")
                   script.append(f"{res_name};")

   return "\n".join(script)

def train_model(name: str):
    X = torch.tensor([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float32)
    Y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

    net = XOR_MLP()
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
    # Train two different models
    net_a = train_model("Network A")
    net_b = train_model("Network B")

    print("\nExporting both to xor.in...")
    
    rules = get_rules()
    wiring_a = export_to_inpla_wiring(net_a, (2,))
    wiring_b = export_to_inpla_wiring(net_b, (2,))

    with open("xor.in", "w") as f:
        f.write(rules)
        f.write("\n\n// Network A\n")
        f.write(wiring_a)
        f.write("\nfree ifce;\n")
        f.write("\n\n// Network B\n")
        f.write(wiring_b)
    
    print("Done. Now run: inpla -f xor.in | python3 prover.py")
