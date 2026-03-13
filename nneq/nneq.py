import z3
import re
import torch.fx as fx, torch.nn as nn
import numpy as np
import subprocess
from typing import List, Dict

__all__ = ["net", "strict_equivalence", "epsilon_equivalence", "argmax_equivalence"]

type inpla_str = str
type z3_str = str

rules: inpla_str = """
Linear(x, float q, float r) >< Add(out, b) => b ~ AddCheckLinear(out, x, q, r);
Concrete(float k) >< Add(out, b)
	| k == 0 => out ~ b
	| _ => b ~ AddCheckConcrete(out, k);
Linear(y, float s, float t) >< AddCheckLinear(out, x, float q, float r)
	| (q == 0) && (r == 0) && (s == 0) && (t == 0) => out ~ Concrete(0), x ~ Eraser, y ~ Eraser
	| (s == 0) && (t == 0) => out ~ Linear(x, q, r), y ~ Eraser
	| (q == 0) && (r == 0) => out ~ (*L)Linear(y, s, t), x ~ Eraser
	| _ => Linear(x, q, r) ~ Materialize(out_x), (*L)Linear(y, s, t) ~ Materialize(out_y), out ~ Linear(TermAdd(out_x, out_y), 1, 0);
Concrete(float j) >< AddCheckLinear(out, x, float q, float r) => out ~ Linear(x, q, r + j);
Linear(y, float s, float t) >< AddCheckConcrete(out, float k) => out ~ Linear(y, s, t + k);
Concrete(float j) >< AddCheckConcrete(out, float k)
	| j == 0 => out ~ Concrete(k)
	| _ => out ~ Concrete(k + j);
Linear(x, float q, float r) >< Mul(out, b) => b ~ MulCheckLinear(out, x, q, r);
Concrete(float k) >< Mul(out, b)
	| k == 0 => b ~ Eraser, out ~ (*L)Concrete(0)
	| k == 1 => out ~ b
	| _ => b ~ MulCheckConcrete(out, k);
Linear(y, float s, float t) >< MulCheckLinear(out, x, float q, float r)
	| ((q == 0) && (r == 0)) || ((s == 0) && (t == 0)) => out ~ Concrete(0), x ~ Eraser, y ~ Eraser
	| _ => Linear(x, q, r) ~ Materialize(out_x), (*L)Linear(y, s, t) ~ Materialize(out_y), out ~ Linear(TermMul(out_x, out_y), 1, 0);
Concrete(float j) >< MulCheckLinear(out, x, float q, float r) => out ~ Linear(x, q * j, r * j);
Linear(y, float s, float t) >< MulCheckConcrete(out, float k) => out ~ Linear(y, s * k, t * k);
Concrete(float j) >< MulCheckConcrete(out, float k)
	| j == 0 => out ~ Concrete(0)
	| j == 1 => out ~ Concrete(k)
	| _ => out ~ Concrete(k * j);
Linear(x, float q, float r) >< ReLU(out) => (*L)Linear(x, q, r) ~ Materialize(out_x), out ~ Linear(TermReLU(out_x), 1, 0);
Concrete(float k) >< ReLU(out)
	| k > 0 => out ~ (*L)Concrete(k)
	| _ => out ~ Concrete(0);
Linear(x, float q, float r) >< Materialize(out)
	| (q == 0) => out ~ Concrete(r), x ~ Eraser
	| (q == 1) && (r == 0) => out ~ x
	| (q == 1) && (r != 0) => out ~ TermAdd(x, Concrete(r))
	| (q != 0) && (r == 0) => out ~ TermMul(Concrete(q), x)
	| _ => out ~ TermAdd(TermMul(Concrete(q), x), Concrete(r));
Concrete(float k) >< Materialize(out) => out ~ (*L)Concrete(k);
"""

class NameGen:
    def __init__(self):
        self.counter = 0
    def next(self) -> str:
        name = f"v{self.counter}"
        self.counter += 1
        return name

def inpla_export(model: nn.Module, input_shape: tuple) -> inpla_str:
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

def inpla_run(model: inpla_str) -> z3_str:
    return subprocess.run(["./inpla"], input=f"{rules}\n{model}", capture_output=True, text=True).stdout

syms = {}
def Symbolic(id):
    if id not in syms:
        syms[id] = z3.Real(id)
    return syms[id]

def Concrete(val): return z3.RealVal(val)

def TermAdd(a, b): return a + b
def TermMul(a, b): return a * b
def TermReLU(x): return z3.If(x > 0, x, 0)

context = {
   'Concrete': Concrete,
   'Symbolic': Symbolic,
   'TermAdd': TermAdd,
   'TermMul': TermMul,
   'TermReLU': TermReLU
}

wrap = re.compile(r"Symbolic\((.*?)\)")

def z3_evaluate(model: z3_str):
    model = wrap.sub(r'Symbolic("\1")', model);
    return eval(model, context)

def net(model: nn.Module, input_shape: tuple):
    return z3_evaluate(inpla_run(inpla_export(model, input_shape)))


def strict_equivalence(net_a, net_b):
    solver = z3.Solver()

    for sym in syms.values():
        solver.add(z3.Or(sym == 0, sym == 1))

    solver.add(net_a != net_b)

    result = solver.check()

    print("Strict Equivalence")
    if result == z3.unsat:
        print("VERIFIED: The networks are strictly equivalent.")
    elif result == z3.sat:
        print("FAILED: The networks are different.")
        print("Counter-example input:")
        print(solver.model())
    else:
        print("UNKNOWN: Solver could not decide.")

def epsilon_equivalence(net_a, net_b, epsilon):
    solver = z3.Solver()

    for sym in syms.values():
        solver.add(z3.Or(sym == 0, sym == 1))
    
    solver.add(z3.Abs(net_a - net_b) > epsilon)

    result = solver.check()

    print(f"Epsilon-Equivalence | Epsilon={epsilon}.")
    if result == z3.unsat:
        print("VERIFIED: The networks are epsilon-equivalent.")
    elif result == z3.sat:
        print("FAILED: The networks are different.")
        print("Counter-example input:")
        print(solver.model())
    else:
        print("UNKNOWN: Solver could not decide.")

def argmax_equivalence(net_a, net_b):
    solver = z3.Solver()

    for sym in syms.values():
        solver.add(z3.Or(sym == 0, sym == 1))
    
    solver.add((net_a > 0.5) != (net_b > 0.5))

    result = solver.check()

    print("ARGMAX Equivalence")
    if result == z3.unsat:
        print("VERIFIED: The networks are ARGMAX equivalent.")
    elif result == z3.sat:
        print("FAILED: The networks are different.")
        print("Counter-example input:")
        print(solver.model())
    else:
        print("UNKNOWN: Solver could not decide.")
