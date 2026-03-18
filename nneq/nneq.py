import z3
import re
import numpy as np
import subprocess
import onnx
import ast
from onnx import numpy_helper
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

def inpla_export(model: onnx.ModelProto) -> inpla_str:
    class NameGen:
        def __init__(self):
            self.counter = 0
        def next(self) -> str:
            name = f"v{self.counter}"
            self.counter += 1
            return name

    def get_initializers(graph) -> Dict[str, np.ndarray]:
        initializers = {}
        for init in graph.initializer:
            initializers[init.name] = numpy_helper.to_array(init)
        return initializers

    def get_attrs(node: onnx.NodeProto) -> Dict:
        return {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
    
    def get_dim(name):
        for i in list(graph.input) + list(graph.output) + list(graph.value_info):
            if i.name == name: return i.type.tensor_type.shape.dim[-1].dim_value
        return None

    def nest_dups(terms: List[str]) -> str:
        if not terms: return "Eraser"
        if len(terms) == 1: return terms[0]
        return f"Dup({nest_dups(terms[:len(terms)//2])}, {nest_dups(terms[len(terms)//2:])})"

    def op_gemm(node):
        attrs = get_attrs(node)
        W = initializers[node.input[1]]
        if not attrs.get("transB", 0): W = W.T
        out_dim, in_dim = W.shape
        B = initializers[node.input[2]] if len(node.input) > 2 else np.zeros(out_dim)
        alpha, beta = attrs.get("alpha", 1.0), attrs.get("beta", 1.0)

        if node.input[0] not in interactions: interactions[node.input[0]] = [[] for _ in range(in_dim)]
        
        out_terms = interactions.get(node.output[0]) or [[] for _ in range(out_dim)]
        
        for j in range(out_dim):
            chain = nest_dups(out_terms[j])
            for i in range(in_dim):
                weight = float(alpha * W[j, i])
                if weight == 0: interactions[node.input[0]][i].append("Eraser")
                else:
                    v = name_gen.next()
                    chain, term = f"Add({chain}, {v})", f"Mul({v}, Concrete({weight}))"
                    interactions[node.input[0]][i].append(term)
            yield f"{chain} ~ Concrete({float(beta * B[j])});"

    def op_relu(node):
        out_name, in_name = node.output[0], node.input[0]
        if out_name in interactions:
            dim = len(interactions[out_name])
            if in_name not in interactions: interactions[in_name] = [[] for _ in range(dim)]
            for i in range(dim):
                interactions[in_name][i].append(f"ReLU({nest_dups(interactions[out_name][i])})")
        yield from []

    def op_flatten(node):
        out_name, in_name = node.output[0], node.input[0]
        if out_name in interactions:
            interactions[in_name] = interactions[out_name]
        yield from []

    graph, initializers, name_gen = model.graph, get_initializers(model.graph), NameGen()
    interactions: Dict[str, List[List[str]]] = {}
    ops = {"Gemm": op_gemm, "Relu": op_relu, "Flatten": op_flatten}

    if graph.output:
        out = graph.output[0].name
        dim = get_dim(out)
        if dim: interactions[out] = [[f"Materialize(result{i})"] for i in range(dim)]

    node_script = []
    for node in reversed(graph.node):
        if node.op_type in ops: node_script.extend(ops[node.op_type](node))

    input_script = []
    if graph.input and graph.input[0].name in interactions:
        for i, terms in enumerate(interactions[graph.input[0].name]):
            input_script.append(f"{nest_dups(terms)} ~ Linear(Symbolic(X_{i}), 1.0, 0.0);")
    
    result_lines = [f"result{i};" for i in range(len(interactions.get(graph.output[0].name, [])))]
    return "\n".join(input_script + list(reversed(node_script)) + result_lines)

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
    model = wrap.sub(r'Symbolic("\1")', model)

    def evaluate_node(node: ast.AST):
        if isinstance(node, ast.Expression):
            return evaluate_node(node.body)
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError(f"Unsupported function call type: {type(node.func)}")
            func_name = node.func.id
            func = context.get(func_name)
            if not func:
                raise ValueError(f"Unknown function: {func_name}")
            return func(*[evaluate_node(arg) for arg in node.args])
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            val = evaluate_node(node.operand)
            if hasattr(val, "__neg__"):
                return -val
            raise ValueError(f"Value does not support negation: {type(val)}")
        raise ValueError(f"Unsupported AST node: {type(node)}")

    lines = [line.strip() for line in model.splitlines() if line.strip()]
    exprs = [evaluate_node(ast.parse(line, mode='eval')) for line in lines]
    
    if not exprs: return None
    return exprs[0] if len(exprs) == 1 else exprs

def net(model: onnx.ModelProto):
    return z3_evaluate(inpla_run(inpla_export(model)))

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
