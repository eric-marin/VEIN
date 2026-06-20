# VErification via interaction Nets (VEIN)
# Copyright (C) 2026 Eric Marin
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU Affero General Public License as published by the Free Software Foundation, either version 3
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License along with this program.
# If not, see <https://www.gnu.org/licenses/>. 

import z3
import numpy as np
import subprocess
import onnx
import onnx.shape_inference
from onnx import numpy_helper
from typing import List, Dict, Optional
import os
import tempfile
import hashlib
from itertools import count
import re
import ast

sat = z3.sat
unsat = z3.unsat

rules = """
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
        | (q == 0) => out ~ TermConcrete(r), x ~ Eraser
        | (q == 1) && (r == 0) => out ~ x
        | (q == 1) && (r != 0) => out ~ TermAdd(x, TermConcrete(r))
        | (q != 0) && (r == 0) => out ~ TermMul(TermConcrete(q), x)
        | _ => out ~ TermAdd(TermMul(TermConcrete(q), x), TermConcrete(r));
    Concrete(float k) >< Materialize(out) => out ~ (*L)TermConcrete(k);
"""

_CACHE  = {}

def inpla_export(model: onnx.ModelProto, bounds: Optional[Dict[str, List[float]]] = None) -> str:
    # TODO: Add Range agent
    _ = bounds

    def get_initializers(graph) -> Dict[str, np.ndarray]:
        initializers = {}
        for init in graph.initializer:
            initializers[init.name] = numpy_helper.to_array(init)
        return initializers

    def get_attrs(node) -> Dict:
        return {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}

    graph, initializers = model.graph, get_initializers(model.graph)
    counter = count()
    wire_gen = lambda: f"w{next(counter)}"
    interactions: Dict[str, List[List[str]]] = {}
    dims = {i.name: i.type.tensor_type.shape.dim[-1].dim_value for i in list(graph.input) + list(graph.output) + list(graph.value_info)}
    script = []

    def balanced_fanout(agent_name: str, terms: List[str]) -> str:
        if not terms: return "Eraser"
        if len(terms) == 1: return terms[0]
        while len(terms) > 1:
            next_level = []
            for i in range(0, len(terms), 2):
                if i + 1 < len(terms):
                    in_w = wire_gen()
                    script.append(f"{in_w} ~ {agent_name}({terms[i]}, {terms[i+1]});")
                    next_level.append(in_w)
                else:
                    next_level.append(terms[i])
            terms = next_level
        return terms[0]

    def balanced_fanin(agent_name: str, terms: List[str]) -> str:
        if not terms: return "Eraser"
        if len(terms) == 1: return terms[0]
        while len(terms) > 1:
            next_level = []
            for i in range(0, len(terms), 2):
                if i + 1 < len(terms):
                    res_w = wire_gen()
                    script.append(f"{terms[i]} ~ {agent_name}({res_w}, {terms[i+1]});")
                    next_level.append(res_w)
                else:
                    next_level.append(terms[i])
            terms = next_level
        return terms[0]

    def gemm(Y, A, B, C, alpha, beta, _, transB):
        weights = initializers[B]
        if transB == 0:
            weights = weights.T
        out_dim, in_dim = weights.shape
        biases = initializers[C] if C is not None else None
        if A not in interactions:
            interactions[A] = [[] for _ in range(in_dim)]
        out_terms = interactions.get(Y) or [[f"Materialize(result{j})"] for j in range(out_dim)]
        for j in range(out_dim):
            sink = balanced_fanout("Dup", out_terms[j])
            neuron_terms = []
            for i in range(in_dim):
                weight = float(alpha * weights[j, i])
                if weight != 0:
                    v = wire_gen()
                    interactions[A][i].append(f"Mul({v}, Concrete({weight}))")
                    neuron_terms.append(v)
            bias = float(beta * biases[j]) if biases is not None else 0.0
            if bias != 0 or len(neuron_terms) == 0:
                neuron_terms.append(f"Concrete({bias})")
            root = balanced_fanin("Add", neuron_terms)
            script.append(f"{root} ~ {sink};")

    def op_gemm(node):
        attrs = get_attrs(node)
        gemm(node.output[0], node.input[0], node.input[1], node.input[2], attrs.get("alpha", 1.0), attrs.get("beta", 1.0), attrs.get("transA", 0), attrs.get("transB", 0))

    def op_matmul(node):
        gemm(node.output[0], node.input[0], node.input[1], None, 1.0, 0.0, 0, 0)

    def op_relu(node):
        in_name, out_name = node.input[0], node.output[0]
        dim = dims.get(out_name) or 1

        if in_name not in interactions: 
            interactions[in_name] = [[] for _ in range(dim)]

        out_terms = interactions.get(node.output[0]) or [[f"Materialize(result{j})"] for j in range(dim)]

        for i in range(dim):
            sink = balanced_fanout("Dup", out_terms[i])
            v = wire_gen()
            interactions[in_name][i].append(f"ReLU({v})")
            script.append(f"{v} ~ {sink};")

    def op_add(node):
        in_a, in_b = node.input[0], node.input[1]
        out_name = node.output[0]

        dim = dims.get(out_name) or dims.get(in_a) or dims.get(in_b) or 1

        if in_a not in interactions:
            interactions[in_a] = [[] for _ in range(dim)]
        if in_b not in interactions:
            interactions[in_b] = [[] for _ in range(dim)]

        out_terms = interactions.get(out_name) or [[f"Materialize(result{j})"] for j in range(dim)]

        b_const = initializers.get(in_b)
        a_const = initializers.get(in_a)

        for i in range(dim):
            sink = balanced_fanout("Dup", out_terms[i])
            if b_const is not None:
                val = float(b_const.flatten()[i % b_const.size])
                interactions[in_a][i].append(f"Add({sink}, Concrete({val}))")
            elif a_const is not None:
                val = float(a_const.flatten()[i % a_const.size])
                interactions[in_b][i].append(f"Add({sink}, Concrete({val}))")
            else:
                v_b = wire_gen()
                interactions[in_a][i].append(f"Add({sink}, {v_b})")
                interactions[in_b][i].append(f"{v_b}")

    def op_sub(node):
        in_a, in_b = node.input[0], node.input[1]
        out_name = node.output[0]

        dim = dims.get(out_name) or dims.get(in_a) or dims.get(in_b) or 1

        if in_a not in interactions:
            interactions[in_a] = [[] for _ in range(dim)]
        if in_b not in interactions:
            interactions[in_b] = [[] for _ in range(dim)]

        out_terms = interactions.get(out_name) or [[f"Materialize(result{j})"] for j in range(dim)]

        b_const = initializers.get(in_b)
        a_const = initializers.get(in_a)

        for i in range(dim):
            sink = balanced_fanout("Dup", out_terms[i])
            if b_const is not None:
                val = float(b_const.flatten()[i % b_const.size])
                interactions[in_a][i].append(f"Add({sink}, Concrete({-val}))")
            elif a_const is not None:
                val = float(a_const.flatten()[i % a_const.size])
                interactions[in_b][i].append(f"Mul(Add({sink}, Concrete({val})), Concrete(-1.0))")
            else:
                v_b = wire_gen()
                interactions[in_a][i].append(f"Add({sink}, {v_b})")
                interactions[in_b][i].append(f"Mul({v_b}, Concrete(-1.0))")

    def op_squeeze(node):
        op_identity(node)

    def op_unsqueeze(node):
        op_identity(node)

    def op_flatten(node):
        op_identity(node)

    def op_reshape(node):
        op_identity(node)

    def op_identity(node):
        in_name, out_name = node.input[0], node.output[0]
        if out_name in interactions:
            interactions[in_name] = interactions[out_name]

    ops = {
        "Gemm": op_gemm,
        "Relu": op_relu,
        "Flatten": op_flatten,
        "Reshape": op_reshape,
        "MatMul": op_matmul,
        "Add": op_add,
        "Sub": op_sub,
        "Squeeze": op_squeeze,
        "Unsqueeze": op_unsqueeze,
        "Identity": op_identity
    }

    if graph.output:
        out = graph.output[0].name
        dim = dims.get(out)
        if dim: 
            interactions[out] = [[f"Materialize(result{i})"] for i in range(dim)]

    for node in reversed(graph.node):
        if node.op_type in ops: 
            ops[node.op_type](node)
        else:
            raise RuntimeError(f"Unsupported ONNX operator: {node.op_type}")

    if graph.input:
        for input in graph.input:
            if input.name in interactions:
                for i, terms in enumerate(interactions[input.name]):
                    sink = balanced_fanout("Dup", terms)
                    script.append(f"{sink} ~ Linear(TermSymbolic(X_{i}), 1.0, 0.0);")

    result_lines = [f"result{i};" for i in range(len(interactions.get(graph.output[0].name, [])))]
    return "\n".join(script + result_lines)


def inpla_run(model: str) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".inpla", delete=False) as f:
        f.write(f"{rules}\n{model}")
        temp_path = f.name
    try:
        res = subprocess.run(["./inpla", "-f", temp_path, "-foptimise-tail-calls"], capture_output=True, text=True)
        if res.stderr:
            raise RuntimeError(res.stderr)
        return res.stdout
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def z3_evaluate(model: str, X: dict):
    def TermSymbolic(id):
        if id not in X:
            X[id] = z3.Real(id)
        return X[id]
    def TermConcrete(val): return z3.RealVal(val)
    def TermAdd(a, b): return a + b
    def TermMul(a, b): return a * b
    def TermReLU(x): return z3.If(x > 0, x, 0)

    context = {
        'TermConcrete': TermConcrete,
        'TermSymbolic': TermSymbolic,
        'TermAdd': TermAdd,
        'TermMul': TermMul,
        'TermReLU': TermReLU
    }

    exprs = []
    allowed_calls = set(context.keys())
    allowed_nodes = (ast.Expression, ast.Call, ast.Name, ast.Load, ast.Constant, ast.UnaryOp, ast.USub)
    model = re.sub(r'X_\d+', lambda m: f'"{m.group(0)}"', model)
    for line in model.splitlines():
        line = line.strip().rstrip(';')
        if not line:
            continue
        tree = ast.parse(line, mode="eval")
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                raise ValueError(f"Disallowed syntax: {type(node).__name__}")
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name) or node.func.id not in allowed_calls:
                    raise ValueError("Disallowed function call")
            if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float, str)):
                raise ValueError(f"Only numeric constants and string names allowed")
        exprs.append(eval(compile(tree, "<model>", "eval"), {"__builtins__": {}}, context))
    return exprs

def net(model: onnx.ModelProto, X, bounds: Optional[Dict[str, List[float]]] = None):
    model_hash = hashlib.sha256(model.SerializeToString()).hexdigest()
    # bounds_key = tuple(sorted((k, tuple(v)) for k, v in bounds.items())) if bounds else None
    # cache_key = (model_hash, bounds_key)
    cache_key = model_hash

    if cache_key not in _CACHE:
        exported = inpla_export(model, bounds)
        reduced = inpla_run(exported)
        evaluated = z3_evaluate(reduced, X)
        _CACHE[cache_key] = evaluated

    exprs = _CACHE[cache_key]
    return exprs if exprs is not None else []

class Solver(z3.Solver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.bounds: Dict[str, List[float]] = {}
        self.pending_nets: List[onnx.ModelProto] = []
        self.X = {}

    def load_smtlib(self, file_path: str):
        with open(file_path, "r") as f:
            content = f.read()

        # for match in re.finditer(r"\(assert\s+\((>=|<=)\s+(X_\d+)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\)\)", content):
            # op, var, val = match.groups()
            # val = float(val)
            # if var not in self.bounds: self.bounds[var] = [float('-inf'), float('inf')]
            # if op == ">=": self.bounds[var][0] = val
            # else: self.bounds[var][1] = val

        assertions = z3.parse_smt2_string(content)
        self.add(assertions)

    def load_onnx(self, file_path: str):
        model = onnx.load(file_path)
        model = onnx.shape_inference.infer_shapes(model)
        self.pending_nets.append(model)

    def _process_nets(self):
        y_count = 0
        for model in self.pending_nets:
            z3_outputs = net(model, self.X)

            if z3_outputs:
                for out_expr in z3_outputs:
                    y_var = z3.Real(f"Y_{y_count}")
                    self.add(y_var == out_expr)
                    y_count += 1
        self.pending_nets = []

    def check(self, *args):
        self._process_nets()
        return super().check(*args)
