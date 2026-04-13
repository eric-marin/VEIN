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
import re
import numpy as np
import subprocess
import onnx
import onnx.shape_inference
from onnx import numpy_helper
from typing import List, Dict, Optional
import os
import tempfile
import hashlib

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
        | (q == 0) => out ~ Concrete(r), x ~ Eraser
        | (q == 1) && (r == 0) => out ~ x
        | (q == 1) && (r != 0) => out ~ TermAdd(x, Concrete(r))
        | (q != 0) && (r == 0) => out ~ TermMul(Concrete(q), x)
        | _ => out ~ TermAdd(TermMul(Concrete(q), x), Concrete(r));
    Concrete(float k) >< Materialize(out) => out ~ (*L)Concrete(k);
"""

_CACHE  = {}

def inpla_export(model: onnx.ModelProto, bounds: Optional[Dict[str, List[float]]] = None) -> str:
    # TODO: Add Range agent
    _ = bounds
    class NameGen:
        def __init__(self, prefix="v"):
            self.counter = 0
            self.prefix = prefix
        def next(self) -> str:
            name = f"{self.prefix}{self.counter}"
            self.counter += 1
            return name

    def get_initializers(graph) -> Dict[str, np.ndarray]:
        initializers = {}
        for init in graph.initializer:
            initializers[init.name] = numpy_helper.to_array(init)
        return initializers

    def get_attrs(node) -> Dict:
        return {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}

    def get_dim(name):
        for i in list(graph.input) + list(graph.output) + list(graph.value_info):
            if i.name == name: return i.type.tensor_type.shape.dim[-1].dim_value
        return None

    def flatten_nest(agent_name: str, terms: List[str]) -> str:
        if not terms: return "Eraser"
        if len(terms) == 1: return terms[0]
        current = terms[0]
        for i in range(1, len(terms)):
            wire = wire_gen.next()
            script.append(f"{wire} ~ {agent_name}({current}, {terms[i]});")
            current = wire
        return current

    def balance_add(terms: List[str], sink: str):
        if not terms:
            script.append(f"{sink} ~ Eraser;")
            return
        if len(terms) == 1:
            script.append(f"{sink} ~ {terms[0]};")
            return

        nodes = terms
        while len(nodes) > 1:
            next_level = []
            for i in range(0, len(nodes), 2):
                if i + 1 < len(nodes):
                    wire_out = wire_gen.next()
                    script.append(f"{nodes[i]} ~ Add({wire_out}, {nodes[i+1]});")
                    next_level.append(wire_out)
                else:
                    next_level.append(nodes[i])
            nodes = next_level
        script.append(f"{nodes[0]} ~ {sink};")

    def op_gemm(node, override_attrs=None):
        attrs = override_attrs if override_attrs is not None else get_attrs(node)

        W = initializers[node.input[1]]
        if not attrs.get("transB", 0): W = W.T
        out_dim, in_dim = W.shape

        B = initializers[node.input[2]] if len(node.input) > 2 else np.zeros(out_dim)
        alpha, beta = attrs.get("alpha", 1.0), attrs.get("beta", 1.0)

        if node.input[0] not in interactions: 
            interactions[node.input[0]] = [[] for _ in range(in_dim)]

        out_terms = interactions.get(node.output[0]) or [[f"Materialize(result{j})"] for j in range(out_dim)]

        for j in range(out_dim):
            sink = flatten_nest("Dup", out_terms[j])
            neuron_terms = []
            for i in range(in_dim):
                weight = float(alpha * W[j, i])
                if weight != 0:
                    v = wire_gen.next()
                    interactions[node.input[0]][i].append(f"Mul({v}, Concrete({weight}))")
                    neuron_terms.append(v)

            bias_val = float(beta * B[j])
            if bias_val != 0 or not neuron_terms:
                neuron_terms.append(f"Concrete({bias_val})")

            balance_add(neuron_terms, sink)

    def op_matmul(node):
        op_gemm(node, override_attrs={"alpha": 1.0, "beta": 0.0, "transB": 0})

    def op_relu(node):
        out_name, in_name = node.output[0], node.input[0]
        dim = get_dim(out_name) or 1

        if in_name not in interactions: 
            interactions[in_name] = [[] for _ in range(dim)]

        out_terms = interactions.get(node.output[0]) or [[f"Materialize(result{j})"] for j in range(dim)]

        for i in range(dim):
            sink = flatten_nest("Dup", out_terms[i])
            v = wire_gen.next()
            interactions[in_name][i].append(f"ReLU({v})")
            script.append(f"{v} ~ {sink};")

    def op_flatten(node):
        op_identity(node)

    def op_reshape(node):
        op_identity(node)

    def op_add(node):
        out_name = node.output[0]
        in_a, in_b = node.input[0], node.input[1]

        dim = get_dim(out_name) or get_dim(in_a) or get_dim(in_b) or 1

        if in_a not in interactions: interactions[in_a] = [[] for _ in range(dim)]
        if in_b not in interactions: interactions[in_b] = [[] for _ in range(dim)]

        out_terms = interactions.get(out_name) or [[f"Materialize(result{j})"] for j in range(dim)]

        b_const = initializers.get(in_b)
        a_const = initializers.get(in_a)

        for i in range(dim):
            sink = flatten_nest("Dup", out_terms[i])
            if b_const is not None:
                val = float(b_const.flatten()[i % b_const.size])
                interactions[in_a][i].append(f"Add({sink}, Concrete({val}))")
            elif a_const is not None:
                val = float(a_const.flatten()[i % a_const.size])
                interactions[in_b][i].append(f"Add({sink}, Concrete({val}))")
            else:
                v_b = wire_gen.next()
                interactions[in_a][i].append(f"Add({sink}, {v_b})")
                interactions[in_b][i].append(f"{v_b}")

    def op_sub(node):
        out_name = node.output[0]
        in_a, in_b = node.input[0], node.input[1]

        dim = get_dim(out_name) or get_dim(in_a) or get_dim(in_b) or 1

        if out_name not in interactions:
            interactions[out_name] = [[f"Materialize(result{i})"] for i in range(dim)]

        out_terms = interactions.get(out_name) or [[f"Materialize(result{j})"] for j in range(dim)]

        if in_a not in interactions: interactions[in_a] = [[] for _ in range(dim)]
        if in_b not in interactions: interactions[in_b] = [[] for _ in range(dim)]

        b_const = initializers.get(in_b)
        a_const = initializers.get(in_a)

        for i in range(dim):
            sink = flatten_nest("Dup", out_terms[i])
            if b_const is not None:
                val = float(b_const.flatten()[i % b_const.size])
                interactions[in_a][i].append(f"Add({sink}, Concrete({-val}))")
            elif a_const is not None:
                val = float(a_const.flatten()[i % a_const.size])
                interactions[in_b][i].append(f"Add({sink}, Concrete({-val}))")
            else:
                v_b = wire_gen.next()
                interactions[in_a][i].append(f"Add({sink}, Mul({v_b}, Concrete(-1.0)))")
                interactions[in_b][i].append(f"{v_b}")

    def op_slice(node):
        in_name, out_name = node.input[0], node.output[0]
        if out_name in interactions:
            starts = initializers.get(node.input[1])
            steps = initializers.get(node.input[4]) if len(node.input) > 4 else None
            
            start = int(starts.flatten()[0]) if starts is not None else 0
            step = int(steps.flatten()[0]) if steps is not None else 1
            
            in_dim = get_dim(in_name) or 1
            if in_name not in interactions:
                interactions[in_name] = [[] for _ in range(in_dim)]

            for i, terms in enumerate(interactions[out_name]):
                input_index = start + (i * step)
                if input_index < in_dim:
                    interactions[in_name][input_index].extend(terms)

    def op_squeeze(node):
        op_identity(node)

    def op_unsqueeze(node):
        op_identity(node)

    def op_identity(node):
        in_name, out_name = node.input[0], node.output[0]
        if out_name in interactions:
            interactions[in_name] = interactions[out_name]


    graph, initializers = model.graph, get_initializers(model.graph)
    wire_gen = NameGen("w")
    interactions: Dict[str, List[List[str]]] = {}
    script = []
    ops = {
        "Gemm": op_gemm,
        "Relu": op_relu,
        "Flatten": op_flatten,
        "Reshape": op_reshape,
        "MatMul": op_matmul,
        "Add": op_add,
        "Sub": op_sub,
        "Slice": op_slice,
        "Squeeze": op_squeeze,
        "Unsqueeze": op_unsqueeze,
        "Identity": op_identity
    }

    if graph.output:
        out = graph.output[0].name
        dim = get_dim(out)
        if dim: 
            interactions[out] = [[f"Materialize(result{i})"] for i in range(dim)]

    for node in reversed(graph.node):
        if node.op_type in ops: 
            ops[node.op_type](node)
        else:
            raise RuntimeError(f"Unsupported ONNX operator: {node.op_type}")

    if graph.input:
        input = "input" if "input" in interactions else graph.input[0].name
        for i, terms in enumerate(interactions[input]):
            sink = flatten_nest("Dup", terms)
            script.append(f"{sink} ~ Linear(Symbolic(X_{i}), 1.0, 0.0);")

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
    def Symbolic(id):
        if id not in X:
            X[id] = z3.Real(id)
        return X[id]
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

    def tokenize(s):
        i = 0
        n = len(s)
        while i < n:
            c = s[i]
            if c in '(),':
                yield c
                i += 1
            elif c.isspace():
                i += 1
            else:
                start = i
                while i < n and s[i] not in '(), ' and not s[i].isspace():
                    i += 1
                yield s[start:i]

    def iterative_eval(tokens_gen):
        stack = [[]]
        for token in tokens_gen:
            if token == '(':
                stack.append([])
            elif token == ')':
                args = stack.pop()
                func_name = stack[-1].pop()
                func = context.get(func_name)
                if not func: raise ValueError(f"Unknown: {func_name}")
                stack[-1].append(func(*args))
            elif token == ',':
                continue
            else:
                if token in context:
                    stack[-1].append(token)
                else:
                    try:
                        stack[-1].append(float(token))
                    except ValueError:
                        stack[-1].append(token)
        return stack[0][0]

    exprs = []
    for line in model.splitlines():
        line = line.strip()
        exprs.append(iterative_eval(tokenize(line)))
    return exprs

def net(model: onnx.ModelProto, bounds: Optional[Dict[str, List[float]]] = None):
    model_hash = hashlib.sha256(model.SerializeToString()).hexdigest()
    bounds_key = tuple(sorted((k, tuple(v)) for k, v in bounds.items())) if bounds else None
    cache_key = (model_hash, bounds_key)

    if cache_key not in _CACHE:
        exported = inpla_export(model, bounds)
        reduced = inpla_run(exported)
        X = {}
        evaluated = z3_evaluate(reduced, X)
        _CACHE[cache_key] = evaluated

    exprs = _CACHE[cache_key]
    return exprs if exprs is not None else []

class Solver(z3.Solver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bounds: Dict[str, List[float]] = {}
        self.pending_nets: List[onnx.ModelProto] = []

    def load_vnnlib(self, file_path: str):
        with open(file_path, "r") as f:
            content = f.read()

        for match in re.finditer(r"\(assert\s+\((>=|<=)\s+(X_\d+)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\)\)", content):
            op, var, val = match.groups()
            val = float(val)
            if var not in self.bounds: self.bounds[var] = [float('-inf'), float('inf')]
            if op == ">=": self.bounds[var][0] = val
            else: self.bounds[var][1] = val

        assertions = z3.parse_smt2_string(content)
        self.add(assertions)

    def load_onnx(self, file_path: str):
        model = onnx.load(file_path)
        model = onnx.shape_inference.infer_shapes(model)
        self.pending_nets.append(model)

    def _process_nets(self):
        y_count = 0
        for model in self.pending_nets:
            z3_outputs = net(model, bounds=self.bounds)
            if z3_outputs:
                for _, out_expr in enumerate(z3_outputs):
                    y_var = z3.Real(f"Y_{y_count}")
                    self.add(y_var == out_expr)
                    y_count += 1
        self.pending_nets = []

    def check(self, *args):
        self._process_nets()
        return super().check(*args)
