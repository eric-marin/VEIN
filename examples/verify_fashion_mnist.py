import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import z3
import nneq

def check_property(onnx_a, onnx_b, vnnlib):
    solver = nneq.Solver()
    
    print(f"--- Checking {vnnlib} ---")
    
    solver.load_onnx(onnx_a)
    solver.load_onnx(onnx_b)
    solver.load_vnnlib(vnnlib)
    
    result = solver.check()
    
    if result == z3.unsat:
        print("VERIFIED (UNSAT): The networks are equivalent under this property.")
    elif result == z3.sat:
        print("FAILED (SAT): The networks are NOT equivalent.")
        print("Counter-example input:")
        print(solver.model())
        # m = solver.model()
        # sorted_symbols = sorted([s for s in m.decls() if s.name().startswith("X_")], key=lambda s: s.name())
        # for s in sorted_symbols:
            # print(f"  {s.name()} = {m[s]}")
    else:
        print("UNKNOWN")
    print("")

if __name__ == "__main__":
    check_property("./examples/fashion_mnist/fashion_mnist_a.onnx", "./examples/fashion_mnist/fashion_mnist_b.onnx", "./examples/fashion_mnist/fashion_mnist_strict.vnnlib")
    check_property("./examples/fashion_mnist/fashion_mnist_a.onnx", "./examples/fashion_mnist/fashion_mnist_b.onnx", "./examples/fashion_mnist/fashion_mnist_epsilon.vnnlib")
    check_property("./examples/fashion_mnist/fashion_mnist_a.onnx", "./examples/fashion_mnist/fashion_mnist_b.onnx", "./examples/fashion_mnist/fashion_mnist_argmax.vnnlib")
