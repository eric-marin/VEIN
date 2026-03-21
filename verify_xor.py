import z3
import nneq

def check_equivalence(onnx_a, onnx_b, vnnlib):
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
        m = solver.model()
        sorted_symbols = sorted([s for s in m.decls() if s.name().startswith("X_")], key=lambda s: s.name())
        for s in sorted_symbols:
            print(f"  {s.name()} = {m[s]}")
    else:
        print("UNKNOWN")
    print("")

if __name__ == "__main__":
    check_equivalence("./xor/xor_a.onnx", "./xor/xor_b.onnx", "./xor/xor_strict.vnnlib")
    check_equivalence("./xor/xor_a.onnx", "./xor/xor_b.onnx", "./xor/xor_epsilon.vnnlib")
    check_equivalence("./xor/xor_a.onnx", "./xor/xor_b.onnx", "./xor/xor_argmax.vnnlib")
