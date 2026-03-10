import z3
import sys

syms = {}
def Symbolic(id):
    id = f"x_{id}"
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

def equivalence(net_a, net_b):
    solver = z3.Solver()

    solver.add(net_a != net_b)

    result = solver.check()

    if result == z3.unsat:
        print("VERIFIED: The networks are equivalent.")
    elif result == z3.sat:
        print("FAILED: The networks are different.")
        print("Counter-example input:")
        print(solver.model())
    else:
        print("UNKNOWN: Solver could not decide.")

def epsilon_equivalence(net_a, net_b, epsilon):
    solver = z3.Solver()
    
    solver.add(z3.Abs(net_a - net_b) > epsilon)

    result = solver.check()

    if result == z3.unsat:
        print(f"VERIFIED: The networks are epsilon equivalent, with epsilon={epsilon}.")
    elif result == z3.sat:
        print("FAILED: The networks are different.")
        print("Counter-example input:")
        print(solver.model())
    else:
        print("UNKNOWN: Solver could not decide.")

def argmax_equivalence(net_a, net_b):
    solver = z3.Solver()
    
    solver.add(z3.IsInt(net_a > 0.5) != z3.IsInt(net_b > 0.5))

    result = solver.check()

    if result == z3.unsat:
        print(f"VERIFIED: The networks are argmax equivalent.")
    elif result == z3.sat:
        print("FAILED: The networks are different.")
        print("Counter-example input:")
        print(solver.model())
    else:
        print("UNKNOWN: Solver could not decide.")


if __name__ == "__main__":
   lines = [line.strip() for line in sys.stdin if line.strip() and not line.startswith("(")]
   
   if len(lines) < 2:
       print(f"; Error: Expected at least 2 Inpla output strings, but got {len(lines)}.")
       sys.exit(1)

   try:
       net_a_str = lines[-2]
       net_b_str = lines[-1]
       
       print(f"Comparing:\nA: {net_a_str}\nB: {net_b_str}")
       
       net_a = eval(net_a_str, context)
       net_b = eval(net_b_str, context)
       
       print("\nStrict Equivalence")
       equivalence(net_a, net_b)
       print("\nEpsilon-Equivalence")
       epsilon_equivalence(net_a, net_b, 1e-5)
       print("\nARGMAX Equivalence")
       argmax_equivalence(net_a, net_b)

   except Exception as e:
       print(f"; Error parsing Inpla output: {e}")
       sys.exit(1)
