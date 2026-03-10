import z3

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

if __name__ == "__main__":
   net_a_str = "TermAdd(TermMul(Symbolic(0), Concrete(2)), Concrete(3))" # 2x + 3
   net_b_str = "TermAdd(Concrete(3), TermMul(Concrete(2), Symbolic(0)))" # 3 + 2x

   try:
       net_a = eval(net_a_str, context)
       net_b = eval(net_b_str, context)
       equivalence(net_a, net_b)
       epsilon_equivalence(net_a, net_b, 1e-5)
   except Exception as e:
       print(f"; Error parsing Inpla output: {e}")
   
