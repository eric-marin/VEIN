
class Concrete:
   def __init__(self, val): self.val = val
   def __str__(self): return str(self.val)

class Symbolic:
   def __init__(self, id): self.id = id
   def __str__(self): return f"x_{self.id}"

class TermAdd:
   def __init__(self, a, b): self.a, self.b = a, b
   def __str__(self): return f"(+ {self.a} {self.b})"

class TermMul:
   def __init__(self, a, b): self.a, self.b = a, b
   def __str__(self): return f"(* {self.a} {self.b})"

class TermReLU:
   def __init__(self, a): self.a = a
   def __str__(self): return f"(ite (> {self.a} 0) {self.a} 0)"

def generate_z3(net_a_str, net_b_str, epsilon=1e-5):
   context = {
       'Concrete': Concrete,
       'Symbolic': Symbolic,
       'TermAdd': TermAdd,
       'TermMul': TermMul,
       'TermReLU': TermReLU
   }

   try:
       tree_a = eval(net_a_str, context)
       tree_b = eval(net_b_str, context)
   except Exception as e:
       print(f"; Error parsing Inpla output: {e}")
       return

   print("(declare-const x_0 Real)") 
   print("(declare-const x_1 Real)")

   print(f"(define-fun net_a () Real {tree_a})")
   print(f"(define-fun net_b () Real {tree_b})")

   print(f"(assert (> (abs (- net_a net_b)) {epsilon:.10f}))")
   
   print("(check-sat)")
   print("(get-model)")

if __name__ == "__main__":
   output_net_a = "TermAdd(TermMul(Symbolic(0), Concrete(2)), Concrete(3))" # 2x + 3
   output_net_b = "TermAdd(Concrete(3), TermMul(Concrete(2), Symbolic(0)))" # 3 + 2x
   
   generate_z3(output_net_a, output_net_b)
