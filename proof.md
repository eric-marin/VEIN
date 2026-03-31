# Soundness Proof
## Mathematical Definitions
![Linear(x, q, r) ~ out => out = q*x + r](https://latex.codecogs.com/svg.image?Linear(x,q,r)\sim&space;out\Rightarrow&space;out=q*x&plus;r&space;)
![Concrete(k) ~ out => out = ](https://latex.codecogs.com/svg.image?Concrete(k)\sim&space;out\Rightarrow&space;out=k&space;)
![Add(out, b) ~ a => out = a + b](https://latex.codecogs.com/svg.image?Add(out,b)\sim&space;a\Rightarrow&space;out=a&plus;b&space;)
![AddCheckLinear(out, x, q, r) ~ b => out = q*x + (r + b)](https://latex.codecogs.com/svg.image?AddCheckLinear(out,x,q,r)\sim&space;b\Rightarrow&space;out=q*x&plus;(r&plus;b))
![AddCheckConcrete(out, k) ~ b => out = k + b](https://latex.codecogs.com/svg.image?AddCheckConcrete(out,k)\sim&space;b\Rightarrow&space;out=k&plus;b&space;)
![Mul(out, b) ~ a => out = a * b](https://latex.codecogs.com/svg.image?Mul(out,b)\sim&space;a\Rightarrow&space;out=a*b&space;)
![MulCheckLinear(out, x, q, r) ~ b => out = q*b*x + r*b](https://latex.codecogs.com/svg.image?MulCheckLinear(out,x,q,r)\sim&space;b\Rightarrow&space;out=q*b*x&plus;r*b&space;)
![MulCheckConcrete(out, k) ~ b => out = k*b](https://latex.codecogs.com/svg.image?MulCheckConcrete(out,k)\sim&space;b\Rightarrow&space;out=k*b&space;)
![ReLU(out) ~ x => out = IF (x > 0) THEN x ELSE 0](https://latex.codecogs.com/svg.image?ReLU(out)\sim&space;x\Rightarrow&space;out=IF\;(x>0)\;THEN\;x\;ELSE\;0&space;)
![Materialize(out) ~ x => out = x](https://latex.codecogs.com/svg.image?Materialize(out)\sim&space;x\Rightarrow&space;out=x&space;)

## Soundness of Translation
### ReLU
ONNX ReLU node is defined as:
![Y = X if X > 0 else 0](https://latex.codecogs.com/svg.image?Y=X\;if\;X>0\;else\;0&space;)

The translation defines the interactions:
![x_i ~ ReLU(y_i)](https://latex.codecogs.com/svg.image?x_i\sim&space;ReLU(y_i))

By definition this interaction is equal to:
![y_i = IF (x_i > 0) THEN x_i ELSE 0](https://latex.codecogs.com/svg.image?y_i=IF\;(x_i>0)\;THEN\;x_i\;ELSE\;0&space;)

### Gemm
ONNX Gemm node is defined as:
![Y = alpha * A * B + beta * C](https://latex.codecogs.com/svg.image?Y=\alpha\cdot&space;A\cdot&space;B&plus;\beta\cdot&space;C&space;)

The translation defines the interactions:
![a_i ~ Mul(v_i, Concrete(alpha * b_i))](https://latex.codecogs.com/svg.image?a_i\sim&space;Mul(v_i,Concrete(\alpha*b_i)))
![Add(...(Add(y_i, v_1), ...), v_n) ~ Concrete(beta * c_i)](https://latex.codecogs.com/svg.image?Add(...(Add(y_i,v_1),...),v_n)\sim&space;Concrete(\beta*c_i))

By definition this interaction is equal to:
![v_i = alpha * a_i * b_i](https://latex.codecogs.com/svg.image?v_i=\alpha*a_i*b_i&space;)
![y_i = v_1 + v_2 + ... + v_n + beta * c_i](https://latex.codecogs.com/svg.image?y_i=v_1&plus;v_2&plus;...&plus;v_n&plus;\beta*c_i&space;)

By grouping the operations we get:
![Y = alpha * A * B + beta * C](https://latex.codecogs.com/svg.image?Y=\alpha\cdot&space;A\cdot&space;B&plus;\beta\cdot&space;C&space;)

### Flatten
Just identity mapping because the wires are always Flatten.
![out_i ~ in_i](https://latex.codecogs.com/svg.image?out_i\sim&space;in_i)

### MatMul
Equal to Gemm with ![alpha=1](https://latex.codecogs.com/svg.image?\inline&space;\alpha=1), ![beta=0](https://latex.codecogs.com/svg.image?\inline&space;\beta=0) and ![C=0](https://latex.codecogs.com/svg.image?\inline&space;&space;C=0).

### Reshape
Just identity mapping because the wires always Flatten.
![out_i ~ iin_i](https://latex.codecogs.com/svg.image?out_i\sim&space;in_i)

### Add
ONNX Add node is defined as:
![C = A + B](https://latex.codecogs.com/svg.image?&space;C=A&plus;B)

The translation defines the interactions:
![Add(c_i, b_i) ~ a_i](https://latex.codecogs.com/svg.image?Add(c_i,b_i)\sim&space;a_i)

By definition this interaction is equal to:
![c_i = a_i + b_i](https://latex.codecogs.com/svg.image?c_i=a_i&plus;b_i)

By grouping the operations we get:
![C = A + B](https://latex.codecogs.com/svg.image?C=A&plus;B)

### Sub
ONNX Sub node is defined as:
![C = A - B](https://latex.codecogs.com/svg.image?C=A-B)

The translation defines the interactions:
![Add(c_i, neg_b_i) ~ a_i](https://latex.codecogs.com/svg.image?Add(c_i,neg_i)\sim&space;a_i)
![Mul(neg_b_i, Concrete(-1)) ~ b_i](https://latex.codecogs.com/svg.image?Mul(neg_i,Concrete(-1))\sim&space;b_i)

By definition this interaction is equal to:
![c_i = a_i + neg_b_i](https://latex.codecogs.com/svg.image?c_i=a_i&plus;neg_i)
![neg_b_i = -1 * b_i](https://latex.codecogs.com/svg.image?neg_i=-1*b_i)

By grouping the operations we get:
![C = A - B](https://latex.codecogs.com/svg.image?C=A-B)

## Soundness of Interaction Rules
### Materialize
The Materialize agent transforms a Linear agent into a tree of explicit mathematical operations
that are used as final representation for the solver.
In the Python module the terms are defined as:
```python
def TermAdd(a, b):
    return a + b  
def TermMul(a, b):
    return a * b
def TermReLU(x):
    return z3.If(x > 0, x, 0)
```

#### Linear >< Materialize
![Linear(x, q, r) >< Materialize(out) => (1), (2), (3), (4), (5)](https://latex.codecogs.com/svg.image?\inline&space;&space;Linear(x,q,r)><Materialize(out)\Rightarrow&space;(1),(2),(3),(4),(5))

LHS:
![Linear(x, q, r) ~ wire](https://latex.codecogs.com/svg.image?Linear(x,q,r)\sim&space;wire)
![Materialize(out) ~ wire](https://latex.codecogs.com/svg.image?Materialize(out)\sim&space;wire)
![q*x + r = wire](https://latex.codecogs.com/svg.image?q*x&plus;r=wire)
![out = wire](https://latex.codecogs.com/svg.image?out=wire)
![out = q*x + r](https://latex.codecogs.com/svg.image?out=q*x&plus;r)

##### Case 1:
![q = 0 => out ~ Concrete(r), x ~ Eraser](https://latex.codecogs.com/svg.image?q=0\Rightarrow&space;out\sim&space;Concrete(r),x\sim&space;Eraser)

RHS:
![out = r](https://latex.codecogs.com/svg.image?out=r)

EQUIVALENCE:
![0*x + r = r => r = r](https://latex.codecogs.com/svg.image?0*x&plus;r=r\Rightarrow&space;r=r)

##### Case 2:
![q = 1, r = 0 => out ~ x](https://latex.codecogs.com/svg.image?q=1,r=0\Rightarrow&space;out~x)

RHS:
![x = out](https://latex.codecogs.com/svg.image?x=out)
![out = x](https://latex.codecogs.com/svg.image?out=x)

EQUIVALENCE:
![1*x + 0 = x => x = x](https://latex.codecogs.com/svg.image?1*x&plus;0=x\Rightarrow&space;x=x)

##### Case 3:
![q = 1 => out ~ TermAdd(x, Concrete(r))](https://latex.codecogs.com/svg.image?q=1\Rightarrow&space;out\sim&space;TermAdd(x,Concrete(r)))

RHS:
![out = x + r](https://latex.codecogs.com/svg.image?out=x&plus;r)

EQUIVALENCE:
![1*x + r = x + r => x + r = x + r](https://latex.codecogs.com/svg.image?1*x&plus;r=x&plus;r\Rightarrow&space;x&plus;r=x&plus;r)

##### Case 4:
![r = 0 => out ~ TermMul(Concrete(q), x)](https://latex.codecogs.com/svg.image?r=0\Rightarrow&space;out\sim&space;TermMul(Concrete(q),x))

RHS:
![out = q*x](https://latex.codecogs.com/svg.image?out=q*x)

EQUIVALENCE:
![q*x + 0 = q*x => q*x = q*x](https://latex.codecogs.com/svg.image?q*x&plus;0=q*x\Rightarrow&space;q*x=q*x)

##### Case 5:
![otherwise => out ~ TermAdd(TermMul(Concrete(q), x), Concrete(r))](https://latex.codecogs.com/svg.image?otherwise\Rightarrow&space;out\sim&space;TermAdd(TermMul(Concrete(q),x),Concrete(r)))

RHS:
![out = q*x + r](https://latex.codecogs.com/svg.image?out=q*x&plus;r)

EQUIVALENCE:
![q*x + r = q*x + r](https://latex.codecogs.com/svg.image?q*x&plus;r=q*x&plus;r)

#### Concrete >< Materialize
![Concrete(k) >< Materialize(out) => out ~ Concrete(k)](https://latex.codecogs.com/svg.image?Concrete(k)><Materialize(out)\Rightarrow&space;out\sim&space;Concrete(k))

LHS:
![Concrete(k) ~ wire](https://latex.codecogs.com/svg.image?Concrete(k)\sim&space;wire)
![Materialize(out) ~ wire](https://latex.codecogs.com/svg.image?Materialize(out)\sim&space;wire)
![k = wire](https://latex.codecogs.com/svg.image?k=wire)
![out = wire](https://latex.codecogs.com/svg.image?out=wire)
![out = k](https://latex.codecogs.com/svg.image?out=k)

RHS:
![out = k](https://latex.codecogs.com/svg.image?out=k)

EQUIVALENCE:
![k = k](https://latex.codecogs.com/svg.image?k=k)

### Add
#### Linear >< Add
![Linear(x, q, r) >< Add(out, b) => b ~ AddCheckLinear(out, x, q, r)](https://latex.codecogs.com/svg.image?Linear(x,q,r)><Add(out,b)\Rightarrow&space;b\sim&space;AddCheckLinear(out,x,q,r))

LHS:
![Linear(x, q, r) ~ wire](https://latex.codecogs.com/svg.image?Linear(x,q,r)\sim&space;wire)
![Add(out, b) ~ wire](https://latex.codecogs.com/svg.image?Add(out,b)\sim&space;wire)
![q*x + r = wire](https://latex.codecogs.com/svg.image?q*x&plus;r=wire)
![out = wire + b](https://latex.codecogs.com/svg.image?out=wire&plus;b)
![out = q*x + r + b](https://latex.codecogs.com/svg.image?out=q*x&plus;r&plus;b)

RHS:
![out = q*x + (r + b)](https://latex.codecogs.com/svg.image?out=q*x&plus;(r&plus;b))

EQUIVALENCE:
![q*x + r + b = q*x + (r + b) => q*x + (r + b) = q*x + (r + b)](https://latex.codecogs.com/svg.image?q*x&plus;r&plus;b=q*x&plus;(r&plus;b)\Rightarrow&space;q*x&plus;(r&plus;b)=q*x&plus;(r&plus;b))

#### Concrete >< Add
![Concrete(k) >< Add(out, b) => (1), (2)](https://latex.codecogs.com/svg.image?Concrete(k)><Add(out,b)\Rightarrow&space;(1),(2))

LHS:
![Concrete(k) ~ wire](https://latex.codecogs.com/svg.image?Concrete(k)\sim&space;wire)
![Add(out, b) ~ wire](https://latex.codecogs.com/svg.image?Add(out,b)\sim&space;wire)
![k = wire](https://latex.codecogs.com/svg.image?k=wire)
![out = wire + b](https://latex.codecogs.com/svg.image?out=wire&plus;b)
![out = k + b](https://latex.codecogs.com/svg.image?out=k&plus;b)

##### Case 1:
![k = 0 => out ~ b](https://latex.codecogs.com/svg.image?k=0\Rightarrow&space;out\sim&space;b)

RHS:
![out = b](https://latex.codecogs.com/svg.image?out=b)

EQUIVALENCE:
![0 + b = b => b = b](https://latex.codecogs.com/svg.image?0&plus;b=b\Rightarrow&space;b=b)

##### Case 2:
![otherwise => b ~ AddCheckConcrete(out, k)](https://latex.codecogs.com/svg.image?otherwise\Rightarrow&space;b\sim&space;AddCheckConcrete(out,k))

RHS:
![out = k + b](https://latex.codecogs.com/svg.image?out=k&plus;b)

EQUIVALENCE:
![k + b = k + b](https://latex.codecogs.com/svg.image?k&plus;b=k&plus;b)

#### Linear >< AddCheckLinear
![Linear(y, s, t) >< AddCheckLinear(out, x, q, r) => (1), (2), (3), (4)](https://latex.codecogs.com/svg.image?Linear(y,s,t)><AddCheckLinear(out,x,q,r)\Rightarrow&space;(1),(2),(3),(4))

LHS:
![Linear(y, s, t) ~ wire](https://latex.codecogs.com/svg.image?Linear(y,s,t)\sim&space;wire)
![AddCheckLinear(out, x, q, r) ~ wire](https://latex.codecogs.com/svg.image?AddCheckLinear(out,x,q,r)\sim&space;wire)
![s*y + t = wire](https://latex.codecogs.com/svg.image?s*y&plus;t=wire)
![out = q*x + (r + wire)](https://latex.codecogs.com/svg.image?out=q*x&plus;(r&plus;wire))
![out = q*x + (r + s*y + t)](https://latex.codecogs.com/svg.image?out=q*x&plus;(r&plus;s*y&plus;t))

##### Case 1:
![q,r,s,t = 0 => out ~ Concrete(0), x ~ Eraser, y ~ Eraser](https://latex.codecogs.com/svg.image?q,r,s,t=0\Rightarrow&space;out\sim&space;Concrete(0),x\sim&space;Eraser,y\sim&space;Eraser)

RHS:
![out = 0](https://latex.codecogs.com/svg.image?out=0)

EQUIVALENCE:
![0*x + (0 + 0*y + 0) = 0 => 0 = 0](https://latex.codecogs.com/svg.image?0*x&plus;(0&plus;0*y&plus;0)=0\Rightarrow&space;0=0)

##### Case 2:
![s,t = 0 => out ~ Linear(x, q, r), y ~ Eraser](https://latex.codecogs.com/svg.image?s,t=0\Rightarrow&space;out\sim&space;Linear(x,q,r),y\sim&space;Eraser)

RHS:
![out = q*x + r](https://latex.codecogs.com/svg.image?out=q*x&plus;r)

EQUIVALENCE:
![q*x + (r + 0*y + 0) = q*x + r => q*x + r = q*x + r](https://latex.codecogs.com/svg.image?q*x&plus;(r&plus;0*y&plus;0)=q*x&plus;r\Rightarrow&space;q*x&plus;r=q*x&plus;r)

##### Case 3:
![q, r = 0 => out ~ Linear(y, s, t), x ~ Eraser](https://latex.codecogs.com/svg.image?q,r=0\Rightarrow&space;out\sim&space;Linear(y,s,t),x\sim&space;Eraser)

RHS:
![out = s*y + t](https://latex.codecogs.com/svg.image?out=s*y&plus;t)

EQUIVALENCE:
![0*x + (0 + s*y + t) = s*y + t => s*y + t = s*y + t](https://latex.codecogs.com/svg.image?0*x&plus;(0&plus;s*y&plus;t)=s*y&plus;t\Rightarrow&space;s*y&plus;t=s*y&plus;t)

##### Case 4:
![otherwise => Linear(x, q, r) ~ Materialize(out_x), Linear(y, s, t) ~ Materialize(out_y), out ~ Linear(TermAdd(out_x, out_y), 1, 0)](https://latex.codecogs.com/svg.image?otherwise\Rightarrow&space;Linear(x,q,r)\sim&space;Materialize(out_x),Linear(y,s,t)\sim&space;Materialize(out_y),out\sim&space;Linear(TermAdd(out_x,out_y),1,0))

RHS:
![Linear(x, q, r) ~ wire_1](https://latex.codecogs.com/svg.image?Linear(x,q,r)\sim&space;wire_1)
![Materialize(out_x) ~ wire_1](https://latex.codecogs.com/svg.image?Materialize(out_x)\sim&space;wire_1)
![q*x + r = wire_1](https://latex.codecogs.com/svg.image?q*x&plus;r=wire_1)
![out_x = wire_1](https://latex.codecogs.com/svg.image?out_x=wire_1)
![Linear(y, s, t) ~ wire_2](https://latex.codecogs.com/svg.image?Linear(y,s,t)\sim&space;wire_2)
![Materialize(out_y) ~ wire_2](https://latex.codecogs.com/svg.image?Materialize(out_y)\sim&space;wire_2)
![s*y + t = wire_2](https://latex.codecogs.com/svg.image?s*y&plus;t=wire_2)
![out_y = wire_2](https://latex.codecogs.com/svg.image?out_y=wire_2)
![out = 1*TermAdd(out_x, out_y) + 0](https://latex.codecogs.com/svg.image?out=1*TermAdd(out_x,out_y)&plus;0)
Because `TermAdd(a, b)` is defined as `a+b`:
![out = 1*(q*x + r + s*y + t) + 0](https://latex.codecogs.com/svg.image?out=1*(q*x&plus;r&plus;s*y&plus;t)&plus;0)

EQUIVALENCE:
![q*x + (r + s*y + t) = 1*(q*x + r + s*y + t) + 0 => q*x + r + s*y + t = q*x + r + s*y + t](https://latex.codecogs.com/svg.image?q*x&plus;(r&plus;s*y&plus;t)=1*(q*x&plus;r&plus;s*y&plus;t)&plus;0\Rightarrow&space;q*x&plus;r&plus;s*y&plus;t=q*x&plus;r&plus;s*y&plus;t)

#### Concrete >< AddCheckLinear
![Concrete(j) >< AddCheckLinear(out, x, q, r) => out ~ Linear(x, q, r + j)](https://latex.codecogs.com/svg.image?Concrete(j)><AddCheckLinear(out,x,q,r)\Rightarrow&space;out\sim&space;Linear(x,q,r&plus;j))

LHS:
![Concrete(j) ~ wire](https://latex.codecogs.com/svg.image?Concrete(j)\sim&space;wire)
![AddCheckLinear(out, x, q, r) ~ wire](https://latex.codecogs.com/svg.image?AddCheckLinear(out,x,q,r)\sim&space;wire)
![j = wire](https://latex.codecogs.com/svg.image?j=wire)
![out = q*x + (r + wire)](https://latex.codecogs.com/svg.image?out=q*x&plus;(r&plus;wire))
![out = q*x + (r + j)](https://latex.codecogs.com/svg.image?out=q*x&plus;(r&plus;j))

RHS:
![out = q*x + (r + j)](https://latex.codecogs.com/svg.image?out=q*x&plus;(r&plus;j))

EQUIVALENCE:
![q*x + (r + j) = q*x + (r + j)](https://latex.codecogs.com/svg.image?q*x&plus;(r&plus;j)=q*x&plus;(r&plus;j))

#### Linear >< AddCheckConcrete
![Linear(y, s, t) >< AddCheckConcrete(out, k) => out ~ Linear(y, s, t + k)](https://latex.codecogs.com/svg.image?Linear(y,s,t)><AddCheckConcrete(out,k)\Rightarrow&space;out\sim&space;Linear(y,s,t&plus;k))

LHS:
![Linear(y, s, t) ~ wire](https://latex.codecogs.com/svg.image?Linear(y,s,t)\sim&space;wire)
![AddCheckConcrete(out, k) ~ wire](https://latex.codecogs.com/svg.image?AddCheckConcrete(out,k)\sim&space;wire)
![s*y + t = wire](https://latex.codecogs.com/svg.image?s*y&plus;t=wire)
![out = k + wire](https://latex.codecogs.com/svg.image?out=k&plus;wire)
![out = k + s*y + t](https://latex.codecogs.com/svg.image?out=k&plus;s*y&plus;t)

RHS:
![out = s*y + (t + k)](https://latex.codecogs.com/svg.image?out=s*y&plus;(t&plus;k))

EQUIVALENCE:
![k + s*y + t = s*y + (t + k) => s*y + (t + k) = s*y + (t + k)](https://latex.codecogs.com/svg.image?k&plus;s*y&plus;t=s*y&plus;(t&plus;k)\Rightarrow&space;s*y&plus;(t&plus;k)=s*y&plus;(t&plus;k))

#### Concrete >< AddCheckConcrete
![Concrete(j) >< AddCheckConcrete(out, k) => (1), (2)](https://latex.codecogs.com/svg.image?Concrete(j)><AddCheckConcrete(out,k)\Rightarrow&space;(1),(2))

LHS:
![Concrete(j) ~ wire](https://latex.codecogs.com/svg.image?Concrete(j)\sim&space;wire)
![AddCheckConcrete(out, k) ~ wire](https://latex.codecogs.com/svg.image?AddCheckConcrete(out,k)\sim&space;wire)
![j = wire](https://latex.codecogs.com/svg.image?j=wire)
![out = k + wire](https://latex.codecogs.com/svg.image?out=k&plus;wire)
![out = k + j](https://latex.codecogs.com/svg.image?out=k&plus;j)

##### Case 1:
![j = 0 => out ~ Concrete(k)](https://latex.codecogs.com/svg.image?j=0\Rightarrow&space;out\sim&space;Concrete(k))

RHS:
![out = k](https://latex.codecogs.com/svg.image?out=k)

EQUIVALENCE:
![k + 0 = k => k = k](https://latex.codecogs.com/svg.image?k&plus;0=k\Rightarrow&space;k=k)

##### Case 2:
![otherwise => out ~ Concrete(k + j)](https://latex.codecogs.com/svg.image?otherwise\Rightarrow&space;out\sim&space;Concrete(k&plus;j))

RHS:
![out = k + j](https://latex.codecogs.com/svg.image?out=k&plus;j)

EQUIVALENCE:
![k + j = k + j](https://latex.codecogs.com/svg.image?k&plus;j=k&plus;j)

### Mul
#### Linear >< Mul
![Linear(x, q, r) >< Mul(out, b) => b ~ MulCheckLinear(out, x, q, r)](https://latex.codecogs.com/svg.image?Linear(x,q,r)><Mul(out,b)\Rightarrow&space;b\sim&space;MulCheckLinear(out,x,q,r))

LHS:
![Linear(x, q, r) ~ wire](https://latex.codecogs.com/svg.image?Linear(x,q,r)\sim&space;wire)
![Mul(out, b) ~ wire](https://latex.codecogs.com/svg.image?Mul(out,b)\sim&space;wire)
![q*x + r = wire](https://latex.codecogs.com/svg.image?q*x&plus;r=wire)
![out = wire * b](https://latex.codecogs.com/svg.image?out=wire*b)
![out = (q*x + r) * b](https://latex.codecogs.com/svg.image?out=(q*x&plus;r)*b)

RHS:
![out = q*b*x + r*b](https://latex.codecogs.com/svg.image?out=q*b*x&plus;r*b)

EQUIVALENCE:
![(q*x + r) * b = q*b*x + r*b => q*b*x + r*b = q*b*x + r*b](https://latex.codecogs.com/svg.image?(q*x&plus;r)*b=q*b*x&plus;r*b\Rightarrow&space;q*b*x&plus;r*b=q*b*x&plus;r*b)

#### Concrete >< Mul
![Concrete(k) >< Mul(out, b) => (1), (2), (3)](https://latex.codecogs.com/svg.image?Concrete(k)><Mul(out,b)\Rightarrow&space;(1),(2),(3))

LHS:
![Concrete(k) ~ wire](https://latex.codecogs.com/svg.image?Concrete(k)\sim&space;wire)
![Mul(out, b) ~ wire](https://latex.codecogs.com/svg.image?Mul(out,b)\sim&space;wire)
![k = wire](https://latex.codecogs.com/svg.image?k=wire)
![out = wire * b](https://latex.codecogs.com/svg.image?out=wire*b)
![out = k * b](https://latex.codecogs.com/svg.image?out=k*b)

##### Case 1:
![k = 0 => out ~ Concrete(0), b ~ Eraser](https://latex.codecogs.com/svg.image?k=0\Rightarrow&space;out\sim&space;Concrete(0),b\sim&space;Eraser)

RHS:
![out = 0](https://latex.codecogs.com/svg.image?out=0)

EQUIVALENCE:
![0 * b = 0 => 0 = 0](https://latex.codecogs.com/svg.image?0*b=0\Rightarrow&space;0=0)

##### Case 2:
![k = 1 => out ~ b](https://latex.codecogs.com/svg.image?k=1\Rightarrow&space;out\sim&space;b)

RHS:
![out = b](https://latex.codecogs.com/svg.image?out=b)

EQUIVALENCE:
![1 * b = b => b = b](https://latex.codecogs.com/svg.image?1*b=b\Rightarrow&space;b=b)

##### Case 3:
![otherwise => b ~ MulCheckConcrete(out, k)](https://latex.codecogs.com/svg.image?otherwise\Rightarrow&space;b\sim&space;MulCheckConcrete(out,k))

RHS:
![out = k * b](https://latex.codecogs.com/svg.image?out=k*b)

EQUIVALENCE:
![k * b = k * b](https://latex.codecogs.com/svg.image?k*b=k*b)

#### Linear >< MulCheckLinear
![Linear(y, s, t) >< MulCheckLinear(out, x, q, r) => (1), (2)](https://latex.codecogs.com/svg.image?Linear(y,s,t)><MulCheckLinear(out,x,q,r)\Rightarrow&space;(1),(2))

LHS:
![Linear(y, s, t) ~ wire](https://latex.codecogs.com/svg.image?Linear(y,s,t)\sim&space;wire)
![MulCheckLinear(out, x, q, r) ~ wire](https://latex.codecogs.com/svg.image?MulCheckLinear(out,x,q,r)\sim&space;wire)
![s*y + t = wire](https://latex.codecogs.com/svg.image?s*y&plus;t=wire)
![out = q*wire*x + r*wire](https://latex.codecogs.com/svg.image?out=q*wire*x&plus;r*wire)
![out = q*(s*y + t)*x + r*(s*y + t)](https://latex.codecogs.com/svg.image?out=q*(s*y&plus;t)*x&plus;r*(s*y&plus;t))

##### Case 1:
![(q,r = 0) or (s,t = 0) => x ~ Eraser, y ~ Eraser, out ~ Concrete(0)](https://latex.codecogs.com/svg.image?(q,r=0)\lor(s,t=0)\Rightarrow&space;x\sim&space;Eraser,y\sim&space;Eraser,out\sim&space;Concrete(0))

RHS:
![out = 0](https://latex.codecogs.com/svg.image?out=0)

EQUIVALENCE:
![0*(s*y + t)*x + 0*(s*y + t) = 0 => 0 = 0](https://latex.codecogs.com/svg.image?0*(s*y&plus;t)*x&plus;0*(s*y&plus;t)=0\Rightarrow&space;0=0)
![or](https://latex.codecogs.com/svg.image?\lor)
![q*(0*y + 0)*x + r*(0*y + 0) = 0 => 0 = 0](https://latex.codecogs.com/svg.image?q*(0*y&plus;0)*x&plus;r*(0*y&plus;0)=0\Rightarrow&space;0=0)

##### Case 2:
![otherwise => Linear(x, q, r) ~ Materialize(out_x), Linear(y, s, t) ~ Materialize(out_y), out ~ Linear(TermMul(out_x, out_y), 1, 0)](https://latex.codecogs.com/svg.image?otherwise\Rightarrow&space;Linear(x,q,r)\sim&space;Materialize(out_x),Linear(y,s,t)\sim&space;Materialize(out_y),out\sim&space;Linear(TermMul(out_x,out_y),1,0))

RHS:
![Linear(x, q, r) ~ wire_1](https://latex.codecogs.com/svg.image?Linear(x,q,r)\sim&space;wire_1)
![Materialize(out_x) ~ wire_1](https://latex.codecogs.com/svg.image?Materialize(out_x)\sim&space;wire_1)
![q*x + r = wire_1](https://latex.codecogs.com/svg.image?q*x&plus;r=wire_1)
![out_x = wire_1](https://latex.codecogs.com/svg.image?out_x=wire_1)
![Linear(y, s, t) ~ wire_2](https://latex.codecogs.com/svg.image?Linear(y,s,t)\sim&space;wire_2)
![Materialize(out_y) ~ wire_2](https://latex.codecogs.com/svg.image?Materialize(out_y)\sim&space;wire_2)
![s*y + t = wire_2](https://latex.codecogs.com/svg.image?s*y&plus;t=wire_2)
![out_y = wire_2](https://latex.codecogs.com/svg.image?out_y=wire_2)
![out = 1*TermMul(out_x, out_y) + 0](https://latex.codecogs.com/svg.image?out=1*TermMul(out_x,out_y)&plus;0)
Because `TermMul(a, b)` is defined as `a*b`:
![out = 1*(q*x + r)*(s*y + t) + 0](https://latex.codecogs.com/svg.image?out=1*(q*x&plus;r)*(s*y&plus;t)&plus;0)

EQUIVALENCE:
![q*(s*y + t)*x + r*(s*y + t) = 1*(q*x + r)*(s*y + t) => 
q*(s*y + t)*x + r*(s*y + t) = (q*x + r)*(s*y + t) => 
q*(s*y + t)*x + r*(s*y + t) = q*(s*y + t)*x + r*(s*y + t)](https://latex.codecogs.com/svg.image?q*(s*y&plus;t)*x&plus;r*(s*y&plus;t)=1*(q*x&plus;r)*(s*y&plus;t)\Rightarrow&space;q*(s*y&plus;t)*x&plus;r*(s*y&plus;t)=(q*x&plus;r)*(s*y&plus;t)\Rightarrow&space;q*(s*y&plus;t)*x&plus;r*(s*y&plus;t)=q*(s*y&plus;t)*x&plus;r*(s*y&plus;t))


#### Concrete >< MulCheckLinear
![Concrete(j) >< MulCheckLinear(out, x, q, r) => out ~ Linear(x, q * j, r * j)](https://latex.codecogs.com/svg.image?Concrete(j)><MulCheckLinear(out,x,q,r)\Rightarrow&space;out\sim&space;Linear(x,q*j,r*j))

LHS:
![Concrete(j) ~ wire](https://latex.codecogs.com/svg.image?Concrete(j)\sim&space;wire)
![MulCheckLinear(out, x, q, r) ~ wire](https://latex.codecogs.com/svg.image?MulCheckLinear(out,x,q,r)\sim&space;wire)
![j = wire](https://latex.codecogs.com/svg.image?j=wire)
![out = q*wire*x + r*wire](https://latex.codecogs.com/svg.image?out=q*wire*x&plus;r*wire)
![out = q*j*x + r*j](https://latex.codecogs.com/svg.image?out=q*j*x&plus;r*j)

RHS:
![out = q*j*x + r*j](https://latex.codecogs.com/svg.image?out=q*j*x&plus;r*j)

EQUIVALENCE:
![q*j*x + r*j = q*j*x + r*j](https://latex.codecogs.com/svg.image?q*j*x&plus;r*j=q*j*x&plus;r*j)

#### Linear >< MulCheckConcrete
![Linear(y, s, t) >< MulCheckConcrete(out, k) => out ~ Linear(y, s * k, t * k)](https://latex.codecogs.com/svg.image?Linear(y,s,t)><MulCheckConcrete(out,k)\Rightarrow&space;out\sim&space;Linear(y,s*k,t*k))

LHS:
![Linear(y, s, t) ~ wire](https://latex.codecogs.com/svg.image?Linear(y,s,t)\sim&space;wire)
![MulCheckConcrete(out, k) ~ wire](https://latex.codecogs.com/svg.image?MulCheckConcrete(out,k)\sim&space;wire)
![s*y + t = wire](https://latex.codecogs.com/svg.image?s*y&plus;t=wire)
![out = k * wire](https://latex.codecogs.com/svg.image?out=k*wire)
![out = k * (s*y + t)](https://latex.codecogs.com/svg.image?out=k*(s*y&plus;t))

RHS:
![out = s*k*y + t*k](https://latex.codecogs.com/svg.image?out=s*k*y&plus;t*k)

EQUIVALENCE:
![k * (s*y + t) = s*k*y + t*k => s*k*y + t*k = s*k*y + t*k](https://latex.codecogs.com/svg.image?k*(s*y&plus;t)=s*k*y&plus;t*k\Rightarrow&space;s*k*y&plus;t*k=s*k*y&plus;t*k)


#### Concrete >< MulCheckConcrete
![Concrete(j) >< MulCheckConcrete(out, k) => (1), (2), (3)](https://latex.codecogs.com/svg.image?Concrete(j)><MulCheckConcrete(out,k)\Rightarrow&space;(1),(2),(3))

LHS:
![Concrete(j) ~ wire](https://latex.codecogs.com/svg.image?Concrete(j)\sim&space;wire)
![MulCheckConcrete(out, k) ~ wire](https://latex.codecogs.com/svg.image?MulCheckConcrete(out,k)\sim&space;wire)
![j = wire](https://latex.codecogs.com/svg.image?j=wire)
![out = k * wire](https://latex.codecogs.com/svg.image?out=k*wire)
![out = k * j](https://latex.codecogs.com/svg.image?out=k*j)

##### Case 1:
![j = 0 => out ~ Concrete(0)](https://latex.codecogs.com/svg.image?j=0\Rightarrow&space;out\sim&space;Concrete(0))

RHS:
![out = 0](https://latex.codecogs.com/svg.image?out=0)

EQUIVALENCE:
![k * 0 = 0 => 0 = 0](https://latex.codecogs.com/svg.image?k*0=0\Rightarrow&space;0=0)

##### Case 2:
![j = 1 => out ~ Concrete(k)](https://latex.codecogs.com/svg.image?j=1\Rightarrow&space;out\sim&space;Concrete(k))

RHS:
![out = k](https://latex.codecogs.com/svg.image?out=k)

EQUIVALENCE:
![k * 1 = k => k = k](https://latex.codecogs.com/svg.image?k*1=k\Rightarrow&space;k=k)

##### Case 3:
![otherwise => out ~ Concrete(k * j)](https://latex.codecogs.com/svg.image?otherwise\Rightarrow&space;out\sim&space;Concrete(k*j))

RHS:
![out = k * j](https://latex.codecogs.com/svg.image?out=k*j)

EQUIVALENCE:
![k * j = k * j](https://latex.codecogs.com/svg.image?k*j=k*j)

### ReLU
#### Linear >< ReLU
![Linear(x, q, r) >< ReLU(out) => Linear(x, q, r) ~ Materialize(out_x), out ~ Linear(TermReLU(out_x), 1, 0)](https://latex.codecogs.com/svg.image?Linear(x,q,r)><ReLU(out)\Rightarrow&space;Linear(x,q,r)\sim&space;Materialize(out_x),out\sim&space;Linear(TermReLU(out_x),1,0))

LHS:
![Linear(x, q, r) ~ wire](https://latex.codecogs.com/svg.image?Linear(x,q,r)\sim&space;wire)
![ReLU(out) ~ wire](https://latex.codecogs.com/svg.image?ReLU(out)\sim&space;wire)
![q*x + r = wire](https://latex.codecogs.com/svg.image?q*x&plus;r=wire)
![out = IF wire > 0 THEN wire ELSE 0](https://latex.codecogs.com/svg.image?out=IF\;wire>0\;THEN\;wire\;ELSE\;0)
![out = IF (q*x + r) > 0 THEN (q*x + r) ELSE 0](https://latex.codecogs.com/svg.image?out=IF\;(q*x&plus;r)>0\;THEN\;(q*x&plus;r)\;ELSE\;0)

RHS:
![Linear(x, q, r) ~ wire](https://latex.codecogs.com/svg.image?Linear(x,q,r)\sim&space;wire)
![Materialize(out_x) ~ wire](https://latex.codecogs.com/svg.image?Materialize(out_x)\sim&space;wire)
![q*x + r = wire](https://latex.codecogs.com/svg.image?q*x&plus;r=wire)
![out_x = wire](https://latex.codecogs.com/svg.image?out_x=wire)
![out = 1*TermReLU(out_x) + 0](https://latex.codecogs.com/svg.image?out=1*TermReLU(out_x)&plus;0)
Because `TermReLU(x)` is defined as `z3.If(x > 0, x, 0)`:
![out = 1*(IF (q*x + r) > 0 THEN (q*x + r) ELSE 0) + 0](https://latex.codecogs.com/svg.image?out=1*(IF\;(q*x&plus;r)>0\;THEN\;(q*x&plus;r)\;ELSE\;0)&plus;0)

EQUIVALENCE:
![IF (q*x + r) > 0 THEN (q*x + r) ELSE 0 = 1*(IF (q*x + r) > 0 THEN (q*x + r) ELSE 0) + 0 =>
IF (q*x + r) > 0 THEN (q*x + r) ELSE 0 = IF (q*x + r) > 0 THEN (q*x + r) ELSE 0](https://latex.codecogs.com/svg.image?IF\;(q*x&plus;r)>0\;THEN\;(q*x&plus;r)\;ELSE\;0=1*(IF\;(q*x&plus;r)>0\;THEN\;(q*x&plus;r)\;ELSE\;0)&plus;0\Rightarrow&space;IF\;(q*x&plus;r)>0\;THEN\;(q*x&plus;r)\;ELSE\;0=IF\;(q*x&plus;r)>0\;THEN\;(q*x&plus;r)\;ELSE\;0)


#### Concrete >< ReLU
![Concrete(k) >< ReLU(out) => (1), (2)](https://latex.codecogs.com/svg.image?Concrete(k)><ReLU(out)\Rightarrow&space;(1),(2))

LHS:
![Concrete(k) ~ wire](https://latex.codecogs.com/svg.image?Concrete(k)\sim&space;wire)
![ReLU(out) ~ wire](https://latex.codecogs.com/svg.image?ReLU(out)\sim&space;wire)
![k = wire](https://latex.codecogs.com/svg.image?k=wire)
![out = IF wire > 0 THEN wire ELSE 0](https://latex.codecogs.com/svg.image?out=IF\;wire>0\;THEN\;wire\;ELSE\;0)
![out = IF k > 0 THEN k ELSE 0](https://latex.codecogs.com/svg.image?out=IF\;k>0\;THEN\;k\;ELSE\;0)

##### Case 1:
![k > 0 => out ~ Concrete(k)](https://latex.codecogs.com/svg.image?k>0\Rightarrow&space;out\sim&space;Concrete(k))

RHS:
![out = k](https://latex.codecogs.com/svg.image?out=k)

EQUIVALENCE:
![IF true THEN k ELSE 0 = k => k = k](https://latex.codecogs.com/svg.image?IF\;true\;THEN\;k\;ELSE\;0=k\Rightarrow&space;k=k)

##### Case 2:
![k <= 0 => out ~ Concrete(0)](https://latex.codecogs.com/svg.image?k\leq&space;0\Rightarrow&space;out\sim&space;Concrete(0))

RHS:
![out = 0](https://latex.codecogs.com/svg.image?out=0)

EQUIVALENCE:
![IF false THEN k ELSE 0 = 0 => 0 = 0](https://latex.codecogs.com/svg.image?IF\;false\;THEN\;k\;ELSE\;0=0\Rightarrow&space;0=0)

## Soundness of Reduction 
Let ![IN_0](https://latex.codecogs.com/svg.image?\mathrm{IN_0}) be the Interaction Net translated from a Neural Network ![NN](https://latex.codecogs.com/svg.image?\inline&space;\mathrm{NN}). Let ![IN_n](https://latex.codecogs.com/svg.image?\inline&space;\mathrm{IN_n}) be the state of the net
after ![n](https://latex.codecogs.com/svg.image?\inline&space;n) reduction steps. Then ![forall n in N, [IN_n] = [NN]](https://latex.codecogs.com/svg.image?\inline&space;\forall&space;n\in\mathbb{N},[\mathrm{IN_n}]=[\mathrm{NN}]).

### Proof by Induction
- Base Case (![n = 0](https://latex.codecogs.com/svg.image?\inline&space;n=0)): By the [Soundness of Translation](#soundness-of-translation), the initial net ![IN_0](https://latex.codecogs.com/svg.image?\mathrm{IN_0}) is constructed such that
its semantics ![[IN_0]](https://latex.codecogs.com/svg.image?\inline&space;[\mathrm{IN_0}]) exactly match the mathematical definition of the ONNX nodes in ![NN](https://latex.codecogs.com/svg.image?\inline&space;\mathrm{NN}).
- Induction Step (![n -> n + 1](https://latex.codecogs.com/svg.image?\inline&space;n\to&space;n&plus;1)): Assume ![[IN_n] = [NN]](https://latex.codecogs.com/svg.image?\inline&space;[\mathrm{IN_n}]=[\mathrm{NN}]). If ![IN_n](https://latex.codecogs.com/svg.image?\inline&space;\mathrm{IN_n}) is in normal form, the proof is complete.
Otherwise, there exists an active pair ![A](https://latex.codecogs.com/svg.image?\inline&space;A) that reduces ![IN_n to IN_{n+1}](https://latex.codecogs.com/svg.image?\inline&space;\mathrm{IN_n}\Rightarrow&space;\mathrm{IN_{n&plus;1}}).
By the [Soundness of Interaction Rules](#soundness-of-interaction-rules), the mathematical definition is preserved after any reduction step,
it follows that ![[IN_{n+1}] = [IN_n]](https://latex.codecogs.com/svg.image?\inline&space;[\mathrm{IN_{n&plus;1}}]=[\mathrm{IN_n}]). By the inductive hypothesis, ![[IN_{n+1}] = [NN]](https://latex.codecogs.com/svg.image?\inline&space;[\mathrm{IN_{n&plus;1}}]=[\mathrm{NN}]).

By the principle of mathematical induction, the Interaction Net remains semantically equivalent to the original 
Neural Network at every step of the reduction process.

Since Interaction Nets are confluent, the reduced mathematical expression is unique regardless
of order in which rules are applied.
