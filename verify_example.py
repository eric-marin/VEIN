import sys
import vein

def check_property(onnx_a, onnx_b, smtlib):
    solver = vein.Solver()

    print(f"--- Checking {smtlib} ---")

    solver.load_onnx(onnx_a)
    solver.load_onnx(onnx_b)
    solver.load_smtlib(smtlib)

    result = solver.check()

    if result == vein.unsat:
        print("VERIFIED (UNSAT): The networks are equivalent under this property.\n")
    elif result == vein.sat:
        print(f"FAILED (SAT): The networks are NOT equivalent.\nCounter-example input:\n{solver.model()}\n")
        # m = solver.model()
        # sorted_symbols = sorted([s for s in m.decls() if s.name().startswith("X_")], key=lambda s: s.name())
        # for s in sorted_symbols:
            # print(f"  {s.name()} = {m[s]}")
    else:
        print("UNKNOWN\n")

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("Net not provided\nAvailable Nets: 'xor', 'mnist', 'iris', 'acasxu', 'pendulum', 'double_integrator'")
        sys.exit()

    match sys.argv[1]:
        case "xor":
            net_a = "./examples/xor/xor_a.onnx"
            net_b = "./examples/xor/xor_b.onnx"
            strict = "./examples/xor/xor_strict.smtlib"
            epsilon = "./examples/xor/xor_epsilon.smtlib"
            argmax = "./examples/xor/xor_argmax.smtlib"
        case "mnist":
            net_a = "./examples/mnist/mnist_a.onnx"
            net_b = "./examples/mnist/mnist_b.onnx"
            strict = "./examples/mnist/mnist_strict.smtlib"
            epsilon = "./examples/mnist/mnist_epsilon.smtlib"
            argmax = "./examples/mnist/mnist_argmax.smtlib"
        case "iris":
            net_a = "./examples/iris/iris_a.onnx"
            net_b = "./examples/iris/iris_b.onnx"
            strict = "./examples/iris/iris_strict.smtlib"
            epsilon = "./examples/iris/iris_epsilon.smtlib"
            argmax = "./examples/iris/iris_argmax.smtlib"
        case "acasxu":
            net_a = "./examples/ACASXU/ACASXU_run2a_1_1_batch_2000.onnx"
            net_b = "./examples/ACASXU/ACASXU_run2a_1_1_batch_2000.onnx"
            strict = "./examples/ACASXU/ACASXU_strict.smtlib"
            epsilon = "./examples/ACASXU/ACASXU_epsilon.smtlib"
            argmax = "./examples/ACASXU/ACASXU_argmax.smtlib"
        case "pendulum":
            net_a = "./examples/pendulum/pendulum_pretrain_con.onnx"
            net_b = "./examples/pendulum/pendulum_finetune_con.onnx"
            strict = "./examples/pendulum/pendulum_strict.smtlib"
            epsilon = "./examples/pendulum/pendulum_epsilon.smtlib"
            argmax = "./examples/pendulum/pendulum_argmax.smtlib"
        case "double_integrator":
            net_a = "./examples/double_integrator/double_integrator_pretrain_inv.onnx"
            net_b = "./examples/double_integrator/double_integrator_finetune_inv.onnx"
            strict = "./examples/double_integrator/double_integrator_strict.smtlib"
            epsilon = "./examples/double_integrator/double_integrator_epsilon.smtlib"
            argmax = "./examples/double_integrator/double_integrator_argmax.smtlib"
        case _:
            print("Available Nets: 'xor', 'mnist', 'iris', 'acasxu', 'pendulum', 'double_integrator'")
            sys.exit()

    print(f"=== Comparing {net_a} and {net_b} ===\n")

    check_property(net_a, net_b, strict)
    check_property(net_a, net_b, epsilon)
    check_property(net_a, net_b, argmax)
