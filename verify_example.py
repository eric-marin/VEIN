import sys
import vein

def check_property(onnx_a, onnx_b, vnnlib):
    solver = vein.Solver()

    print(f"--- Checking {vnnlib} ---")

    solver.load_onnx(onnx_a)
    solver.load_onnx(onnx_b)
    solver.load_vnnlib(vnnlib)

    result = solver.check()

    if result == vein.unsat:
        print("VERIFIED (UNSAT): The networks are equivalent under this property.", end="\n\n")
    elif result == vein.sat:
        print("FAILED (SAT): The networks are NOT equivalent.")
        print("Counter-example input:")
        print(solver.model(), end="\n\n")
        # m = solver.model()
        # sorted_symbols = sorted([s for s in m.decls() if s.name().startswith("X_")], key=lambda s: s.name())
        # for s in sorted_symbols:
            # print(f"  {s.name()} = {m[s]}")
    else:
        print("UNKNOWN", end="\n\n")

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("Net not provided")
        print("Available Nets: 'xor', 'mnist', 'iris', 'acasxu', 'pendulum', 'double_integrator'")
        sys.exit()

    match sys.argv[1]:
        case "xor":
            net_a = "./examples/xor/xor_a.onnx"
            net_b = "./examples/xor/xor_b.onnx"
            strict = "./examples/xor/xor_strict.vnnlib"
            epsilon = "./examples/xor/xor_epsilon.vnnlib"
            argmax = "./examples/xor/xor_argmax.vnnlib"
        case "mnist":
            net_a = "./examples/mnist/mnist_a.onnx"
            net_b = "./examples/mnist/mnist_b.onnx"
            strict = "./examples/mnist/mnist_strict.vnnlib"
            epsilon = "./examples/mnist/mnist_epsilon.vnnlib"
            argmax = "./examples/mnist/mnist_argmax.vnnlib"
        case "iris":
            net_a = "./examples/iris/iris_a.onnx"
            net_b = "./examples/iris/iris_b.onnx"
            strict = "./examples/iris/iris_strict.vnnlib"
            epsilon = "./examples/iris/iris_epsilon.vnnlib"
            argmax = "./examples/iris/iris_argmax.vnnlib"
        case "acasxu":
            net_a = "./examples/ACASXU/ACASXU_run2a_1_1_batch_2000.onnx"
            net_b = "./examples/ACASXU/ACASXU_run2a_1_1_batch_2000.onnx"
            strict = "./examples/ACASXU/ACASXU_strict.vnnlib"
            epsilon = "./examples/ACASXU/ACASXU_epsilon.vnnlib"
            argmax = "./examples/ACASXU/ACASXU_argmax.vnnlib"
        case "pendulum":
            net_a = "./examples/pendulum/pendulum_finetune_con.onnx"
            net_b = "./examples/pendulum/pendulum_finetune_con.onnx"
            strict = "./examples/pendulum/pendulum_strict.vnnlib"
            epsilon = "./examples/pendulum/pendulum_epsilon.vnnlib"
            argmax = "./examples/pendulum/pendulum_argmax.vnnlib"
        case "double_integrator":
            net_a = "./examples/double_integrator/double_integrator_finetune_inv.onnx"
            net_b = "./examples/double_integrator/double_integrator_finetune_inv.onnx"
            strict = "./examples/double_integrator/double_integrator_strict.vnnlib"
            epsilon = "./examples/double_integrator/double_integrator_epsilon.vnnlib"
            argmax = "./examples/double_integrator/double_integrator_argmax.vnnlib"
        case _:
            print("Available Nets: 'xor', 'mnist', 'iris', 'acasxu', 'pendulum', 'double_integrator'")
            sys.exit()

    print(f"=== Comparing {net_a} and {net_b} ===", end="\n\n")

    check_property(net_a, net_b, strict)
    check_property(net_a, net_b, epsilon)
    check_property(net_a, net_b, argmax)
