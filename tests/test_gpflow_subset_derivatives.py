"""
Test whether GPflow supports derivative observations at a SUBSET of training points.

GPflow's approach: X has shape (N, 2*D_x)
  - First D_x columns: input locations
  - Last D_x columns: derivative order for each dimension

For a 2D problem:
  - [x1, x2, 0, 0] = function value at (x1, x2)
  - [x1, x2, 1, 0] = df/dx1 at (x1, x2)
  - [x1, x2, 0, 1] = df/dx2 at (x1, x2)

Since each ROW independently specifies its type, we should be able to
have function values at all points but derivatives at only a subset.
"""

import numpy as np

try:
    import gpflow
    from gpflow.kernels import SquaredExponential
    print(f"GPflow version: {gpflow.__version__}")
    GPFLOW_AVAILABLE = True
except ImportError:
    print("GPflow not installed. Install with: pip install gpflow")
    GPFLOW_AVAILABLE = False

try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
except ImportError:
    print("TensorFlow not installed.")


# ============================================================
# Test function: f(x1, x2) = sin(x1) * cos(x2)
# ============================================================
def test_func(X):
    return np.sin(X[:, 0]) * np.cos(X[:, 1])

def test_func_dx1(X):
    return np.cos(X[:, 0]) * np.cos(X[:, 1])

def test_func_dx2(X):
    return -np.sin(X[:, 0]) * np.sin(X[:, 1])


if GPFLOW_AVAILABLE:

    # Check if DifferentialObservationsKernel exists
    print("\n" + "=" * 60)
    print("Checking GPflow derivative kernel availability")
    print("=" * 60)

    # Try different possible locations for the derivative kernel
    deriv_kernel_class = None
    tried = []

    for path in [
        "gpflow.kernels.DifferentialObservationsKernelDynamic",
        "gpflow.kernels.DifferentialObservationsKernel",
        "gpflow.derivative_kernel.DifferentialObservationsKernelDynamic",
        "gpflow.derivative_kernel.DifferentialObservationsKernel",
    ]:
        tried.append(path)
        try:
            parts = path.rsplit(".", 1)
            module = eval(parts[0])
            deriv_kernel_class = getattr(module, parts[1])
            print(f"  FOUND: {path}")
            break
        except (AttributeError, ModuleNotFoundError):
            continue

    if deriv_kernel_class is None:
        print(f"  Derivative kernel NOT found. Tried:")
        for t in tried:
            print(f"    - {t}")
        print(f"\n  Listing available kernels in gpflow.kernels:")
        kernel_names = [k for k in dir(gpflow.kernels) if not k.startswith('_')]
        for name in sorted(kernel_names):
            print(f"    - {name}")

        # Also check if there's a derivative_kernel submodule
        print(f"\n  Checking for derivative_kernel submodule:")
        try:
            import gpflow.derivative_kernel
            print(f"    Found! Contents: {dir(gpflow.derivative_kernel)}")
        except (ImportError, ModuleNotFoundError):
            print(f"    Not found.")

        print("\n  GPflow may not have built-in derivative observation support.")
        print("  This would need to be implemented as a custom kernel.")

    else:
        # ============================================================
        # If derivative kernel exists, run the tests
        # ============================================================
        np.random.seed(42)
        n_all = 20
        n_deriv = 10
        noise_std = 0.05
        D = 2  # input dimension

        X_locs = np.random.rand(n_all, D)
        y_func = test_func(X_locs)
        dy_dx1 = test_func_dx1(X_locs)
        dy_dx2 = test_func_dx2(X_locs)


        # ============================================================
        # TEST 1: Derivatives at ALL points
        # ============================================================
        print("\n" + "=" * 60)
        print("TEST 1: Derivatives at ALL 20 points")
        print("=" * 60)

        try:
            # Build X: each row is [x1, x2, d_order_x1, d_order_x2]
            # Function values: [x1, x2, 0, 0]
            X_func = np.hstack([X_locs, np.zeros((n_all, D))])
            # df/dx1:          [x1, x2, 1, 0]
            X_dx1 = np.hstack([X_locs, np.column_stack([np.ones(n_all), np.zeros(n_all)])])
            # df/dx2:          [x1, x2, 0, 1]
            X_dx2 = np.hstack([X_locs, np.column_stack([np.zeros(n_all), np.ones(n_all)])])

            X_train = np.vstack([X_func, X_dx1, X_dx2])
            Y_train = np.concatenate([
                y_func + noise_std * np.random.randn(n_all),
                dy_dx1 + noise_std * np.random.randn(n_all),
                dy_dx2 + noise_std * np.random.randn(n_all),
            ]).reshape(-1, 1)

            print(f"  X_train shape: {X_train.shape}")
            print(f"  Y_train shape: {Y_train.shape}")

            base_kernel = SquaredExponential()
            deriv_kernel = deriv_kernel_class(D, base_kernel, D)
            model = gpflow.models.GPR(
                data=(X_train, Y_train),
                kernel=deriv_kernel,
            )

            opt = gpflow.optimizers.Scipy()
            opt.minimize(model.training_loss, model.trainable_variables,
                        options=dict(maxiter=100))

            print(f"  => SUCCESS: trained with all derivatives")
            print(f"  Log likelihood: {model.log_marginal_likelihood().numpy():.4f}")

        except Exception as e:
            print(f"  => FAILED: {type(e).__name__}: {e}")


        # ============================================================
        # TEST 2: Derivatives at SUBSET (10 of 20 points)
        #   Function values at ALL 20 points
        #   Derivatives only at first 10 points
        # ============================================================
        print("\n" + "=" * 60)
        print("TEST 2: Function values at 20 pts, derivatives at 10 pts only")
        print("=" * 60)

        try:
            # Function values at ALL 20 points
            X_func = np.hstack([X_locs, np.zeros((n_all, D))])

            # Derivatives at FIRST 10 points only
            X_dx1_sub = np.hstack([X_locs[:n_deriv],
                                    np.column_stack([np.ones(n_deriv), np.zeros(n_deriv)])])
            X_dx2_sub = np.hstack([X_locs[:n_deriv],
                                    np.column_stack([np.zeros(n_deriv), np.ones(n_deriv)])])

            X_train_sub = np.vstack([X_func, X_dx1_sub, X_dx2_sub])
            Y_train_sub = np.concatenate([
                y_func + noise_std * np.random.randn(n_all),           # 20 function values
                dy_dx1[:n_deriv] + noise_std * np.random.randn(n_deriv),  # 10 dx1 derivs
                dy_dx2[:n_deriv] + noise_std * np.random.randn(n_deriv),  # 10 dx2 derivs
            ]).reshape(-1, 1)

            print(f"  X_train shape: {X_train_sub.shape}  (20 func + 10 dx1 + 10 dx2 = 40)")
            print(f"  Y_train shape: {Y_train_sub.shape}")

            base_kernel2 = SquaredExponential()
            deriv_kernel2 = deriv_kernel_class(D, base_kernel2, D)
            model2 = gpflow.models.GPR(
                data=(X_train_sub, Y_train_sub),
                kernel=deriv_kernel2,
            )

            opt2 = gpflow.optimizers.Scipy()
            opt2.minimize(model2.training_loss, model2.trainable_variables,
                         options=dict(maxiter=100))

            print(f"  => SUCCESS: trained with subset derivatives!")
            print(f"  Log likelihood: {model2.log_marginal_likelihood().numpy():.4f}")
            print(f"  NOTE: GPflow's row-wise encoding naturally supports this!")

        except Exception as e:
            print(f"  => FAILED: {type(e).__name__}: {e}")


        # ============================================================
        # TEST 3: Only dx1 at some points, only dx2 at others
        # ============================================================
        print("\n" + "=" * 60)
        print("TEST 3: Mixed — dx1 at pts 0-9, dx2 at pts 10-19")
        print("=" * 60)

        try:
            X_func = np.hstack([X_locs, np.zeros((n_all, D))])

            # dx1 at first 10 points only
            X_dx1_first = np.hstack([X_locs[:10],
                                      np.column_stack([np.ones(10), np.zeros(10)])])
            # dx2 at last 10 points only
            X_dx2_last = np.hstack([X_locs[10:],
                                     np.column_stack([np.zeros(10), np.ones(10)])])

            X_train_mix = np.vstack([X_func, X_dx1_first, X_dx2_last])
            Y_train_mix = np.concatenate([
                y_func + noise_std * np.random.randn(n_all),
                dy_dx1[:10] + noise_std * np.random.randn(10),
                dy_dx2[10:] + noise_std * np.random.randn(10),
            ]).reshape(-1, 1)

            print(f"  X_train shape: {X_train_mix.shape}  (20 func + 10 dx1 + 10 dx2 = 40)")
            print(f"  Y_train shape: {Y_train_mix.shape}")

            base_kernel3 = SquaredExponential()
            deriv_kernel3 = deriv_kernel_class(D, base_kernel3, D)
            model3 = gpflow.models.GPR(
                data=(X_train_mix, Y_train_mix),
                kernel=deriv_kernel3,
            )

            opt3 = gpflow.optimizers.Scipy()
            opt3.minimize(model3.training_loss, model3.trainable_variables,
                         options=dict(maxiter=100))

            print(f"  => SUCCESS: trained with mixed derivative locations!")
            print(f"  Log likelihood: {model3.log_marginal_likelihood().numpy():.4f}")

        except Exception as e:
            print(f"  => FAILED: {type(e).__name__}: {e}")


    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if deriv_kernel_class is not None:
        print("""
GPflow encodes derivative information ROW-WISE in the input X:
  X shape: (N, 2*D) where last D columns specify derivative order.

This means:
  - Each observation independently specifies its type
  - Function values and derivatives can be at DIFFERENT locations
  - Different derivative components can be at different subsets
  - This is fundamentally more flexible than GPyTorch's approach

However, note:
  - Only first-order derivatives are typically supported
  - No directional derivatives
  - No weighted submodel decomposition
  - No arbitrary-order derivative support
  - The derivative kernel may not be in the core GPflow distribution
""")
    else:
        print("""
GPflow does NOT appear to have a built-in derivative observation kernel
in the core distribution. Derivative support requires:
  - Custom kernel implementation, OR
  - Third-party extensions (e.g., thermoextrap)

This means derivative-enhanced GP is not a first-class feature in GPflow.
""")

else:
    print("\nSkipping tests — GPflow not available.")
    print("Install with: pip install gpflow tensorflow")