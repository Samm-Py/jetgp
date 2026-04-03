"""
Test SMT's derivative-enhanced Kriging capabilities.

Questions to answer:
1. Can SMT handle derivative observations at a SUBSET of training points?
   (i.e., can set_training_derivatives receive a different xt than set_training_values?)
2. Does SMT support higher-order derivatives (second order, Hessian)?
3. What happens if we try directional derivatives?
"""

import numpy as np

try:
    import smt
    from smt.surrogate_models import KRG, GEKPLS
    print(f"SMT version: {smt.__version__}")
    SMT_AVAILABLE = True
except ImportError:
    print("SMT not installed. Install with: pip install smt")
    SMT_AVAILABLE = False


# ============================================================
# Test function: f(x1, x2) = sin(x1) * cos(x2)
# ============================================================
def func(X):
    return (np.sin(X[:, 0]) * np.cos(X[:, 1])).reshape(-1, 1)

def dfunc_dx1(X):
    return (np.cos(X[:, 0]) * np.cos(X[:, 1])).reshape(-1, 1)

def dfunc_dx2(X):
    return (-np.sin(X[:, 0]) * np.sin(X[:, 1])).reshape(-1, 1)

def d2func_dx1dx1(X):
    return (-np.sin(X[:, 0]) * np.cos(X[:, 1])).reshape(-1, 1)

def d2func_dx2dx2(X):
    return (-np.sin(X[:, 0]) * np.cos(X[:, 1])).reshape(-1, 1)


if SMT_AVAILABLE:

    np.random.seed(42)
    n_all = 20
    n_deriv = 10

    X_all = np.random.rand(n_all, 2) * 2  # [0, 2]^2
    y_all = func(X_all)
    dy_dx1_all = dfunc_dx1(X_all)
    dy_dx2_all = dfunc_dx2(X_all)

    X_sub = X_all[:n_deriv]
    dy_dx1_sub = dfunc_dx1(X_sub)
    dy_dx2_sub = dfunc_dx2(X_sub)

    # ============================================================
    # TEST 1: Standard GEK — derivatives at ALL points
    # ============================================================
    print("=" * 60)
    print("TEST 1: GEK with derivatives at ALL 20 points")
    print("=" * 60)

    try:
        sm1 = KRG(theta0=[1e-2] * 2, print_global=False)
        sm1.set_training_values(X_all, y_all)
        sm1.set_training_derivatives(X_all, dy_dx1_all, kx=0)
        sm1.set_training_derivatives(X_all, dy_dx2_all, kx=1)
        sm1.train()

        # Quick prediction test
        X_test = np.random.rand(5, 2) * 2
        y_pred = sm1.predict_values(X_test)
        y_true = func(X_test)
        rmse = np.sqrt(np.mean((y_pred - y_true)**2))

        print(f"  X_train shape: {X_all.shape}")
        print(f"  y_train shape: {y_all.shape}")
        print(f"  => SUCCESS: trained with derivatives at all points")
        print(f"  Test RMSE: {rmse:.6f}")

    except Exception as e:
        print(f"  => FAILED: {type(e).__name__}: {e}")


    # ============================================================
    # TEST 2: Derivatives at SUBSET — same xt shape as training?
    #   Use set_training_derivatives with only 10 of 20 points
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 2: Derivatives at SUBSET (10 of 20 points)")
    print("  set_training_values with 20 pts, set_training_derivatives with 10 pts")
    print("=" * 60)

    try:
        sm2 = KRG(theta0=[1e-2] * 2, print_global=False)
        sm2.set_training_values(X_all, y_all)  # 20 points
        sm2.set_training_derivatives(X_sub, dy_dx1_sub, kx=0)  # 10 points
        sm2.set_training_derivatives(X_sub, dy_dx2_sub, kx=1)  # 10 points
        sm2.train()

        X_test = np.random.rand(5, 2) * 2
        y_pred = sm2.predict_values(X_test)
        y_true = func(X_test)
        rmse = np.sqrt(np.mean((y_pred - y_true)**2))

        print(f"  X_train (values) shape: {X_all.shape}")
        print(f"  X_train (derivs) shape: {X_sub.shape}")
        print(f"  => SUCCESS: trained with derivatives at subset!")
        print(f"  Test RMSE: {rmse:.6f}")

    except Exception as e:
        print(f"  => FAILED: {type(e).__name__}: {e}")


    # ============================================================
    # TEST 3: Different derivative components at different points
    #   dx1 at points 0-9, dx2 at points 10-19
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 3: dx1 at pts 0-9, dx2 at pts 10-19 (mixed locations)")
    print("=" * 60)

    try:
        X_first = X_all[:10]
        X_last = X_all[10:]
        dy_dx1_first = dfunc_dx1(X_first)
        dy_dx2_last = dfunc_dx2(X_last)

        sm3 = KRG(theta0=[1e-2] * 2, print_global=False)
        sm3.set_training_values(X_all, y_all)  # 20 points
        sm3.set_training_derivatives(X_first, dy_dx1_first, kx=0)  # dx1 at 10 pts
        sm3.set_training_derivatives(X_last, dy_dx2_last, kx=1)    # dx2 at 10 other pts
        sm3.train()

        X_test = np.random.rand(5, 2) * 2
        y_pred = sm3.predict_values(X_test)
        y_true = func(X_test)
        rmse = np.sqrt(np.mean((y_pred - y_true)**2))

        print(f"  => SUCCESS: trained with mixed derivative locations!")
        print(f"  Test RMSE: {rmse:.6f}")

    except Exception as e:
        print(f"  => FAILED: {type(e).__name__}: {e}")


    # ============================================================
    # TEST 4: Higher-order derivatives (second order)
    #   Can we pass kx in a way that specifies second-order?
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 4: Second-order derivatives")
    print("  Checking if set_training_derivatives supports higher orders")
    print("=" * 60)

    # First, check the API signature
    import inspect
    sig = inspect.signature(KRG.set_training_derivatives)
    print(f"  set_training_derivatives signature: {sig}")

    # Try with kx as a tuple or higher value to see what happens
    print("\n  4a: Attempting second-order via repeated kx (if supported)...")
    try:
        sm4a = KRG(theta0=[1e-2] * 2, print_global=False)
        sm4a.set_training_values(X_all, y_all)
        sm4a.set_training_derivatives(X_all, dy_dx1_all, kx=0)
        sm4a.set_training_derivatives(X_all, dy_dx2_all, kx=1)
        # Try second-order: d2f/dx1^2
        d2y = d2func_dx1dx1(X_all)
        sm4a.set_training_derivatives(X_all, d2y, kx=0)  # Will this overwrite or add?
        sm4a.train()
        print(f"  => Completed (but likely overwrote first-order dx1)")
    except Exception as e:
        print(f"  => FAILED: {type(e).__name__}: {e}")

    print("\n  4b: Looking for any 'order' parameter...")
    try:
        # Check if there's a way to specify derivative order
        help_text = inspect.getdoc(KRG.set_training_derivatives)
        print(f"  Docstring:\n  {help_text}")
    except Exception as e:
        print(f"  Could not get docstring: {e}")


    # ============================================================
    # TEST 5: GEKPLS — does it have different capabilities?
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 5: GEKPLS — subset derivatives")
    print("=" * 60)

    try:
        sm5 = GEKPLS(theta0=[1e-2] * 2, n_comp=2, print_global=False)
        sm5.set_training_values(X_all, y_all)  # 20 points
        sm5.set_training_derivatives(X_sub, dy_dx1_sub, kx=0)  # 10 points
        sm5.set_training_derivatives(X_sub, dy_dx2_sub, kx=1)  # 10 points
        sm5.train()

        X_test = np.random.rand(5, 2) * 2
        y_pred = sm5.predict_values(X_test)
        y_true = func(X_test)
        rmse = np.sqrt(np.mean((y_pred - y_true)**2))

        print(f"  => SUCCESS: GEKPLS trained with subset derivatives!")
        print(f"  Test RMSE: {rmse:.6f}")

    except Exception as e:
        print(f"  => FAILED: {type(e).__name__}: {e}")


    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
    Tests completed. Check results above for:
    1. Whether SMT supports derivative observations at a subset of points
    2. Whether different derivative components can be at different locations
    3. Whether higher-order (second-order) derivatives are supported
    4. Whether GEKPLS has the same capabilities as KRG for subset derivatives
    """)

else:
    print("\nSkipping tests — SMT not available.")
    print("Install with: pip install smt")