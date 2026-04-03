import sys, warnings
sys.path.insert(0, '.')
import numpy as np
from jetgp.full_ddegp.ddegp import ddegp

def f(X): return X[:,0]**2 + X[:,1]**2
def grad(X): return np.column_stack([2*X[:,0], 2*X[:,1]])
def dir_deriv(X, ray): return (grad(X) @ ray).flatten()

angles = [np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
RAYS = np.array([[np.cos(a) for a in angles],[np.sin(a) for a in angles]])

np.random.seed(7)
X_train = np.random.uniform(-1,1,(15,2))
y_func = f(X_train).reshape(-1,1)
y_dirs = [dir_deriv(X_train, RAYS[:,i]).reshape(-1,1) for i in range(3)]
y_train = [y_func] + y_dirs
der_indices = [[[[1,1]],[[2,1]],[[3,1]]]]

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    model = ddegp(X_train, y_train, n_order=1, der_indices=der_indices,
                  rays=RAYS, normalize=True, kernel='SE', kernel_type='isotropic')

np.random.seed(99)
X_test = np.random.uniform(-1,1,(50,2))
dir4_true = dir_deriv(X_test, RAYS[:,3])

for restart in [3, 5, 10, 15, 20]:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        params = model.optimize_hyperparameters(
            optimizer='powell', n_restart_optimizer=restart, debug=False)

    pred = model.predict(X_test, params, calc_cov=False,
                         return_deriv=True, derivs_to_predict=[[[4,1]]])
    dir4_pred = pred[1,:].flatten()
    corr = float(np.corrcoef(dir4_pred, dir4_true)[0,1])
    print(f"  n_restart={restart:2d}: corr={corr:.4f}")
