import warnings, numpy as np
from jetgp.full_degp.degp import degp

np.random.seed(42)
n_train_per_dim = 5
x1_tr = np.linspace(0.3, 2*np.pi - 0.3, n_train_per_dim)
x2_tr = np.linspace(0.3, 2*np.pi - 0.3, n_train_per_dim)
g1, g2 = np.meshgrid(x1_tr, x2_tr)
X_train = np.column_stack([g1.ravel(), g2.ravel()])
all_idx = list(range(len(X_train)))

def f(X): return np.sin(X[:,0]) * np.cos(X[:,1])
def df1(X): return np.cos(X[:,0]) * np.cos(X[:,1])
def df2(X): return -np.sin(X[:,0]) * np.sin(X[:,1])
def d2f11(X): return -np.sin(X[:,0]) * np.cos(X[:,1])
def d2f12(X): return -np.cos(X[:,0]) * np.sin(X[:,1])
def d2f22(X): return -np.sin(X[:,0]) * np.cos(X[:,1])

y = [f(X_train).reshape(-1,1), df1(X_train).reshape(-1,1), df2(X_train).reshape(-1,1),
     d2f11(X_train).reshape(-1,1), d2f12(X_train).reshape(-1,1), d2f22(X_train).reshape(-1,1)]

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    model = degp(X_train, y, n_order=2, n_bases=2,
        der_indices=[[[[1,1]],[[2,1]]],[[[1,2]],[[1,1],[2,1]],[[2,2]]]],
        derivative_locations=[all_idx]*5,
        normalize=True, kernel='SE', kernel_type='anisotropic')

params = model.optimize_hyperparameters(optimizer='pso', pop_size=100, n_generations=15, local_opt_every=15,
debug=False)

x1_te = np.linspace(0, 2*np.pi, 30)
x2_te = np.linspace(0, 2*np.pi, 30)
g1t, g2t = np.meshgrid(x1_te, x2_te)
X_test = np.column_stack([g1t.ravel(), g2t.ravel()])

mean, var = model.predict(X_test, params, calc_cov=True, return_deriv=True,
    derivs_to_predict=[[[1,1]],[[2,1]],[[1,2]],[[1,1],[2,1]], [[2,2]]])

labels = ['f', 'df/dx1', 'df/dx2', 'd2f/dx1^2','d2f/dx1dx2', 'd2f/dx2^2']
trues = [f, df1, df2, d2f11, d2f12, d2f22]
for i, (label, fn) in enumerate(zip(labels, trues)):
    true_vals = fn(X_test) 
    std_vals = np.sqrt(np.abs(var[i,:]))
    rmse = np.sqrt(np.mean((mean[i] - true_vals)**2))
    print(f'{label:>15}  RMSE={rmse:.4e}  std: min={std_vals.min():.4e}  max={std_vals.max():.4e} mean={std_vals.mean():.4e}')