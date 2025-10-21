#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pyoti.sparse as oti
from wdegp.wdegp import wdegp
import utils


# In[2]:


def oscillatory_function_with_trend(X, alg=oti):
    x1 = X[:, 0]
    return alg.sin(10 * np.pi * x1) / (2 * x1) + (x1 - 1) ** 4


# In[3]:


n_order = 2
n_bases = 1
lb_x = 0.5
ub_x = 2.5
num_points = 10
test_points = 250
kernel = "SE"
kernel_type = "anisotropic"
normalize = True
n_restart_optimizer = 15
swarm_size = 50
random_seed = 42

np.random.seed(random_seed)


# In[4]:


X_train = np.linspace(lb_x, ub_x, num_points).reshape(-1, 1)
print("Training points:", X_train.ravel())


# In[5]:


submodel_indices = [[i] for i in range(num_points)]
base_derivative_indices = utils.gen_OTI_indices(n_bases, n_order)
derivative_specs = [base_derivative_indices for _ in range(num_points)]

print(f"Number of submodels: {len(submodel_indices)}")
print(f"Derivative types per submodel: {len(base_derivative_indices)}")


# In[6]:


import numpy as np
import sympy as sp

# ------------------------------------------------------------
# Define the example function symbolically
# ------------------------------------------------------------
x = sp.symbols('x')
f_sym = sp.sin(10 * sp.pi * x) / (2 * x) + (x - 1)**4

# Compute first and second derivatives symbolically
f1_sym = sp.diff(f_sym, x)
f2_sym = sp.diff(f_sym, x, 2)

# Convert to callable NumPy functions
f_fun = sp.lambdify(x, f_sym, "numpy")
f1_fun = sp.lambdify(x, f1_sym, "numpy")
f2_fun = sp.lambdify(x, f2_sym, "numpy")

# ------------------------------------------------------------
# Define experiment parameters
# ------------------------------------------------------------
lb_x, ub_x = 0.5, 2.5
num_points = 10
X_train = np.linspace(lb_x, ub_x, num_points).reshape(-1, 1)

# Each submodel corresponds to one training point
submodel_indices = [[i] for i in range(num_points)]

# ------------------------------------------------------------
# Compute function values and derivatives
# ------------------------------------------------------------
y_function_values = f_fun(X_train.flatten()).reshape(-1, 1)

submodel_data = []

for k, idx in enumerate(submodel_indices):
    X_point = X_train[idx].reshape(-1)
    xval = float(X_point[0])

    # Compute derivatives analytically at the submodel point
    d1 = np.array([[f1_fun(xval)]])  # First derivative
    d2 = np.array([[f2_fun(xval)]])  # Second derivative

    # Append [all function values] + [local derivatives]
    submodel_data.append([y_function_values] + [d1, d2])

# ------------------------------------------------------------
# Display results for verification
# ------------------------------------------------------------
print("Training points (X_train):")
print(X_train.flatten())

print("\nFunction values (y_function_values):")
print(y_function_values.flatten())

print("\nExample submodel data for first training point:")
for i, data in enumerate(submodel_data[0]):
    print(f"  Element {i}: {data.flatten()}")


# In[7]:


gp_model = wdegp(
    X_train,
    submodel_data,
    n_order,
    n_bases,
    submodel_indices,
    derivative_specs,
    normalize=normalize,
    kernel=kernel,
    kernel_type=kernel_type
)


# In[8]:


params = gp_model.optimize_hyperparameters(
     optimizer='jade',
     pop_size = 100,
     n_generations = 15,
     local_opt_every = None,
     debug = False
     )
print("Optimized hyperparameters:", params)


# In[9]:


X_test = np.linspace(lb_x, ub_x, test_points).reshape(-1, 1)
y_pred, y_cov, submodel_vals, submodel_cov = gp_model.predict(
    X_test, params, calc_cov=True, return_submodels=True
)
y_true = oscillatory_function_with_trend(X_test, alg=np)
nrmse = utils.nrmse(y_true, y_pred)
print(f"NRMSE: {nrmse:.6f}")


# In[10]:


plt.figure(figsize=(10, 6))
plt.plot(X_test, y_true, 'b-', label='True Function', linewidth=2)
plt.plot(X_test.flatten(), y_pred.flatten(), 'r--', label='GP Prediction', linewidth=2)
plt.fill_between(X_test.ravel(),
                 (y_pred.flatten() - 2*np.sqrt(y_cov)).ravel(),
                 (y_pred.flatten() + 2*np.sqrt(y_cov)).ravel(),
                 color='red', alpha=0.3, label='95% Confidence')
plt.scatter(X_train, y_function_values, color='black', label='Training Points')
plt.title("Weighted Individual Submodel DEGP")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# In[11]:


colors = plt.cm.tab10(np.linspace(0, 1, len(submodel_vals)))
plt.figure(figsize=(10, 6))
for i, color in enumerate(colors):
    plt.plot(X_test.flatten(), submodel_vals[i].flatten(), color=color, alpha=0.6, label=f'Submodel {i+1}')
plt.title("Individual Submodel Predictions")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# In[12]:


import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from wdegp.wdegp import wdegp
import utils


# In[13]:


n_order = 2
n_bases = 1
lb_x = 0.5
ub_x = 2.5
num_points = 10
test_points = 250
kernel = "SE"
kernel_type = "anisotropic"
normalize = True
n_restart_optimizer = 15
swarm_size = 50
random_seed = 42

np.random.seed(random_seed)


# In[14]:


x = sp.symbols('x')
f_sym = sp.sin(10 * sp.pi * x) / (2 * x) + (x - 1)**4

# Compute first and second derivatives symbolically
f1_sym = sp.diff(f_sym, x)
f2_sym = sp.diff(f_sym, x, 2)

# Convert to callable NumPy functions
f_fun = sp.lambdify(x, f_sym, "numpy")
f1_fun = sp.lambdify(x, f1_sym, "numpy")
f2_fun = sp.lambdify(x, f2_sym, "numpy")


# In[15]:


X_train = np.linspace(lb_x, ub_x, num_points).reshape(-1, 1)
print("Training points:", X_train.ravel())


# In[16]:


# Sparse derivative selection: only include derivatives at these points
derivative_indices = [2, 3, 4, 5]

# Single submodel covering all selected derivative points
submodel_indices = [derivative_indices]

# Derivative specs: full derivative set for this submodel
base_derivative_indices = utils.gen_OTI_indices(n_bases, n_order)
derivative_specs = [base_derivative_indices]

print(f"Number of submodels: {len(submodel_indices)}")
print(f"Derivative observation points: {derivative_indices}")
print(f"Derivative types per submodel: {len(base_derivative_indices)}")


# In[17]:


# Compute function values at all training points
y_function_values = f_fun(X_train.flatten()).reshape(-1, 1)

# First and second derivatives at selected derivative points only
d1_all = np.array([[f1_fun(X_train[idx, :])[0] for idx in derivative_indices]]).T
d2_all = np.array([[f2_fun(X_train[idx, :])[0] for idx in derivative_indices]]).T

# Submodel data: [[function values], [first-order derivatives], [second-order derivatives]]
submodel_data = [[y_function_values, d1_all, d2_all]]

# Display for verification
print("\nFunction values (y_function_values):")
print(y_function_values.flatten())

print("\nFirst-order derivatives at selected points:")
print(d1_all.flatten())

print("\nSecond-order derivatives at selected points:")
print(d2_all.flatten())


# In[18]:


gp_model = wdegp(
    X_train,
    submodel_data,
    n_order,
    n_bases,
    submodel_indices,
    derivative_specs,
    normalize=normalize,
    kernel=kernel,
    kernel_type=kernel_type
)


# In[19]:


params = gp_model.optimize_hyperparameters(
    optimizer='jade',
    pop_size = 100,
    n_generations = 15,
    local_opt_every = None,
    debug = False
    )
print("\nOptimized hyperparameters:", params)


# In[20]:


X_test = np.linspace(lb_x, ub_x, test_points).reshape(-1, 1)
y_pred, y_cov, submodel_vals, submodel_cov = gp_model.predict(
    X_test, params, calc_cov=True, return_submodels=True
)
y_true = f_fun(X_test.flatten())
nrmse = np.sqrt(np.mean((y_true - y_pred.flatten())**2)) / (y_true.max() - y_true.min())
print(f"\nNRMSE: {nrmse:.6f}")


# In[21]:


plt.figure(figsize=(10, 6))
plt.plot(X_test, y_true, 'b-', label='True Function', linewidth=2)
plt.plot(X_test.flatten(), y_pred.flatten(), 'r--', label='GP Prediction', linewidth=2)
plt.fill_between(X_test.ravel(),
                 (y_pred.flatten() - 2*np.sqrt(y_cov)).ravel(),
                 (y_pred.flatten() + 2*np.sqrt(y_cov)).ravel(),
                 color='red', alpha=0.3, label='95% Confidence')
plt.scatter(X_train, y_function_values, color='black', label='Training Points')
plt.scatter(X_train[derivative_indices], y_function_values[derivative_indices],
            color='orange', s=100, marker='s', label='Derivative Points', zorder=5)
plt.title("Sparse Analytic Weighted DEGP")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# In[22]:


colors = plt.cm.tab10(np.linspace(0, 1, len(submodel_vals)))
plt.figure(figsize=(10, 6))
for i, color in enumerate(colors):
    plt.plot(X_test.flatten(), submodel_vals[i].flatten(), color=color,
             alpha=0.6, label=f'Submodel {i+1}')
plt.title("Submodel Predictions")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# In[23]:


import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from wdegp.wdegp import wdegp
import utils


# In[24]:


n_order = 2
n_bases = 1
lb_x = 0.5
ub_x = 2.5
num_points = 10
test_points = 250
kernel = "SE"
kernel_type = "anisotropic"
normalize = True
n_restart_optimizer = 15
swarm_size = 50
random_seed = 42

np.random.seed(random_seed)


# In[25]:


x = sp.symbols('x')
f_sym = sp.sin(10 * sp.pi * x) / (2 * x) + (x - 1)**4

# Compute first and second derivatives symbolically
f1_sym = sp.diff(f_sym, x)
f2_sym = sp.diff(f_sym, x, 2)

# Convert to callable NumPy functions
f_fun = sp.lambdify(x, f_sym, "numpy")
f1_fun = sp.lambdify(x, f1_sym, "numpy")
f2_fun = sp.lambdify(x, f2_sym, "numpy")


# In[26]:


# Generate all potential training points
X_all = np.linspace(lb_x, ub_x, num_points).reshape(-1, 1)

print("All available training points:")
for i, x in enumerate(X_all.ravel()):
    print(f"  Index {i}: x = {x:.4f}")


# In[27]:


# Select non-contiguous training points for derivative observations
original_indices = [0, 2, 4, 6, 8]

print(f"\nOriginal training point indices where we want derivatives: {original_indices}")
print("\nSelected training points for derivatives:")
for i, orig_idx in enumerate(original_indices):
    print(f"  Original index {orig_idx}: x = {X_all[orig_idx, 0]:.4f}")


# In[28]:


# Reorder X_all so that selected points come first with sequential indices
# This allows us to use submodel_indices = [[0, 1, 2, 3, 4]]

# Create reordered array: selected points first, then unused points
X_train = np.vstack([X_all[original_indices],
                     X_all[[i for i in range(len(X_all)) if i not in original_indices]]])

print("\nReordered training array (X_train):")
print("  SELECTED points (with derivatives, indices 0-4):")
for i in range(len(original_indices)):
    orig_idx = original_indices[i]
    print(f"    X_train[{i}] = {X_train[i, 0]:.4f} (was X_all[{orig_idx}])")

print("  UNUSED points (function values only, indices 5-9):")
unused_original_indices = [i for i in range(len(X_all)) if i not in original_indices]
for i in range(len(original_indices), len(X_train)):
    orig_idx = unused_original_indices[i - len(original_indices)]
    print(f"    X_train[{i}] = {X_train[i, 0]:.4f} (was X_all[{orig_idx}])")

print(f"\n*** CRITICAL: X_train has ALL {len(X_train)} points, reordered so selected points have indices 0-4 ***")


# In[29]:


# CORRECT: Use sequential indices for the reordered training data
# Even though we selected original points [0,2,4,6,8], we now use [0,1,2,3,4]
# Create ONE submodel that uses derivatives at the first 5 points
submodel_indices = [[0, 1, 2, 3, 4]]  # Single submodel with derivatives at indices 0-4

# Derivative specs: full derivative set for this submodel
base_derivative_indices = utils.gen_OTI_indices(n_bases, n_order)
derivative_specs = [base_derivative_indices]  # One spec for one submodel

print(f"Number of submodels: {len(submodel_indices)}")
print(f"Submodel indices (sequential): {submodel_indices}")
print(f"Derivative types per submodel: {len(base_derivative_indices)}")

print("\n*** These indices reference positions in X_train (0-4 are selected points) ***")


# In[30]:


# Compute function values at ALL reordered training points
y_function_values = f_fun(X_train.flatten()).reshape(-1, 1)

print("Function values at ALL reordered training points (X_train):")
for i, (x, y) in enumerate(zip(X_train.ravel(), y_function_values.ravel())):
    status = "SELECTED (with derivatives)" if i < len(original_indices) else "unused (function only)"
    print(f"  X_train[{i}] = {x:.4f}: f(x) = {y:.4f}  ({status})")

# Prepare submodel data with derivatives at the selected points only
# For this single submodel, we need derivatives at the first 5 points (indices 0-4)
derivative_indices = [0, 1, 2, 3, 4]  # First 5 indices in X_train

# Compute derivatives at selected points
d1_all = np.array([[f1_fun(X_train[idx, 0])] for idx in derivative_indices])
d2_all = np.array([[f2_fun(X_train[idx, 0])] for idx in derivative_indices])

print("\nFirst derivatives at SELECTED points (indices 0-4 in X_train):")
for i, d1_val in enumerate(d1_all):
    print(f"  d1(X_train[{i}]) = {d1_val[0]:.4f}")

print("\nSecond derivatives at SELECTED points (indices 0-4 in X_train):")
for i, d2_val in enumerate(d2_all):
    print(f"  d2(X_train[{i}]) = {d2_val[0]:.4f}")

# Single submodel data: [function values at ALL points, derivatives at selected points]
# y_function_values contains ALL function values from X_train (10 points, reordered)
# d1_all and d2_all contain derivatives only at first 5 points (indices 0-4)
submodel_data = [[y_function_values, d1_all, d2_all]]

print("\nCRITICAL: Submodel data structure")
print(f"  {len(submodel_data)} submodel (single submodel)")
print(f"  This submodel has {len(submodel_data[0])} elements:")
print(f"    - Element 0: Function values at ALL {len(y_function_values)} points in X_train")
print(f"    - Element 1: First derivatives at {len(d1_all)} points (indices 0-4)")
print(f"    - Element 2: Second derivatives at {len(d2_all)} points (indices 0-4)")


# In[31]:


gp_model = wdegp(
    X_train,
    submodel_data,
    n_order,
    n_bases,
    submodel_indices,
    derivative_specs,
    normalize=normalize,
    kernel=kernel,
    kernel_type=kernel_type
)


# In[32]:


params = gp_model.optimize_hyperparameters(
    optimizer='jade',
    pop_size = 100,
    n_generations = 15,
    local_opt_every = None,
    debug = False
    )
print("Optimized hyperparameters:", params)


# In[33]:


X_test = np.linspace(lb_x, ub_x, test_points).reshape(-1, 1)
y_pred, y_cov, submodel_vals, submodel_cov = gp_model.predict(
    X_test, params, calc_cov=True, return_submodels=True
)
y_true = f_fun(X_test.flatten())
nrmse = np.sqrt(np.mean((y_true - y_pred.flatten())**2)) / (y_true.max() - y_true.min())
print(f"NRMSE: {nrmse:.6f}")


# In[34]:


plt.figure(figsize=(10, 6))
plt.plot(X_test, y_true, 'b-', label='True Function', linewidth=2)
plt.plot(X_test.flatten(), y_pred.flatten(), 'r--', label='GP Prediction', linewidth=2)
plt.fill_between(X_test.ravel(),
                 (y_pred.flatten() - 2*np.sqrt(y_cov)).ravel(),
                 (y_pred.flatten() + 2*np.sqrt(y_cov)).ravel(),
                 color='red', alpha=0.3, label='95% Confidence')

# Show selected training points (used in model with derivatives)
plt.scatter(X_train[:len(original_indices)],
            f_fun(X_train[:len(original_indices), 0].flatten()).reshape(-1, 1),
            color='black', s=100, label='Points with derivatives', zorder=5)

# Show unused training points (function values only, no derivatives)
plt.scatter(X_train[len(original_indices):],
            f_fun(X_train[len(original_indices):, 0].flatten()).reshape(-1, 1),
            color='gray', s=100, marker='x',
            label='Points without derivatives', zorder=5)

plt.title("Weighted DEGP with Selective Derivative Observations")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# In[35]:


print("=" * 70)
print("INDEX MAPPING SUMMARY")
print("=" * 70)
print("\nOriginal Data (X_all) --> Reordered Data (X_train):")
print("-" * 70)

# Show mapping for selected points
print("SELECTED points (with derivatives):")
for i in range(len(original_indices)):
    orig_idx = original_indices[i]
    print(f"  X_all[{orig_idx}] = {X_all[orig_idx,0]:.4f}  -->  X_train[{i}] = {X_train[i,0]:.4f}")

# Show mapping for unused points
print("\nUNUSED points (function values only):")
unused_original_indices = [i for i in range(len(X_all)) if i not in original_indices]
for j, orig_idx in enumerate(unused_original_indices):
    new_idx = len(original_indices) + j
    print(f"  X_all[{orig_idx}] = {X_all[orig_idx,0]:.4f}  -->  X_train[{new_idx}] = {X_train[new_idx,0]:.4f}")

print("\nSubmodel Indices:")
print("-" * 70)
print(f"  submodel_indices = {submodel_indices}")
print(f"  These reference the FIRST 5 positions in reordered X_train")
print(f"  Which correspond to original indices: {original_indices}")
print("=" * 70)


# In[36]:


plt.figure(figsize=(10, 6))
plt.plot(X_test, y_true, 'b-', label='True Function', linewidth=2, alpha=0.5)
plt.plot(X_test.flatten(), submodel_vals[0].flatten(), 'g-',
         label='Submodel Prediction', linewidth=2, alpha=0.7)
plt.scatter(X_train[:len(original_indices)],
            f_fun(X_train[:len(original_indices), 0].flatten()).reshape(-1, 1),
            color='black', s=100, label='Points with derivatives', zorder=5)
plt.scatter(X_train[len(original_indices):],
            f_fun(X_train[len(original_indices):, 0].flatten()).reshape(-1, 1),
            color='gray', s=100, marker='x', label='Points without derivatives', zorder=5)
plt.title("Single Submodel Prediction")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# In[37]:


import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from wdegp.wdegp import wdegp
import utils


# In[38]:


n_order = 2
n_bases = 1
lb_x = 0.5
ub_x = 2.5
num_points = 10
test_points = 250
kernel = "SE"
kernel_type = "anisotropic"
normalize = True
n_restart_optimizer = 15
swarm_size = 50
random_seed = 42

np.random.seed(random_seed)


# In[39]:


x = sp.symbols('x')
f_sym = sp.sin(10 * sp.pi * x) / (2 * x) + (x - 1)**4

# Compute first and second derivatives symbolically
f1_sym = sp.diff(f_sym, x)
f2_sym = sp.diff(f_sym, x, 2)

# Convert to callable NumPy functions
f_fun = sp.lambdify(x, f_sym, "numpy")
f1_fun = sp.lambdify(x, f1_sym, "numpy")
f2_fun = sp.lambdify(x, f2_sym, "numpy")


# In[40]:


# Generate all training points
X_all = np.linspace(lb_x, ub_x, num_points).reshape(-1, 1)

print("All available training points:")
for i, x in enumerate(X_all.ravel()):
    print(f"  Index {i}: x = {x:.4f}")


# In[41]:


# Submodel 1: uses points [0,2,4,6,8]
# Submodel 2: uses points [1,3,5,7,9]
submodel1_original_indices = [0, 2, 4, 6, 8]
submodel2_original_indices = [1, 3, 5, 7, 9]

print(f"Submodel 1 will use derivatives at original indices: {submodel1_original_indices}")
print(f"Submodel 2 will use derivatives at original indices: {submodel2_original_indices}")


# In[42]:


# Reorder so submodel 1 points come first (indices 0-4),
# then submodel 2 points (indices 5-9)
X_train = np.vstack([X_all[submodel1_original_indices],
                     X_all[submodel2_original_indices]])

print("\nReordered training array (X_train):")
print("  Submodel 1 points (indices 0-4):")
for i in range(len(submodel1_original_indices)):
    orig_idx = submodel1_original_indices[i]
    print(f"    X_train[{i}] = {X_train[i, 0]:.4f} (was X_all[{orig_idx}])")

print("  Submodel 2 points (indices 5-9):")
for i in range(len(submodel1_original_indices), len(X_train)):
    orig_idx = submodel2_original_indices[i - len(submodel1_original_indices)]
    print(f"    X_train[{i}] = {X_train[i, 0]:.4f} (was X_all[{orig_idx}])")

print(f"\n*** X_train has ALL {len(X_train)} points, reordered for contiguous submodel indices ***")


# In[43]:


# Submodel 1: references indices 0-4 in X_train
# Submodel 2: references indices 5-9 in X_train
submodel_indices = [[0, 1, 2, 3, 4],    # Submodel 1
                    [5, 6, 7, 8, 9]]    # Submodel 2

# Derivative specs: full derivative set for each submodel
base_derivative_indices = utils.gen_OTI_indices(n_bases, n_order)
derivative_specs = [base_derivative_indices, base_derivative_indices]

print(f"Number of submodels: {len(submodel_indices)}")
print(f"Submodel 1 indices: {submodel_indices[0]}")
print(f"Submodel 2 indices: {submodel_indices[1]}")
print(f"Derivative types per submodel: {len(base_derivative_indices)}")


# In[44]:


# Function values at ALL reordered training points
y_function_values = f_fun(X_train.flatten()).reshape(-1, 1)

print("Function values at ALL reordered training points:")
for i, (x, y) in enumerate(zip(X_train.ravel(), y_function_values.ravel())):
    submodel = "Submodel 1" if i < 5 else "Submodel 2"
    print(f"  X_train[{i}] = {x:.4f}: f(x) = {y:.4f}  ({submodel})")

# Submodel 1: derivatives at indices 0-4
print("\nSubmodel 1 - Derivatives at indices 0-4:")
submodel1_derivative_indices = [0, 1, 2, 3, 4]
d1_submodel1 = np.array([[f1_fun(X_train[idx, 0])] for idx in submodel1_derivative_indices])
d2_submodel1 = np.array([[f2_fun(X_train[idx, 0])] for idx in submodel1_derivative_indices])

for i in range(len(d1_submodel1)):
    print(f"  d1(X_train[{i}]) = {d1_submodel1[i, 0]:.4f}")

# Submodel 2: derivatives at indices 5-9
print("\nSubmodel 2 - Derivatives at indices 5-9:")
submodel2_derivative_indices = [5, 6, 7, 8, 9]
d1_submodel2 = np.array([[f1_fun(X_train[idx, 0])] for idx in submodel2_derivative_indices])
d2_submodel2 = np.array([[f2_fun(X_train[idx, 0])] for idx in submodel2_derivative_indices])

for i, idx in enumerate(submodel2_derivative_indices):
    print(f"  d1(X_train[{idx}]) = {d1_submodel2[i, 0]:.4f}")

# Prepare submodel data
# Each submodel gets: [function values at ALL points, derivatives at its points]
submodel_data = [
    [y_function_values, d1_submodel1, d2_submodel1],  # Submodel 1
    [y_function_values, d1_submodel2, d2_submodel2]   # Submodel 2
]

print("\nCRITICAL: Submodel data structure")
print(f"  {len(submodel_data)} submodels")
print(f"  Each submodel has {len(submodel_data[0])} elements:")
print(f"    - Element 0: Function values at ALL {len(y_function_values)} points")
print(f"    - Element 1: First derivatives at submodel's 5 points")
print(f"    - Element 2: Second derivatives at submodel's 5 points")


# In[45]:


gp_model = wdegp(
    X_train,
    submodel_data,
    n_order,
    n_bases,
    submodel_indices,
    derivative_specs,
    normalize=normalize,
    kernel=kernel,
    kernel_type=kernel_type
)


# In[46]:


params = gp_model.optimize_hyperparameters(
    optimizer='jade',
    pop_size = 100,
    n_generations = 15,
    local_opt_every = None,
    debug = False
    )
print("Optimized hyperparameters:", params)


# In[47]:


X_test = np.linspace(lb_x, ub_x, test_points).reshape(-1, 1)
y_pred, y_cov, submodel_vals, submodel_cov = gp_model.predict(
    X_test, params, calc_cov=True, return_submodels=True
)
y_true = f_fun(X_test.flatten())
nrmse = np.sqrt(np.mean((y_true - y_pred.flatten())**2)) / (y_true.max() - y_true.min())
print(f"NRMSE: {nrmse:.6f}")


# In[48]:


plt.figure(figsize=(10, 6))
plt.plot(X_test, y_true, 'b-', label='True Function', linewidth=2)
plt.plot(X_test.flatten(), y_pred.flatten(), 'r--', label='GP Prediction', linewidth=2)
plt.fill_between(X_test.ravel(),
                 (y_pred.flatten() - 2*np.sqrt(y_cov)).ravel(),
                 (y_pred.flatten() + 2*np.sqrt(y_cov)).ravel(),
                 color='red', alpha=0.3, label='95% Confidence')

# Show submodel 1 points
plt.scatter(X_train[:5],
            f_fun(X_train[:5, 0].flatten()).reshape(-1, 1),
            color='green', s=100, marker='o', label='Submodel 1 points', zorder=5)

# Show submodel 2 points
plt.scatter(X_train[5:],
            f_fun(X_train[5:, 0].flatten()).reshape(-1, 1),
            color='purple', s=100, marker='s',
            label='Submodel 2 points', zorder=5)

plt.title("Weighted DEGP with Two Submodels")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# In[49]:


print("=" * 70)
print("INDEX MAPPING SUMMARY")
print("=" * 70)
print("\nOriginal Data (X_all) --> Reordered Data (X_train):")
print("-" * 70)

print("Submodel 1 points:")
for i in range(len(submodel1_original_indices)):
    orig_idx = submodel1_original_indices[i]
    print(f"  X_all[{orig_idx}] = {X_all[orig_idx,0]:.4f}  -->  X_train[{i}] = {X_train[i,0]:.4f}")

print("\nSubmodel 2 points:")
for i, orig_idx in enumerate(submodel2_original_indices):
    new_idx = len(submodel1_original_indices) + i
    print(f"  X_all[{orig_idx}] = {X_all[orig_idx,0]:.4f}  -->  X_train[{new_idx}] = {X_train[new_idx,0]:.4f}")

print("\nSubmodel Indices:")
print("-" * 70)
print(f"  Submodel 1: {submodel_indices[0]} (original indices: {submodel1_original_indices})")
print(f"  Submodel 2: {submodel_indices[1]} (original indices: {submodel2_original_indices})")
print("=" * 70)


# In[50]:


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Original X_all with submodel assignments
axes[0].scatter(X_all[submodel1_original_indices],
                np.zeros(len(submodel1_original_indices)),
                color='green', s=100, alpha=0.8, marker='o', label='Submodel 1')
axes[0].scatter(X_all[submodel2_original_indices],
                np.zeros(len(submodel2_original_indices)),
                color='purple', s=100, alpha=0.8, marker='s', label='Submodel 2')
axes[0].set_xlim(lb_x, ub_x)
axes[0].set_ylim(-0.5, 0.5)
axes[0].set_xlabel('x')
axes[0].set_title('Original X_all: Submodel Assignments')
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[0].set_yticks([])

# Right: Reordered X_train
colors = ['green']*5 + ['purple']*5
markers = ['o']*5 + ['s']*5
for i, (x, color, marker) in enumerate(zip(X_train.ravel(), colors, markers)):
    axes[1].scatter(x, i, c=color, s=100, alpha=0.8, marker=marker)
    submodel = "SM1" if i < 5 else "SM2"
    axes[1].text(x, i, f'  idx={i} ({submodel})', va='center', fontsize=8)
axes[1].set_xlim(lb_x, ub_x)
axes[1].set_xlabel('x')
axes[1].set_ylabel('X_train index')
axes[1].set_title('Reordered X_train: Contiguous Submodel Indices')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()


# In[51]:


plt.figure(figsize=(10, 6))
plt.plot(X_test, y_true, 'b-', label='True Function', linewidth=2, alpha=0.3)
plt.plot(X_test.flatten(), submodel_vals[0].flatten(), 'g-',
         label='Submodel 1 Prediction', linewidth=2, alpha=0.7)
plt.plot(X_test.flatten(), submodel_vals[1].flatten(), 'purple', linestyle='--',
         label='Submodel 2 Prediction', linewidth=2, alpha=0.7)
plt.scatter(X_train[:5],
            f_fun(X_train[:5, 0].flatten()).reshape(-1, 1),
            color='green', s=100, marker='o', label='Submodel 1 points', zorder=5)
plt.scatter(X_train[5:],
            f_fun(X_train[5:, 0].flatten()).reshape(-1, 1),
            color='purple', s=100, marker='s', label='Submodel 2 points', zorder=5)
plt.title("Individual Submodel Predictions")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# In[52]:


import numpy as np
import pyoti.sparse as oti
import itertools
from wdegp.wdegp import wdegp
from matplotlib import pyplot as plt
import utils

plt.rcParams.update({'font.size': 12})


# In[53]:


# Random seed for reproducibility
random_seed = 0
np.random.seed(random_seed)

# GP configuration
n_order = 3  # Maximum derivative order (for center submodel)
n_bases = 2
lb_x, ub_x = -1.0, 1.0
lb_y, ub_y = -1.0, 1.0
points_per_axis = 4
kernel = "RQ"
kernel_type = "isotropic"
normalize = True
n_restart_optimizer = 15
swarm_size = 250
test_points_per_axis = 35

# Submodel point groups (using original grid indices)
submodel_point_groups = [
    [0, 3, 12, 15],                 # Submodel 1: Corners (no derivatives)
    [1, 2, 4, 8, 7, 11, 13, 14],    # Submodel 2: Edges (1st order)
    [5, 6, 9, 10]                   # Submodel 3: Center (3rd order)
]


# In[54]:


def six_hump_camel_function(X, alg=np):
    """
    Six-hump camel function - a challenging 2D benchmark.

    Features:
    - Multiple local minima
    - Two global minima at approximately (0.0898, -0.7126) and (-0.0898, 0.7126)
    - Varying curvature across the domain

    Parameters
    ----------
    X : array_like
        Input array of shape (n, 2) with columns [x1, x2]
    alg : module
        Algorithm module (np or oti) for polymorphic evaluation

    Returns
    -------
    array_like
        Function values
    """
    x1 = X[:, 0]
    x2 = X[:, 1]
    return ((4 - 2.1 * x1**2 + (x1**4)/3.0) * x1**2 +
            x1*x2 + (-4 + 4*x2**2) * x2**2)


# In[55]:


def generate_training_points():
    """Generate a 4×4 grid of training points."""
    x_vals = np.linspace(lb_x, ub_x, points_per_axis)
    y_vals = np.linspace(lb_y, ub_y, points_per_axis)
    return np.array(list(itertools.product(x_vals, y_vals)))

X_train_initial = generate_training_points()

print(f"Training points shape: {X_train_initial.shape}")
print(f"Total training points: {len(X_train_initial)}")
print(f"\nGrid layout (indices):")
grid_indices = np.arange(16).reshape(4, 4)
print(grid_indices)


# In[56]:


def visualize_submodel_groups(X_train, submodel_groups):
    """Visualize which points belong to which submodel."""
    fig, ax = plt.subplots(figsize=(8, 8))

    colors = ['red', 'blue', 'green']
    labels = ['Corners (no deriv)', 'Edges (1st order)', 'Center (3rd order)']
    markers = ['s', 'o', '^']

    for i, (group, color, label, marker) in enumerate(zip(submodel_groups, colors, labels, markers)):
        points = X_train[group]
        ax.scatter(points[:, 0], points[:, 1], c=color, s=200,
                  marker=marker, label=label, edgecolors='black', linewidths=2, zorder=5)

        # Add index labels
        for idx in group:
            ax.text(X_train[idx, 0], X_train[idx, 1] + 0.08, str(idx),
                   fontsize=9, ha='center', fontweight='bold')

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title('Submodel Point Groupings (Original Indices)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()

visualize_submodel_groups(X_train_initial, submodel_point_groups)


# In[57]:


def reorder_training_data(X_train_initial, submodel_point_groups):
    """
    Reorder training data so each submodel's points are contiguous.

    Original indices are remapped to sequential positions:
    - Submodel 1 points → indices 0-3
    - Submodel 2 points → indices 4-11
    - Submodel 3 points → indices 12-15

    Returns
    -------
    X_train_reordered : array
        Reordered training points
    sequential_indices : list of lists
        Contiguous index ranges for each submodel
    """
    # Flatten all submodel indices into a single reordering
    arbitrary_flat = list(itertools.chain.from_iterable(submodel_point_groups))

    # Find any unused points (none in this example, but code is general)
    all_indices = set(range(X_train_initial.shape[0]))
    used_indices = set(arbitrary_flat)
    unused_indices = sorted(list(all_indices - used_indices))

    # Create reordering map
    reorder_map = arbitrary_flat + unused_indices
    X_train_reordered = X_train_initial[reorder_map]

    # Generate sequential indices for each submodel
    sequential_indices = []
    current_pos = 0
    for group in submodel_point_groups:
        group_len = len(group)
        sequential_indices.append(list(range(current_pos, current_pos + group_len)))
        current_pos += group_len

    return X_train_reordered, sequential_indices, reorder_map

X_train_reordered, sequential_indices, reorder_map = reorder_training_data(
    X_train_initial, submodel_point_groups
)

print("Reordered training points shape:", X_train_reordered.shape)
print("\nSequential submodel indices:")
print(f"  Submodel 1 (corners): {sequential_indices[0]}")
print(f"  Submodel 2 (edges):   {sequential_indices[1]}")
print(f"  Submodel 3 (center):  {sequential_indices[2]}")
print("\nReordering map:")
print(f"  Original → Reordered: {reorder_map}")


# In[58]:


def prepare_submodel_data(X_train, submodel_indices):
    """
    Prepare submodel data with heterogeneous derivative orders.

    Submodel 1: Function values only (no derivatives)
    Submodel 2: Function values + 1st order derivatives
    Submodel 3: Function values + up to 3rd order derivatives

    Parameters
    ----------
    X_train : array
        Reordered training points
    submodel_indices : list of lists
        Contiguous indices for each submodel

    Returns
    -------
    submodel_data : list of lists
        Data for each submodel
    derivative_specs : list of lists
        Derivative specifications for each submodel
    """
    # Define derivative structures for each submodel
    derivative_specs = [
        [],                              # Submodel 1: no derivatives
        utils.gen_OTI_indices(n_bases, 1),  # Submodel 2: 1st order only
        utils.gen_OTI_indices(n_bases, 3)   # Submodel 3: up to 3rd order
    ]

    print("Derivative specifications:")
    for i, spec in enumerate(derivative_specs):
        if len(spec) == 0:
            print(f"  Submodel {i+1}: No derivatives")
        else:
            print(f"  Submodel {i+1}: {len(spec)} derivative types")
            print(f"    Examples: {spec[:3]}")  # Show first few

    # Compute function values at ALL training points (shared by all submodels)
    y_function_values = six_hump_camel_function(X_train, alg=np).reshape(-1, 1)

    submodel_data = []

    for k, point_indices in enumerate(submodel_indices):
        print(f"\nProcessing Submodel {k+1} (indices {point_indices[0]}-{point_indices[-1]})...")

        # Extract submodel points and apply hypercomplex perturbations
        X_sub_oti = oti.array(X_train[point_indices])

        # Add hypercomplex perturbations for each dimension
        for i in range(n_bases):
            for j in range(X_sub_oti.shape[0]):
                X_sub_oti[j, i] += oti.e(i + 1, order=n_order)

        # Evaluate function with hypercomplex AD
        y_with_derivatives = oti.array([
            six_hump_camel_function(x, alg=oti)
            for x in X_sub_oti
        ])

        # Build data list: function values (all points) + derivatives (submodel points)
        current_submodel_data = [y_function_values]
        current_derivative_spec = derivative_specs[k]

        # Extract derivatives according to this submodel's specification
        for i in range(len(current_derivative_spec)):
            for j in range(len(current_derivative_spec[i])):
                deriv = y_with_derivatives.get_deriv(
                    current_derivative_spec[i][j]
                ).reshape(-1, 1)
                current_submodel_data.append(deriv)

        print(f"  Function values: shape {current_submodel_data[0].shape}")
        print(f"  Derivative arrays: {len(current_submodel_data) - 1}")

        submodel_data.append(current_submodel_data)

    return submodel_data, derivative_specs

submodel_data, derivative_specs = prepare_submodel_data(
    X_train_reordered, sequential_indices
)

print(f"\n{'='*60}")
print("Summary of submodel data:")
for i, data in enumerate(submodel_data):
    print(f"  Submodel {i+1}: {len(data)} arrays")
    print(f"    - Array 0: Function values at ALL {len(data[0])} points")
    print(f"    - Arrays 1-{len(data)-1}: Derivatives at {len(sequential_indices[i])} points")


# In[59]:


def build_and_optimize_gp(X_train, submodel_data, submodel_indices, derivative_specs):
    """
    Build and optimize the weighted DEGP model.

    Parameters
    ----------
    X_train : array
        Reordered training points (all 16 points)
    submodel_data : list of lists
        Data for each submodel
    submodel_indices : list of lists
        Contiguous indices for each submodel
    derivative_specs : list of lists
        Derivative specifications for each submodel

    Returns
    -------
    gp_model : wdegp
        Trained GP model
    params : dict
        Optimized hyperparameters
    """
    print("Building WDEGP model...")
    gp_model = wdegp(
        X_train,
        submodel_data,
        n_order,
        n_bases,
        submodel_indices,
        derivative_specs,
        normalize=normalize,
        kernel=kernel,
        kernel_type=kernel_type
    )

    print("Optimizing hyperparameters...")
    params = gp_model.optimize_hyperparameters(
    optimizer='jade',
    pop_size = 100,
    n_generations = 15,
    local_opt_every = None,
    debug = False
    )

    print("Optimization complete.")
    return gp_model, params

gp_model, params = build_and_optimize_gp(
    X_train_reordered, submodel_data, sequential_indices, derivative_specs
)


# In[60]:


# Create dense test grid
x_lin = np.linspace(lb_x, ub_x, test_points_per_axis)
y_lin = np.linspace(lb_y, ub_y, test_points_per_axis)
X1_grid, X2_grid = np.meshgrid(x_lin, y_lin)
X_test = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])

print(f"Test grid: {test_points_per_axis}×{test_points_per_axis} = {len(X_test)} points")

# Predict on test grid
y_pred, submodel_vals = gp_model.predict(
    X_test, params, calc_cov=False, return_submodels=True
)

# Compute ground truth and error
y_true = six_hump_camel_function(X_test, alg=np)
nrmse = utils.nrmse(y_true, y_pred)
abs_error = np.abs(y_true - y_pred)

print(f"\nModel Performance:")
print(f"  NRMSE: {nrmse:.6f}")
print(f"  Max absolute error: {abs_error.max():.6f}")
print(f"  Mean absolute error: {abs_error.mean():.6f}")


# In[61]:


fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)

# True function
c1 = axes[0].contourf(X1_grid, X2_grid,
                      y_true.reshape(test_points_per_axis, test_points_per_axis),
                      levels=50, cmap="viridis")
fig.colorbar(c1, ax=axes[0])
axes[0].set_title("True Function (Six-Hump Camel)")
axes[0].scatter(X_train_initial[:, 0], X_train_initial[:, 1],
                c="red", edgecolor="k", s=50, label="Training Points", zorder=5)
axes[0].legend()

# GP prediction
c2 = axes[1].contourf(X1_grid, X2_grid,
                      y_pred.reshape(test_points_per_axis, test_points_per_axis),
                      levels=50, cmap="viridis")
fig.colorbar(c2, ax=axes[1])
axes[1].set_title("WDEGP Prediction")
axes[1].scatter(X_train_initial[:, 0], X_train_initial[:, 1],
                c="red", edgecolor="k", s=50, zorder=5)

# Absolute error
c3 = axes[2].contourf(X1_grid, X2_grid,
                      abs_error.reshape(test_points_per_axis, test_points_per_axis),
                      levels=50, cmap="magma")
fig.colorbar(c3, ax=axes[2])
axes[2].set_title("Absolute Error")
axes[2].scatter(X_train_initial[:, 0], X_train_initial[:, 1],
                c="red", edgecolor="k", s=50, zorder=5)

for ax in axes:
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_aspect('equal')

plt.tight_layout()
plt.show()


# In[62]:


fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)

submodel_titles = [
    "Submodel 1: Corners (No Derivatives)",
    "Submodel 2: Edges (1st Order)",
    "Submodel 3: Center (3rd Order)"
]

for i, (ax, title) in enumerate(zip(axes, submodel_titles)):
    c = ax.contourf(X1_grid, X2_grid,
                   submodel_vals[i].reshape(test_points_per_axis, test_points_per_axis),
                   levels=50, cmap="viridis")
    fig.colorbar(c, ax=ax)
    ax.set_title(title)

    # Highlight this submodel's training points
    submodel_points = X_train_reordered[sequential_indices[i]]
    ax.scatter(submodel_points[:, 0], submodel_points[:, 1],
              c='red', s=100, edgecolor='k', linewidths=2, zorder=5)

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_aspect('equal')

plt.tight_layout()
plt.show()

