#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from jetgp.wdegp.wdegp import wdegp
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12})

# Define test function: f(x) = sin(10*pi*x)/(2*x) + (x-1)^4
def f_fun(x):
    return np.sin(10*np.pi*x)/(2*x) + (x-1)**4

def f1_fun(x):  # First derivative
    return (10*np.pi*np.cos(10*np.pi*x))/(2*x) - \
           np.sin(10*np.pi*x)/(2*x**2) + 4*(x-1)**3

def f2_fun(x):  # Second derivative
    return -(100*np.pi**2*np.sin(10*np.pi*x))/(2*x) - \
           (20*np.pi*np.cos(10*np.pi*x))/(2*x**2) + \
           np.sin(10*np.pi*x)/(x**3) + 12*(x-1)**2

# Generate training points
X_all = np.linspace(0.5, 2.5, 10).reshape(-1, 1)

# Partition into two submodels (alternating points)
submodel1_indices = [0, 2, 4, 6, 8]
submodel2_indices = [1, 3, 5, 7, 9]

# Reorder for contiguous indexing
X_train = np.vstack([X_all[submodel1_indices],
                     X_all[submodel2_indices]])
y_vals = f_fun(X_train.flatten()).reshape(-1, 1)

print("Training data prepared with 2 submodels (5 points each)")


# In[2]:


# Compute derivatives for each submodel
d1_sm1 = np.array([[f1_fun(X_train[i,0])] for i in range(5)])
d2_sm1 = np.array([[f2_fun(X_train[i,0])] for i in range(5)])
d1_sm2 = np.array([[f1_fun(X_train[i,0])] for i in range(5,10)])
d2_sm2 = np.array([[f2_fun(X_train[i,0])] for i in range(5,10)])

# Package submodel data
submodel_data = [
    [y_vals, d1_sm1, d2_sm1],  # Submodel 1
    [y_vals, d1_sm2, d2_sm2]   # Submodel 2
]

submodel_indices = [[0,1,2,3,4], [5,6,7,8,9]]
derivative_specs = [[[[[1,1]]], [[[1,2]]]], [[[[1,1]]], [[[1,2]]]]]

print("Initializing WDEGP model...")


# In[3]:


# Initialize and optimize
model = wdegp(X_train, submodel_data, n_order=2, n_bases=1,
              index=submodel_indices,
              der_indices=derivative_specs,
              normalize=True, kernel="SE",
              kernel_type="anisotropic")

print("Optimizing hyperparameters...")
params = model.optimize_hyperparameters(optimizer='jade',
                                         pop_size=100,
                                         n_generations=15)

print("Optimization complete!")


# In[4]:


# Predict
X_test = np.linspace(0.5, 2.5, 250).reshape(-1, 1)
y_pred, y_cov, submodel_preds, submodel_covs = model.predict(X_test, params, calc_cov=True, return_submodels = True)

# Predict individual submodels
y_pred_sm1 = submodel_preds[0].flatten()
y_cov_sm1  = submodel_covs[0].flatten()

y_pred_sm2 = submodel_preds[1].flatten()
y_cov_sm2  = submodel_covs[1].flatten()

# Compute true function
y_true = f_fun(X_test.flatten())

# Compute confidence intervals (95%)
std_global = np.sqrt(y_cov)
std_sm1 = np.sqrt(y_cov_sm1)
std_sm2 = np.sqrt(y_cov_sm2)

print(f"Predictions complete for {len(X_test)} test points")

