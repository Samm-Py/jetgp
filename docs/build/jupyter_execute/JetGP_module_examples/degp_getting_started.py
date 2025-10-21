#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from full_degp.degp import degp
import matplotlib.pyplot as plt

print("Modules imported successfully.")


# In[2]:


X_train = np.array([[0.0], [0.5], [1.0]])
y_func = np.sin(X_train)
y_deriv1 = np.cos(X_train)
y_train = [y_func, y_deriv1]

print("X_train:", X_train)
print("y_train:", y_train)


# In[3]:


der_indices = [[[[1, 1]]]]
print("der_indices:", der_indices)


# In[4]:


model = degp(X_train, y_train, n_order=1, n_bases=1,
             der_indices=der_indices, normalize=True,
             kernel="SE", kernel_type="anisotropic")

print("DEGP model initialized.")


# In[5]:


params =  model.optimize_hyperparameters(
     optimizer='jade',
     pop_size = 100,
     n_generations = 15,
     local_opt_every = None,
     debug = False
     )
print("Optimized hyperparameters:", params)


# In[6]:


y_train_pred = model.predict(X_train, params, calc_cov=False, return_deriv=True)
y_func_pred = y_train_pred[:len(X_train)]
y_deriv_pred = y_train_pred[len(X_train):]

abs_error_func = np.abs(y_func_pred.flatten() - y_func.flatten())
abs_error_deriv = np.abs(y_deriv_pred.flatten() - y_deriv1.flatten())

print("Absolute error (function) at training points:", abs_error_func)
print("Absolute error (derivative) at training points:", abs_error_deriv)


# In[7]:


X_test = np.linspace(0, 1, 100).reshape(-1, 1)
y_test_pred = model.predict(X_test, params, calc_cov=False, return_deriv=True)

y_func_test = y_test_pred[:len(X_test)]
y_deriv_test = y_test_pred[len(X_test):]

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(X_test, np.sin(X_test), 'k--', label='True f(x)')
plt.plot(X_test, y_func_test, 'r-', label='GP prediction')
plt.scatter(X_train, y_func, c='b', s=50, label='Training points')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Function and GP Prediction')
plt.legend()

plt.subplot(1,2,2)
plt.plot(X_test, np.cos(X_test), 'k--', label="True f'(x)")
plt.plot(X_test, y_deriv_test, 'r-', label='GP derivative prediction')
plt.scatter(X_train, y_deriv1, c='b', s=50, label='Training points')
plt.xlabel('x')
plt.ylabel("f'(x)")
plt.title('Derivative and GP Prediction')
plt.legend()

plt.tight_layout()
plt.show()


# In[8]:


X_train = np.array([[0.0], [0.5], [1.0]])
y_func = np.sin(X_train)
y_deriv1 = np.cos(X_train)
y_deriv2 = -np.sin(X_train)
y_train = [y_func, y_deriv1, y_deriv2]

print("X_train:", X_train)
print("y_train:", y_train)


# In[9]:


der_indices = [[[[1, 1]], [[1, 2]]]]
print("der_indices:", der_indices)


# In[10]:


model = degp(X_train, y_train, n_order=2, n_bases=1,
             der_indices=der_indices, normalize=True,
             kernel="SE", kernel_type="anisotropic")

print("DEGP model (2nd order) initialized.")


# In[11]:


params = model.optimize_hyperparameters(
     optimizer='jade',
     pop_size = 100,
     n_generations = 15,
     local_opt_every = None,
     debug = True
     )
print("Optimized hyperparameters:", params)


# In[12]:


y_train_pred = model.predict(X_train, params, calc_cov=False, return_deriv=True)

n_train = len(X_train)
y_func_pred = y_train_pred[:n_train]
y_deriv1_pred = y_train_pred[n_train:2*n_train]
y_deriv2_pred = y_train_pred[2*n_train:]

abs_error_func = np.abs(y_func_pred.flatten() - y_func.flatten())
abs_error_deriv1 = np.abs(y_deriv1_pred.flatten() - y_deriv1.flatten())
abs_error_deriv2 = np.abs(y_deriv2_pred.flatten() - y_deriv2.flatten())

print("Absolute error (function) at training points:", abs_error_func)
print("Absolute error (1st derivative) at training points:", abs_error_deriv1)
print("Absolute error (2nd derivative) at training points:", abs_error_deriv2)


# In[13]:


X_test = np.linspace(0, 1, 100).reshape(-1, 1)
y_test_pred = model.predict(X_test, params, calc_cov=False, return_deriv=True)

n_test = len(X_test)
y_func_test = y_test_pred[:n_test]
y_deriv1_test = y_test_pred[n_test:2*n_test]
y_deriv2_test = y_test_pred[2*n_test:]

plt.figure(figsize=(18,5))

plt.subplot(1,3,1)
plt.plot(X_test, np.sin(X_test), 'k--', label='True f(x)')
plt.plot(X_test, y_func_test, 'r-', label='GP prediction')
plt.scatter(X_train, y_func, c='b', s=50, label='Training points')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Function')
plt.legend()

plt.subplot(1,3,2)
plt.plot(X_test, np.cos(X_test), 'k--', label="True f'(x)")
plt.plot(X_test, y_deriv1_test, 'r-', label='GP derivative prediction')
plt.scatter(X_train, y_deriv1, c='b', s=50, label='Training points')
plt.xlabel('x')
plt.ylabel("f'(x)")
plt.title('1st Derivative')
plt.legend()

plt.subplot(1,3,3)
plt.plot(X_test, -np.sin(X_test), 'k--', label="True f''(x)")
plt.plot(X_test, y_deriv2_test, 'r-', label='GP 2nd derivative prediction')
plt.scatter(X_train, y_deriv2, c='b', s=50, label='Training points')
plt.xlabel('x')
plt.ylabel("f''(x)")
plt.title('2nd Derivative')
plt.legend()

plt.tight_layout()
plt.show()


# In[14]:


import numpy as np
from full_degp.degp import degp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

print("Modules imported successfully for 2D example.")


# In[15]:


X1 = np.array([0.0, 0.5, 1.0])
X2 = np.array([0.0, 0.5, 1.0])
X1_grid, X2_grid = np.meshgrid(X1, X2)
X_train = np.column_stack([X1_grid.flatten(), X2_grid.flatten()])

y_func = np.sin(X_train[:,0]) * np.cos(X_train[:,1])
y_deriv_x = np.cos(X_train[:,0]) * np.cos(X_train[:,1])
y_deriv_y = -np.sin(X_train[:,0]) * np.sin(X_train[:,1])
y_train = [y_func.reshape(-1,1), y_deriv_x.reshape(-1,1), y_deriv_y.reshape(-1,1)]

print("X_train shape:", X_train.shape)
print("y_train shapes:", [v.shape for v in y_train])


# In[16]:


der_indices = [[[[1, 1]], [[2, 1]]]]
print("der_indices:", der_indices)


# In[17]:


model = degp(X_train, y_train, n_order=1, n_bases=2,
             der_indices=der_indices, normalize=True,
             kernel="SE", kernel_type="anisotropic")

print("2D DEGP model initialized.")


# In[18]:


params = model.optimize_hyperparameters(
     optimizer='jade',
     pop_size = 100,
     n_generations = 15,
     local_opt_every = None,
     debug = False
     )
print("Optimized hyperparameters:", params)


# In[19]:


y_train_pred = model.predict(X_train, params, calc_cov=False, return_deriv=True)

n_train = len(X_train)
y_func_pred = y_train_pred[:n_train]
y_deriv_x_pred = y_train_pred[n_train:2*n_train]
y_deriv_y_pred = y_train_pred[2*n_train:]

print("Predictions at training points (function):", y_func_pred)
print("Predictions at training points (deriv x):", y_deriv_x_pred)
print("Predictions at training points (deriv y):", y_deriv_y_pred)


# In[20]:


abs_error_func = np.abs(y_func_pred.flatten() - y_func)
abs_error_dx = np.abs(y_deriv_x_pred.flatten() - y_deriv_x)
abs_error_dy = np.abs(y_deriv_y_pred.flatten() - y_deriv_y)

print("Absolute error (function) at training points:", abs_error_func)
print("Absolute error (deriv x) at training points:", abs_error_dx)
print("Absolute error (deriv y) at training points:", abs_error_dy)


# In[21]:


x1_test = np.linspace(0, 1, 50)
x2_test = np.linspace(0, 1, 50)
X1_test, X2_test = np.meshgrid(x1_test, x2_test)
X_test = np.column_stack([X1_test.flatten(), X2_test.flatten()])

y_test_pred = model.predict(X_test, params, calc_cov=False, return_deriv=True)
n_test = len(X_test)

y_func_test = y_test_pred[:n_test].reshape(50,50)
y_deriv_x_test = y_test_pred[n_test:2*n_test].reshape(50,50)
y_deriv_y_test = y_test_pred[2*n_test:].reshape(50,50)


# In[22]:


import matplotlib.pyplot as plt

# Compute absolute errors for function and first derivatives
abs_error_func = np.abs(y_func_test - (np.sin(X1_test) * np.cos(X2_test)))
abs_error_dx   = np.abs(y_deriv_x_test - (np.cos(X1_test) * np.cos(X2_test)))
abs_error_dy   = np.abs(y_deriv_y_test - (-np.sin(X1_test) * np.sin(X2_test)))

# Setup figure
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Titles and error arrays
titles = [
   r'Absolute Error: $f(x,y)$',
   r'Absolute Error: $\partial f / \partial x$',
   r'Absolute Error: $\partial f / \partial y$'
]
errors = [abs_error_func, abs_error_dx, abs_error_dy]
cmaps  = ['Reds', 'Greens', 'Blues']

# Plot each contour
for ax, err, title, cmap in zip(axs, errors, titles, cmaps):
   cs = ax.contourf(X1_test, X2_test, err, levels=50, cmap=cmap)
   ax.scatter(X_train[:,0], X_train[:,1], c='k', s=40)
   ax.set_title(title, fontsize=14)
   ax.set_xlabel('x')
   ax.set_ylabel('y')
   fig.colorbar(cs, ax=ax)

plt.tight_layout()
plt.show()


# In[23]:


import numpy as np
from full_degp.degp import degp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

print("Modules imported successfully for 2D example with second-order derivatives.")


# In[24]:


X1 = np.array([0.0, 0.5, 1.0])
X2 = np.array([0.0, 0.5, 1.0])
X1_grid, X2_grid = np.meshgrid(X1, X2)
X_train = np.column_stack([X1_grid.flatten(), X2_grid.flatten()])

# True function and derivatives
y_func = np.sin(X_train[:,0]) * np.cos(X_train[:,1])
y_deriv_x = np.cos(X_train[:,0]) * np.cos(X_train[:,1])
y_deriv_y = -np.sin(X_train[:,0]) * np.sin(X_train[:,1])
y_deriv_xx = -np.sin(X_train[:,0]) * np.cos(X_train[:,1])
y_deriv_yy = -np.sin(X_train[:,0]) * np.cos(X_train[:,1])

y_train = [
   y_func.reshape(-1,1),
   y_deriv_x.reshape(-1,1),
   y_deriv_y.reshape(-1,1),
   y_deriv_xx.reshape(-1,1),
   y_deriv_yy.reshape(-1,1)
]

print("X_train shape:", X_train.shape)
print("y_train shapes:", [v.shape for v in y_train])


# In[25]:


der_indices = [
    [ [[1, 1]], [[2, 1]] ],  # first-order derivatives
    [ [[1, 2]], [[2, 2]] ]   # second-order derivatives
]
print("der_indices:", der_indices)


# In[26]:


model = degp(X_train, y_train, n_order=2, n_bases=2,
             der_indices=der_indices, normalize=True,
             kernel="SE", kernel_type="anisotropic")

print("2D DEGP model with second-order derivatives initialized.")


# In[27]:


params =   model.optimize_hyperparameters(
     optimizer='jade',
     pop_size = 100,
     n_generations = 15,
     local_opt_every = None,
     debug = False
     )
print("Optimized hyperparameters:", params)


# In[28]:


y_train_pred = model.predict(X_train, params, calc_cov=False, return_deriv=True)

n_train = len(X_train)

# First-order
y_func_pred    = y_train_pred[:n_train]
y_deriv_x_pred = y_train_pred[n_train:2*n_train]
y_deriv_y_pred = y_train_pred[2*n_train:3*n_train]

# Second-order main derivatives
y_deriv_xx_pred = y_train_pred[3*n_train:4*n_train]
y_deriv_yy_pred = y_train_pred[4*n_train:5*n_train]


# In[29]:


# First-order errors
abs_error_func = np.abs(y_func_pred.flatten() - y_func)
abs_error_dx   = np.abs(y_deriv_x_pred.flatten() - y_deriv_x)
abs_error_dy   = np.abs(y_deriv_y_pred.flatten() - y_deriv_y)

# Second-order main derivative errors
abs_error_dxx  = np.abs(y_deriv_xx_pred.flatten() - y_deriv_xx)
abs_error_dyy  = np.abs(y_deriv_yy_pred.flatten() - y_deriv_yy)

# Print absolute errors
print("Absolute error (function)       :", abs_error_func)
print("Absolute error (derivative x)  :", abs_error_dx)
print("Absolute error (derivative y)  :", abs_error_dy)
print("Absolute error (second x-x)    :", abs_error_dxx)
print("Absolute error (second y-y)    :", abs_error_dyy)


# In[30]:


x1_test = np.linspace(0, 1, 50)
x2_test = np.linspace(0, 1, 50)
X1_test, X2_test = np.meshgrid(x1_test, x2_test)
X_test = np.column_stack([X1_test.flatten(), X2_test.flatten()])

y_test_pred = model.predict(X_test, params, calc_cov=False, return_deriv=True)
n_test = len(X_test)

y_func_test = y_test_pred[:n_test].reshape(50,50)
y_deriv_x_test = y_test_pred[n_test:2*n_test].reshape(50,50)
y_deriv_y_test = y_test_pred[2*n_test:3*n_test].reshape(50,50)
y_deriv_xx_test = y_test_pred[3*n_test:4*n_test].reshape(50,50)
y_deriv_yy_test = y_test_pred[4*n_test:5*n_test].reshape(50,50)


# In[31]:


import matplotlib.pyplot as plt

# Compute absolute errors
abs_error_func = np.abs(y_func_test - (np.sin(X1_test) * np.cos(X2_test)))
abs_error_dx   = np.abs(y_deriv_x_test - (np.cos(X1_test) * np.cos(X2_test)))
abs_error_dy   = np.abs(y_deriv_y_test - (-np.sin(X1_test) * np.sin(X2_test)))
abs_error_xx   = np.abs(y_deriv_xx_test - (-np.sin(X1_test) * np.cos(X2_test)))
abs_error_yy   = np.abs(y_deriv_yy_test - (-np.sin(X1_test) * np.cos(X2_test)))

# Setup figure
fig, axs = plt.subplots(1, 5, figsize=(24, 5))

# Titles and errors
titles = [
   r'Absolute Error: $f(x,y)$',
   r'Absolute Error: $\partial f / \partial x$',
   r'Absolute Error: $\partial f / \partial y$',
   r'Absolute Error: $\partial^2 f / \partial x^2$',
   r'Absolute Error: $\partial^2 f / \partial y^2$'
]

errors = [abs_error_func, abs_error_dx, abs_error_dy, abs_error_xx, abs_error_yy]
cmaps  = ['Reds', 'Greens', 'Blues', 'Greens', 'Blues']

# Plot all
for ax, err, title, cmap in zip(axs, errors, titles, cmaps):
   cs = ax.contourf(X1_test, X2_test, err, levels=50, cmap=cmap)
   ax.scatter(X_train[:,0], X_train[:,1], c='k', s=40)
   ax.set_title(title, fontsize=12)
   ax.set_xlabel('x')
   ax.set_ylabel('y')
   fig.colorbar(cs, ax=ax)

plt.tight_layout()
plt.show()

