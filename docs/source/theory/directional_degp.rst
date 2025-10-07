Directional Derivative-Enhanced Gaussian Processes
===================================================


First-order directional derivatives have also been used to train derivative informed gaussian processes :cite:`Yao2024-hp, Padidar2021-fr`. A key feature of this approach is that the directions can be chosen uniquely for each training point, allowing for a more flexible model. The training data is augmented with a small, user-selected number of $q$ directional derivatives at each point.

This is achieved by augmenting the observation vector. The training vector, :math:`y_{dd}`, is formed by concatenating the :math:`n` function values with all :math:`n \times q` directional derivative values. In the general case, the predictions at the test locations are also augmented, forming a vector :math:`y_{dd,*}`:

.. math::

    y_{dd} = 
    \begin{bmatrix} 
    f(X) \\ 
    \frac{\partial f(X)}{\partial V} 
    \end{bmatrix}, \quad
    y_{dd,*} = 
    \begin{bmatrix} 
    f(X_*) \\ 
    \frac{\partial f(X_*)}{\partial V_*} 
    \end{bmatrix}.

Here, :math:`V` and :math:`V_*` schematically represent the sets of point-specific direction vectors for the training and test sets, respectively. The joint distribution over these augmented vectors is a multivariate Gaussian:

.. math::

    \begin{pmatrix}
        y^{DD} \\
        y^{DD}_*
    \end{pmatrix}
    \sim \mathcal{N}\left(
    0,
    \begin{pmatrix}
        \Sigma_{11}^{DD} & \Sigma_{12}^{DD} \\
        \Sigma_{21}^{DD} & \Sigma_{22}^{DD}
    \end{pmatrix}
    \right).

The training covariance block, :math:`\Sigma_{11}^{DD}`, is an :math:`n(1+q) \times n(1+q)` matrix containing the covariances between all function values and point-specific directional derivatives:

.. math::

    \Sigma_{11}^{DD} =
    \begin{pmatrix}
        K(X, X) & \frac{\partial K(X, X)}{\partial V'} \\
        \frac{\partial K(X, X)}{\partial V} & \frac{\partial^2 K(X, X)}{\partial V \partial V'}
    \end{pmatrix}.

Similarly, the training-test covariance block, :math:`\Sigma_{12}^{DD}`, contains the covariances between all training and test observations:

.. math::

    \Sigma_{12}^{DD} =
    \begin{pmatrix}
        K(X, X_*) & \frac{\partial K(X, X_*)}{\partial V_*'} \\
        \frac{\partial K(X, X_*)}{\partial V} & \frac{\partial^2 K(X, X_*)}{\partial V \partial V_*'}
    \end{pmatrix}.

The remaining blocks are defined as :math:`\Sigma_{21}^{DD} = (\Sigma_{12}^{DD})^T`, and :math:`\Sigma_{22}^{DD}` has the same structure as :math:`\Sigma_{11}^{DD}` but is evaluated at the test points :math:`X_*`. The posterior predictive distribution is then given by:

.. math::

    \begin{split}
        \mu_{*} &= \Sigma_{21}^{DD} (\Sigma_{11}^{DD})^{-1} y_{DD} \\
        \Sigma_{*} &=  \Sigma_{22}^{DD} - \Sigma_{21}^{DD} (\Sigma_{11}^{DD})^{-1} \Sigma_{12}^{DD}
    \end{split}

The posterior mean :math:`\mu_*` provides predictions for function values and the selected directional derivatives.

Similar to other GP models, the kernel hyperparameters are found by maximizing the log marginal likelihood of the augmented training data:

.. math::

    \log p(y_{DD}|X, \psi) = -\frac{1}{2} y_{DD}^T (\Sigma_{11}^{DD})^{-1} y_{DD} - \frac{1}{2}\log|\Sigma_{11}^{DD}| - \frac{n(1+q)}{2}\log 2\pi.


.. bibliography::
   :cited:
   :style: unsrt

