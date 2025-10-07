Derivative-Enhanced Gaussian Processes
=======================================

When derivative information is available, it can be incorporated to improve model accuracy in high-dimensional or highly nonlinear problems :cite:`GEK, GHEK`. The most common approach involves including first-order gradient information, which has been shown to significantly improve the accuracy of GP models, particularly for functions with a high number of dimensions (:math:`d \ge 8`) :cite:`Ulaganathan2016-xq`.

This is achieved by augmenting the observation vector to include the partial derivatives of the function at each training point. The observation vector :math:`y` is expanded into an augmented vector, :math:`y^G`. In the general case, the predictions at the test locations :math:`X_*` are also augmented to include derivatives, forming a vector :math:`y^G_*`:

.. math::

    y^{G} = 
    \begin{bmatrix} 
    f(X) \\ 
    \frac{\partial f(X)}{\partial x_1} \\ 
    \vdots \\ 
    \frac{\partial f(X)}{\partial x_d} 
    \end{bmatrix}, \quad
    y^{G}_* = 
    \begin{bmatrix} 
    f(X_*) \\ 
    \frac{\partial f(X_*)}{\partial x_1} \\ 
    \vdots \\ 
    \frac{\partial f(X_*)}{\partial x_d} 
    \end{bmatrix}.

The joint distribution between the augmented training observations and the augmented test predictions is a multivariate Gaussian:

.. math::

    \begin{pmatrix}
        y^{G} \\
        y^{G}_*
    \end{pmatrix}
    \sim \mathcal{N}\left(
    0,
    \begin{pmatrix}
        \Sigma^{G}_{11} & \Sigma^{G}_{12} \\
        \Sigma^{G}_{21} & \Sigma^{G}_{22}
    \end{pmatrix}
    \right).

The blocks of this covariance matrix are also augmented. The training covariance block, :math:`\Sigma^G_{11}`, is an :math:`n(d+1) \times n(d+1)` matrix:

.. math::

    \Sigma^{G}_{11} =
    \begin{pmatrix}
        K(X, X) & \frac{\partial K(X, X)}{\partial X'} \\
        \frac{\partial K(X, X)}{\partial X} & \frac{\partial^2 K(X, X)}{\partial X \partial X'}
    \end{pmatrix}.

The training-test covariance block, :math:`\Sigma^G_{12}`, contains the covariances between all training and test observations:

.. math::

    \Sigma^{G}_{12} =
    \begin{pmatrix}
        K(X, X_*) & \frac{\partial K(X, X_*)}{\partial X_*'} \\
        \frac{\partial K(X, X_*)}{\partial X} & \frac{\partial^2 K(X, X_*)}{\partial X \partial X_*'}
    \end{pmatrix}.

The remaining blocks are defined as :math:`\Sigma^G_{21} = (\Sigma^G_{12})^T`, and :math:`\Sigma^G_{22}` has the same structure as :math:`\Sigma^G_{11}` but is evaluated at the test points :math:`X_*`. The posterior predictive distribution for the augmented test vector :math:`y^G_*` is then given by:

.. math::

    \begin{split}
        \mu_{*} &= \Sigma^{G}_{21} (\Sigma^{G}_{11})^{-1} y \\
        \Sigma_{*} &=  \Sigma^{G}_{22} - \Sigma^{G}_{21} (\Sigma^{G}_{11})^{-1} \Sigma^{G}_{12}
    \end{split}

The posterior mean :math:`\mu_*` now provides predictions for both function values and derivatives, while :math:`\Sigma_*` provides their uncertainty.

Similar to the standard GP, the kernel hyperparameters :math:`\psi` are determined by maximizing the log marginal likelihood (MLL) of the augmented observations:

.. math::

    \log p(y^{G}|X, \psi) = -\frac{1}{2} (y^{G})^T (\Sigma^{G}_{11})^{-1} y^{G} - \frac{1}{2}\log|\Sigma^{G}_{11}| - \frac{n(d+1)}{2}\log 2\pi.

Evaluating this function during optimization is computationally demanding. The primary bottleneck is computing the inverse of :math:`\Sigma^G_{11}` and the log-determinant :math:`\log|\Sigma^G_{11}|`, typically via Cholesky decomposition. The cost of this decomposition is approximately :math:`O(M^3)`, where :math:`M` is the matrix dimension. For the gradient-enhanced case, :math:`M = n(d+1)`, resulting in a cost of :math:`O((n(d+1))^3)`. This cubic scaling with respect to both :math:`n` and :math:`d` makes hyperparameter optimization prohibitively expensive for problems with many data points or high dimensionality :cite:`FORRESTER200950, HeYouwei2023Aegk`.

The framework can be further extended to include second-order derivative information (Hessians), which is particularly useful for capturing behavior in highly nonlinear problems :cite:`GHEK`. This is achieved by further augmenting the observation vectors to include all function values, gradients, and unique Hessian components. The augmented training vector, now denoted :math:`y^H`, concatenates the function values, gradients, and the :math:`n \times d(d+1)/2` unique components of the Hessian matrix from each of the :math:`n` training points. For a general model that also predicts these quantities, the test vector :math:`y^H_*` is augmented similarly.

The joint distribution over these fully augmented vectors remains Gaussian, but the covariance matrix blocks are expanded further to include up to the fourth-order derivatives of the kernel function. The augmented training-training covariance block, :math:`\Sigma^H_{11}`, is a :math:`3 \times 3` block matrix with the following structure:

.. math::

    \Sigma^{H}_{11} =
    \begin{pmatrix}
        K & \frac{\partial K}{\partial X'} & \frac{\partial^2 K}{\partial (X')^2} \\
        \frac{\partial K}{\partial X} & \frac{\partial^2 K}{\partial X \partial X'} & \frac{\partial^3 K}{\partial X \partial (X')^2} \\
        \frac{\partial^2 K}{\partial X^2} & \frac{\partial^3 K}{\partial X^2 \partial X'} & \frac{\partial^4 K}{\partial X^2 \partial (X')^2}
    \end{pmatrix}, 

where :math:`K = K(X, X)`. The training-test block :math:`\Sigma^H_{12}` and test-test block :math:`\Sigma^H_{22}` are constructed analogously. The posterior predictive equations retain their standard form but now operate on these much larger matrices.

.. bibliography::
   :cited:
   :style: unsrt
