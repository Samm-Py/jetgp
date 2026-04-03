Derivative-Enhanced Gaussian Processes
======================================

When derivative information is available, it can be incorporated to improve model accuracy in high-dimensional or highly nonlinear problems :cite:`GEK,GHEK`. The most common approach involves including first-order gradient information, which has been shown to significantly improve the accuracy of GP models, particularly for functions with a high number of dimensions :math:`(d \geq 8)` :cite:`Ulaganathan2016-xq`.

This is achieved by augmenting the observation vector to include the partial derivatives of the function at each training point. The observation vector :math:`\mathbf{y}` is expanded into an augmented vector, :math:`\mathbf{y}^{G}`. In the general case, the predictions at the test locations :math:`\mathbf{X}_*` are also augmented to include derivatives, forming a vector :math:`\mathbf{y}^{G}_*`:

.. math::
    :label: eqn_full_aug_vectors

    \mathbf{y}^{G} = \begin{bmatrix} f(\mathbf{X}) \\ \frac{\partial f(\mathbf{X})}{\partial x_1} \\ \vdots \\ \frac{\partial f(\mathbf{X})}{\partial x_d} \end{bmatrix}, \quad
    \mathbf{y}^{G}_* = \begin{bmatrix} f(\mathbf{X}_*) \\ \frac{\partial f(\mathbf{X}_*)}{\partial x_1} \\ \vdots \\ \frac{\partial f(\mathbf{X}_*)}{\partial x_d} \end{bmatrix}

The joint distribution between the augmented training observations and the augmented test predictions is a multivariate Gaussian:

.. math::
    :label: eqn_full_gek_joint_dist

    \begin{pmatrix}
        \mathbf{y}^{G} \\
        \mathbf{y}^{G}_*
    \end{pmatrix}
    \sim \mathcal{N}\left(
    \mathbf{0},
    \begin{pmatrix}
        \boldsymbol{\Sigma}^{G}_{11} & \boldsymbol{\Sigma}^{G}_{12} \\
        \boldsymbol{\Sigma}^{G}_{21} & \boldsymbol{\Sigma}^{G}_{22}
    \end{pmatrix}
    \right)

The blocks of this covariance matrix are also augmented. The training covariance block, :math:`\boldsymbol{\Sigma}^{G}_{11}`, is an :math:`n(d + 1) \times n(d + 1)` matrix:

.. math::
    :label: eqn_full_gek_S11

    \boldsymbol{\Sigma}^{G}_{11} =
    \begin{pmatrix}
        K(\mathbf{X}, \mathbf{X}') & \frac{\partial K(\mathbf{X}, \mathbf{X}')}{\partial \mathbf{X}'} \\
        \frac{\partial K(\mathbf{X}, \mathbf{X}')}{\partial \mathbf{X}} & \frac{\partial^2 K(\mathbf{X}, \mathbf{X}')}{\partial \mathbf{X} \partial \mathbf{X}'}
    \end{pmatrix}

The training-test covariance block, :math:`\boldsymbol{\Sigma}^{G}_{12}`, contains the covariances between all training and test observations:

.. math::
    :label: eqn_full_gek_S12

    \boldsymbol{\Sigma}^{G}_{12} =
    \begin{pmatrix}
        K(\mathbf{X}, \mathbf{X}_*) & \frac{\partial K(\mathbf{X}, \mathbf{X}_*)}{\partial \mathbf{X}_*'} \\
        \frac{\partial K(\mathbf{X}, \mathbf{X}_*)}{\partial \mathbf{X}} & \frac{\partial^2 K(\mathbf{X}, \mathbf{X}_*)}{\partial \mathbf{X} \partial \mathbf{X}_*'}
    \end{pmatrix}

The remaining blocks are defined as :math:`\boldsymbol{\Sigma}^{G}_{21} = \left(\boldsymbol{\Sigma}^{G}_{12}\right)^T`, and :math:`\boldsymbol{\Sigma}^{G}_{22}` has the same structure as :math:`\boldsymbol{\Sigma}^{G}_{11}` but is evaluated at the test points :math:`\mathbf{X}_*`. The posterior predictive distribution for the augmented test vector :math:`\mathbf{y}^{G}_*` is then given by:

.. math::
    :label: eqn_full_gek_posterior

    \boldsymbol{\mu}_{*} &= \boldsymbol{\Sigma}^{G}_{21} \left(\boldsymbol{\Sigma}^{G}_{11}\right)^{-1} \mathbf{y}^G \\
    \boldsymbol{\Sigma}_{*} &=  \boldsymbol{\Sigma}^{G}_{22} - \boldsymbol{\Sigma}^{G}_{21} \left(\boldsymbol{\Sigma}^{G}_{11}\right)^{-1} \boldsymbol{\Sigma}^{G}_{12}

The posterior mean :math:`\boldsymbol{\mu}_{*}` now provides predictions for both function values and derivatives, while :math:`\boldsymbol{\Sigma}_{*}` provides their uncertainty.

Similar to the standard GP, the kernel hyperparameters :math:`\boldsymbol{\psi}` are determined by maximizing the log marginal likelihood (MLL) of the augmented observations:

.. math::
    :label: eqn_gek_mll

    \log p(\mathbf{y}^{G}|\mathbf{X}, \boldsymbol{\psi}) = -\frac{1}{2} \left(\mathbf{y}^{G}\right)^\top \left(\boldsymbol{\Sigma}^{G}_{11}\right)^{-1} \mathbf{y}^{G} - \frac{1}{2}\log|\boldsymbol{\Sigma}^{G}_{11}| - \frac{n(d+1)}{2}\log 2\pi

Evaluating this function during optimization is computationally demanding. The primary bottleneck is computing the inverse of :math:`\boldsymbol{\Sigma}^{G}_{11}` and the log-determinant :math:`\log|\boldsymbol{\Sigma}^{G}_{11}|`, typically via Cholesky decomposition. The cost of this decomposition is approximately :math:`\mathcal{O}(M^3)`, where :math:`M` is the matrix dimension. For the gradient-enhanced case, :math:`M=n(d + 1)`, resulting in a cost of :math:`\mathcal{O}\left((n(d + 1))^3\right)`. This cubic scaling with respect to both *n* and *d* makes hyperparameter optimization prohibitively expensive for problems with many data points or high dimensionality :cite:`FORRESTER200950,HeYouwei2023Aegk`.

Hessian-Enhanced Gaussian Processes
------------------------------------

The framework can be further extended to include second-order derivative information (Hessians), which is particularly useful for capturing behavior in highly nonlinear problems :cite:`GHEK`. This is achieved by further augmenting the observation vectors to include all function values, gradients, and unique Hessian components.

The augmented training vector, now denoted :math:`\mathbf{y}^{H}`, concatenates the function values, gradients, and the :math:`n\times d(d+1)/2` unique components of the Hessian matrix from each of the :math:`n` training points. For a general model that also predicts these quantities, the test vector :math:`\mathbf{y}^{H}_*` is augmented similarly.

The joint distribution over these fully augmented vectors remains Gaussian, but the covariance matrix blocks are expanded further to include up to the fourth-order derivatives of the kernel function. The augmented training-training covariance block, :math:`\boldsymbol{\Sigma}^{H}_{11}`, is a 3 × 3 block matrix with the following structure:

.. math::
    :label: eqn_hegp_cov_matrix

    \boldsymbol{\Sigma}^{H}_{11} =
    \begin{pmatrix}
        K & \frac{\partial K}{\partial \mathbf{X}'} & \frac{\partial^2 K}{\partial (\mathbf{X}')^2} \\
        \frac{\partial K}{\partial \mathbf{X}} & \frac{\partial^2 K}{\partial \mathbf{X} \partial \mathbf{X}'} & \frac{\partial^3 K}{\partial \mathbf{X} \partial (\mathbf{X}')^2} \\
        \frac{\partial^2 K}{\partial \mathbf{X}^2} & \frac{\partial^3 K}{\partial \mathbf{X}^2 \partial \mathbf{X}'} & \frac{\partial^4 K}{\partial \mathbf{X}^2 \partial (\mathbf{X}')^2}
    \end{pmatrix}

where *K* = *K*\ (:math:`\mathbf{X}`, :math:`\mathbf{X}`). The training-test block :math:`\boldsymbol{\Sigma}^{H}_{12}` and test-test block :math:`\boldsymbol{\Sigma}^{H}_{22}` are constructed analogously. The posterior predictive equations retain their standard form but now operate on these much larger matrices.

Note on Simplified Test Predictions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In many applications, only the function values :math:`f(\mathbf{X}_*)` are required at the test points, not their derivatives. In this case, the augmented training-test covariance blocks :math:`\Sigma_{12}^{(\cdot)}` simplify by retaining only the columns corresponding to the test function values, while still leveraging derivative information from the training points. Additionally, the test-test covariance block :math:`\Sigma_{22}^{(\cdot)}` reduces to the standard form :math:`K(\mathbf{X}_*, \mathbf{X}_*')`, since no derivatives are predicted at the test locations.

For a gradient-enhanced GP (GEK), the full training-test block is

.. math::

    \Sigma_{12}^{G} =
    \begin{pmatrix}
        K(\mathbf{X}, \mathbf{X}_*) & \frac{\partial K(\mathbf{X}, \mathbf{X}_*)}{\partial \mathbf{X}_*'} \\
        \frac{\partial K(\mathbf{X}, \mathbf{X}_*)}{\partial \mathbf{X}} & \frac{\partial^2 K(\mathbf{X}, \mathbf{X}_*)}{\partial \mathbf{X} \partial \mathbf{X}_*'}
    \end{pmatrix}

When only :math:`f(\mathbf{X}_*)` is needed, this reduces to

.. math::

    \Sigma_{12}^{G} \;\longrightarrow\;
    \begin{pmatrix}
        K(\mathbf{X}, \mathbf{X}_*) \\
        \frac{\partial K(\mathbf{X}, \mathbf{X}_*)}{\partial \mathbf{X}}
    \end{pmatrix}, \quad \text{and} \quad
    \Sigma_{22}^{G} \;\longrightarrow\; K(\mathbf{X}_*, \mathbf{X}_*')

For a Hessian-enhanced GP (HEGP), the full block is

.. math::

    \Sigma_{12}^{H} =
    \begin{pmatrix}
        K & \frac{\partial K}{\partial \mathbf{X}_*'} & \frac{\partial^2 K}{\partial (\mathbf{X}_*')^2} \\
        \frac{\partial K}{\partial \mathbf{X}} & \frac{\partial^2 K}{\partial \mathbf{X} \partial \mathbf{X}_*'} & \frac{\partial^3 K}{\partial \mathbf{X} \partial (\mathbf{X}_*')^2} \\
        \frac{\partial^2 K}{\partial \mathbf{X}^2} & \frac{\partial^3 K}{\partial \mathbf{X}^2 \partial \mathbf{X}_*'} & \frac{\partial^4 K}{\partial \mathbf{X}^2 \partial (\mathbf{X}_*')^2}
    \end{pmatrix}

which reduces to

.. math::

    \Sigma_{12}^{H} \;\longrightarrow\;
    \begin{pmatrix}
        K \\
        \frac{\partial K}{\partial \mathbf{X}} \\
        \frac{\partial^2 K}{\partial \mathbf{X}^2}
    \end{pmatrix}, \quad \text{and} \quad
    \Sigma_{22}^{H} \;\longrightarrow\; K(\mathbf{X}_*, \mathbf{X}_*')

This simplification significantly reduces the computational cost of making predictions while still benefiting from derivative information during training.

.. bibliography::
   :cited:
   :style: unsrt