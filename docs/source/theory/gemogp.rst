.. _gemogps:

Gradient Enhanced Multi-output Gaussian Processes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The formulation described in the :ref:`mogps` section can be generalized to a gradient enhanced MOGP in a relatively straightforward manner. As before, isotropic training sets are considered so that for :math:`t \in \left\{ 1 , \ldots, T \right\}, \mathbf{X}_T = \mathbf{X}`. Then when including gradient information the training data set can be written as:

.. math::
    :label: GEMOGP_training

    \mathcal{D} = \left\{\left((\mathbf{X}_1, \ldots, \mathbf{X}_T),(\mathbf{y}_1, \ldots, \mathbf{y}_T),(\nabla \mathbf{y}_1 , \ldots, \nabla \mathbf{y}_T)\right) \right\} \in \mathbb{R}^{nT + dnT}

Similar to the derivative GPs section, the observation vector can be augmented to include the partial derivatives of the function at each training point for each output. The observation vector :math:`\mathbf{y}` is expanded into an augmented vector, :math:`\mathbf{y}^{GMO}`. In the general case, the predictions at the test locations :math:`\mathbf{X}_*` are also augmented to include derivatives, forming a vector :math:`\mathbf{y}^{GMO}_*`:

.. math::
    :label: eqn_full_aug_vectors

    \mathbf{y}^{GMO} = \begin{bmatrix} f_1(\mathbf{X}) \\ \vdots \\ f_T(\mathbf{X}) \\ \frac{\partial f_1(\mathbf{X})}{\partial x_1} \\ \vdots \\ \frac{\partial f_1(\mathbf{X})}{\partial x_d} \\ \vdots \\ \frac{\partial f_T(\mathbf{X})}{\partial x_1} \\ \vdots \\ \frac{\partial f_T(\mathbf{X})}{\partial x_d} \end{bmatrix}, \quad
    \mathbf{y}^{GMO}_* = \begin{bmatrix} f_1(\mathbf{X}_*) \\ \vdots \\ f_T(\mathbf{X}_*) \\ \frac{\partial f_1(\mathbf{X}_*)}{\partial x_1} \\ \vdots \\ \frac{\partial f_1(\mathbf{X}_*)}{\partial x_d} \\ \vdots \\ \frac{\partial f_T(\mathbf{X}_*)}{\partial x_1} \\ \vdots \\ \frac{\partial f_T(\mathbf{X}_*)}{\partial x_d} \end{bmatrix}.

Note that since we are assuming isotropic training sets the inputs are assumed to be the same across model outputs.

The joint distribution between the augmented training observations and the augmented test predictions is a multivariate Gaussian:

.. math::
    :label: eqn_full_gek_joint_dist

    \begin{pmatrix}
        \mathbf{y}^{GMO} \\
        \mathbf{y}^{GMO}_*
    \end{pmatrix}
    \sim \mathcal{N}\left(
    \mathbf{0},
    \begin{pmatrix}
        \boldsymbol{\Sigma}^{GMO}_{11} & \boldsymbol{\Sigma}^{GMO}_{12} \\
        \boldsymbol{\Sigma}^{GMO}_{21} & \boldsymbol{\Sigma}^{GMO}_{22}
    \end{pmatrix}
    \right).

The blocks of this covariance matrix are also augmented. Following the Kronecker structure introduced in the :ref:`mogps` section, we employ the separable covariance formulation where the spatial covariance :math:`K(\mathbf{X}, \mathbf{X}') = k^x(\mathbf{X}, \mathbf{X}')` and the task correlation matrix :math:`k^t` are combined. Here, :math:`k^t_{tt'}` denotes the :math:`(t,t')`-th element of :math:`k^t`, representing the correlation between outputs :math:`t` and :math:`t'`. In the Kronecker formulation, the task correlations :math:`k^t_{tt'}` are scalar constants, while derivatives are applied only to the spatial covariance :math:`K(\mathbf{X}, \mathbf{X}')`. The training covariance block, :math:`\boldsymbol{\Sigma}^{GMO}_{11}`, is an :math:`nT(d+1) \times nT(d+1)` matrix:

.. math::
    :label: eqn_full_gek_S11

    \boldsymbol{\Sigma}^{GMO}_{11} =
    \begin{pmatrix}
        k^t_{11}K(\mathbf{X}, \mathbf{X}') & \ldots & k^t_{1T}K(\mathbf{X}, \mathbf{X}') & k^t_{11}\frac{\partial K(\mathbf{X}, \mathbf{X}')}{\partial \mathbf{X}'}  & \ldots & k^t_{1T}\frac{\partial K(\mathbf{X}, \mathbf{X}')}{\partial \mathbf{X}'}  \\
                k^t_{21}K(\mathbf{X}, \mathbf{X}') & \ldots & k^t_{2T}K(\mathbf{X}, \mathbf{X}') & k^t_{21}\frac{\partial K(\mathbf{X}, \mathbf{X}')}{\partial \mathbf{X}'}  & \ldots & k^t_{2T} \frac{\partial K(\mathbf{X}, \mathbf{X}')}{\partial \mathbf{X}'}  \\
                                \vdots & \ddots & \vdots & \vdots & \ddots & \vdots \\
                                                k_{T1}^tK(\mathbf{X}, \mathbf{X}') & \ldots & k_{TT}^tK(\mathbf{X}, \mathbf{X}') & k_{T1}^t\frac{\partial K(\mathbf{X}, \mathbf{X}')}{\partial \mathbf{X}'}  & \ldots & k^t_{TT}\frac{\partial K(\mathbf{X}, \mathbf{X}')}{\partial \mathbf{X}'}  \\
        k_{11}^t\frac{\partial K(\mathbf{X}, \mathbf{X}')}{\partial \mathbf{X}} & \ldots & k_{1T}^t\frac{\partial K(\mathbf{X}, \mathbf{X}')}{\partial \mathbf{X}} & k_{11}^t\frac{\partial^2 K(\mathbf{X}, \mathbf{X}')}{\partial \mathbf{X} \partial \mathbf{X}'} & \ldots &  k_{1T}^t\frac{\partial^2 K(\mathbf{X}, \mathbf{X}')}{\partial \mathbf{X} \partial \mathbf{X}'} \\  k_{21}^t\frac{\partial K(\mathbf{X}, \mathbf{X}')}{\partial \mathbf{X}} & \ldots & k_{2T}^t\frac{\partial K(\mathbf{X}, \mathbf{X}')}{\partial \mathbf{X}} & k_{21}^t\frac{\partial^2 K(\mathbf{X}, \mathbf{X}')}{\partial \mathbf{X} \partial \mathbf{X}'} & \ldots &  k_{2T}^t\frac{\partial^2 K(\mathbf{X}, \mathbf{X}')}{\partial \mathbf{X} \partial \mathbf{X}'} \\ \vdots & \ddots & \vdots & \vdots & \ddots & \vdots \\  k_{T1}^t\frac{\partial K(\mathbf{X}, \mathbf{X}')}{\partial \mathbf{X}} & \ldots & k_{TT}^t\frac{\partial K(\mathbf{X}, \mathbf{X}')}{\partial \mathbf{X}} & k_{T1}^t\frac{\partial^2 K(\mathbf{X}, \mathbf{X}')}{\partial \mathbf{X} \partial \mathbf{X}'} & \ldots &  k_{TT}^t\frac{\partial^2 K(\mathbf{X}, \mathbf{X}')}{\partial \mathbf{X} \partial \mathbf{X}'}
    \end{pmatrix}.

As in the MOGP case, observation noise is accounted for through :math:`\boldsymbol{\Sigma}_M \in \mathbb{R}^{nT(d+1) \times nT(d+1)}`, which adds appropriate noise variances to the diagonal blocks corresponding to function values and derivatives. The training-test covariance block, :math:`\boldsymbol{\Sigma}^{GMO}_{12}` follows a similar format as the training covariance block but now contains the covariances between all training and test observations. The remaining blocks are defined as :math:`\boldsymbol{\Sigma}^{GMO}_{21} = \left(\boldsymbol{\Sigma}^{GMO}_{12}\right)^T`, and :math:`\boldsymbol{\Sigma}^{GMO}_{22}` has the same structure as :math:`\boldsymbol{\Sigma}^{GMO}_{11}` but is evaluated at the test points :math:`\mathbf{X}_*`. The posterior predictive distribution for the augmented test vector :math:`\mathbf{y}^{GMO}_*` is then given by:

.. math::
    :label: eqn_full_gek_posterior

    \begin{split}
        \boldsymbol{\mu}_{*} &= \boldsymbol{\Sigma}^{GMO}_{21} \left(\boldsymbol{\Sigma}^{GMO}_{11} + \boldsymbol{\Sigma}_M\right)^{-1} \mathbf{y}^{GMO} \\
        \boldsymbol{\Sigma}_{*} &=  \boldsymbol{\Sigma}^{GMO}_{22} - \boldsymbol{\Sigma}^{GMO}_{21} \left(\boldsymbol{\Sigma}^{GMO}_{11} + \boldsymbol{\Sigma}_M\right)^{-1} \boldsymbol{\Sigma}^{GMO}_{12}
    \end{split}

The posterior mean :math:`\boldsymbol{\mu}_{*}` now provides predictions for both function values and derivatives, while :math:`\boldsymbol{\Sigma}_{*}` provides their uncertainty.

Similar to the standard GP, the kernel hyperparameters :math:`\boldsymbol{\psi}` are determined by maximizing the log marginal likelihood (MLL) of the augmented observations:

.. math::
    :label: eqn_gek_mll

    \log p(\mathbf{y}^{GMO}|\mathbf{X}, \boldsymbol{\psi}) = -\frac{1}{2} \left(\mathbf{y}^{GMO}\right)^\top \left(\boldsymbol{\Sigma}^{GMO}_{11} + \boldsymbol{\Sigma}_M\right)^{-1} \mathbf{y}^{GMO} - \frac{1}{2}\log|\boldsymbol{\Sigma}^{GMO}_{11} + \boldsymbol{\Sigma}_M| - \frac{nT(d+1)}{2}\log 2\pi.

In the GEMOGP formulation, :math:`\boldsymbol{\psi}` includes the spatial covariance parameters, the Cholesky factors :math:`\{a_i\}` of :math:`k^t`, and the noise variances in :math:`\boldsymbol{\Sigma}_M`, all jointly optimized via MLL.
References
----------

.. bibliography::
   :cited:
   :style: unsrt