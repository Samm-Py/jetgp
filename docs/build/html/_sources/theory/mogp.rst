Multi-output Gaussian Processes
================================

A multi-output Gaussian process (MOGP) extends the standard single-output GP to jointly approximate multiple outputs :math:`\{ \mathbf{y}_t \}_{t=1}^T`, explicitly modeling their correlations to improve predictive accuracy compared to independent modeling :cite:`LIU2018102`. For convenience isotropic training sets are considered so that for :math:`t \in \{ 1 , \ldots, T \}`, :math:`\mathbf{X}_T = \mathbf{X}`.

Formally,

.. math::

    \mathbf{f}(\mathbf{x}) = \big( f_1(\mathbf{x}), \ldots, f_T(\mathbf{x}) \big)^T 
    \sim \mathcal{GP}\big(0, \mathcal{K}_M(\mathbf{x}, \mathbf{x}')\big)

where the multi-output covariance is defined as

.. math::

    \mathcal{K}_M(\mathbf{x}, \mathbf{x}') =
    \begin{bmatrix}
    k_{11}(\mathbf{x}, \mathbf{x}') & \cdots & k_{1T}(\mathbf{x}, \mathbf{x}') \\
    \vdots & \ddots & \vdots \\
    k_{T1}(\mathbf{x}, \mathbf{x}') & \cdots & k_{TT}(\mathbf{x}, \mathbf{x}')
    \end{bmatrix}

Where in particular, :math:`k_{tt'}(\mathbf{x}, \mathbf{x}')` corresponds to the correlation between outputs :math:`f_{t}(\mathbf{x})` and :math:`f_{t'}(\mathbf{x})`. Then, it is assumed that:

.. math::
    :label: MOGP_input

    y_t(\mathbf{x}) = f_t(\mathbf{x}) + \epsilon_t

where :math:`\epsilon_t \sim \mathcal{N}(0, \sigma_{n, t}^2)` is an additive independent and identically distributed (i.i.d) Gaussian noise of the :math:`t^{\text{th}}` output. Consideration of this noise not only improves robustness of the covariance matrix but also information transfer across the outputs :cite:`MOGP_MS`.

Joint Distribution
------------------

To predict outputs at a new set of test locations, :math:`\mathbf{X}_*`, we model the joint distribution given by Equation :eq:`eqn_joint_MO` over the training and test outputs where the training outputs are described by :math:`[f_1(X), \ldots, f_T(\mathbf{X})]^T` and the test outputs by :math:`[f_1(X_*), \ldots, f_T(\mathbf{X}_*)]^T`.

The joint distribution between the augmented training observations and the augmented test predictions is a multivariate Gaussian:

.. math::
    :label: eqn_joint_MO

    \begin{pmatrix}
        \mathbf{y}^{MO} \\
        \mathbf{y}^{MO}_*
    \end{pmatrix}
    \sim \mathcal{N}\left(
    \mathbf{0},
    \begin{pmatrix}
        \boldsymbol{\Sigma}_{11}^{MO} & \boldsymbol{\Sigma}_{12}^{MO} \\
        \boldsymbol{\Sigma}^{MO}_{21} & \boldsymbol{\Sigma}_{22}^{MO}
    \end{pmatrix}
    \right)

Here, the training covariance block, :math:`\boldsymbol{\Sigma}^{MO}_{11}`, is an :math:`nT \times nT` matrix:

.. math::
    :label: eqn_full_MO_S11

    \boldsymbol{\Sigma}^{MO}_{11} =
    \begin{pmatrix}
    k_{11}(\mathbf{X}, \mathbf{X}') & \cdots & k_{1T}(\mathbf{X}, \mathbf{X}') \\
    \vdots & \ddots & \vdots \\
    k_{T1}(\mathbf{X}, \mathbf{X}') & \cdots & k_{TT}(\mathbf{X}, \mathbf{X}')
    \end{pmatrix}

The training-test covariance block, :math:`\boldsymbol{\Sigma}_{12}^{MO}`, contains the covariances between all training and test observations:

.. math::
    :label: eqn_full_MO_S12

    \boldsymbol{\Sigma}^{MO}_{12} =
    \begin{pmatrix}
    k_{11}(\mathbf{X}, \mathbf{X}_*) & \cdots & k_{1T}(\mathbf{X}, \mathbf{X}_*) \\
    \vdots & \ddots & \vdots \\
    k_{T1}(\mathbf{X}, \mathbf{X}_*) & \cdots & k_{TT}(\mathbf{X}, \mathbf{X}_*)
    \end{pmatrix}

The remaining blocks are defined as :math:`\boldsymbol{\Sigma}_{21}^{MO} = \boldsymbol{\Sigma}_{12}^{MO}`, and :math:`\boldsymbol{\Sigma}_{22}^{MO}` has the same structure as :math:`\boldsymbol{\Sigma}_{11}^{MO}` but is evaluated at the test points :math:`\mathbf{X}_*`. The posterior predictive distribution for the augmented test vector :math:`\mathbf{y}_*^{MO}` is then given by:

.. math::
    :label: eqn_full_MO_posterior

    \boldsymbol{\mu}_{*} &= \boldsymbol{\Sigma}_{21}^{MO} \left(\boldsymbol{\Sigma}^{MO}_{11}\right)^{-1} \mathbf{y}^{MO} \\
    \boldsymbol{\Sigma}_{*} &=  \boldsymbol{\Sigma}_{22}^{MO} - \boldsymbol{\Sigma}_{21}^{MO} \left(\boldsymbol{\Sigma}_{11}^{MO} \right)^{-1} \boldsymbol{\Sigma}_{12}^{MO}

Each element :math:`k_{tt'}(\mathbf{X}, \mathbf{X}')` specifies the covariance between outputs :math:`f_t(\mathbf{X})` and :math:`f_{t'}(\mathbf{X}')`, and the model can be trained by maximum likelihood estimation of kernel hyperparameters.

Connection to Gradient-Enhanced GPs
------------------------------------

A gradient-enhanced GP (GEK) or derivative-enhanced GP (DEGP) can be interpreted as a *structured MOGP*, where the outputs correspond to the function value and its partial derivatives:

.. math::

    \mathbf{f}(\mathbf{x}) = \Big(f(\mathbf{x}), \; \tfrac{\partial f}{\partial x_1}(\mathbf{x}), \; \ldots, \; \tfrac{\partial f}{\partial x_d}(\mathbf{x}) \Big)^T

In this setting, the cross-covariances :math:`k_{tt'}` are not arbitrary but derived by differentiation of a single latent covariance function :math:`k`:

.. math::

    k_{11}(\mathbf{x}, \mathbf{x}') &= k(\mathbf{x}, \mathbf{x}'), \\  
    k_{1j}(\mathbf{x}, \mathbf{x}') &= \frac{\partial}{\partial x'_j} k(\mathbf{x}, \mathbf{x}'), \\  
    k_{i1}(\mathbf{x}, \mathbf{x}') &= \frac{\partial}{\partial x_i} k(\mathbf{x}, \mathbf{x}'), \\  
    k_{ij}(\mathbf{x}, \mathbf{x}') &= \frac{\partial^2}{\partial x_i \partial x'_j} k(\mathbf{x}, \mathbf{x}')

Thus, GEK/DEGP can be viewed as a special case of MOGP where the correlation between outputs is dictated by calculus rather than learned freely. This structured interpretation highlights that gradient information augments the GP model within the same multi-output framework, providing richer posterior inference without introducing additional latent processes.

References
----------

.. bibliography::
   :cited:
   :style: unsrt