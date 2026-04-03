Directional Derivative-Enhanced Gaussian Processes
==================================================

First-order directional derivatives offer an alternative approach to incorporating derivative information in Gaussian processes :cite:`Yao2024-hp,Padidar2021-fr`. Rather than using all :math:`d` partial derivatives at each training point as in GEK, directional derivative GPs (DDGP) use only :math:`q \ll d` derivatives along user-selected directions. This reduces computational cost while still capturing important local geometric information. A key feature of this approach is that the directions can be chosen uniquely for each training point, allowing for a more flexible and adaptive model.

For a training point :math:`\mathbf{x}_i`, let :math:`\mathbf{v}_{i,j} \in \mathbb{R}^d` denote the :math:`j`-th direction vector (with :math:`\|\mathbf{v}_{i,j}\| = 1`), for :math:`j = 1, \ldots, q`. The directional derivative of :math:`f` at :math:`\mathbf{x}_i` along direction :math:`\mathbf{v}_{i,j}` is:

.. math::

    \frac{\partial f(\mathbf{x}_i)}{\partial \mathbf{v}_{i,j}} = \nabla f(\mathbf{x}_i)^T \mathbf{v}_{i,j} = \sum_{k=1}^{d} \frac{\partial f(\mathbf{x}_i)}{\partial x_k} v_{i,j,k}

The training data is augmented with :math:`q` directional derivatives at each of the :math:`n` points. Let :math:`\mathbf{V} = \{\mathbf{v}_{i,j} \mid i = 1,\ldots,n, \, j = 1,\ldots,q\}` denote the collection of all training direction vectors. The augmented training vector is:

.. math::
    :label: eqn_ddgp_aug_vector

    \mathbf{y}^{DD} = \begin{bmatrix} 
        f(\mathbf{x}_1) \\
        \vdots \\
        f(\mathbf{x}_n) \\
        \frac{\partial f(\mathbf{x}_1)}{\partial \mathbf{v}_{1,1}} \\
        \vdots \\
        \frac{\partial f(\mathbf{x}_1)}{\partial \mathbf{v}_{1,q}} \\
        \vdots \\
        \frac{\partial f(\mathbf{x}_n)}{\partial \mathbf{v}_{n,q}}
    \end{bmatrix} \in \mathbb{R}^{n(1+q)}

For predictions at test locations :math:`\mathbf{X}_*` with direction vectors :math:`\mathbf{V}_*`, the augmented test vector :math:`\mathbf{y}^{DD}_*` is defined analogously.

Covariance Structure
--------------------

The joint distribution over the augmented training and test vectors is multivariate Gaussian:

.. math::
    :label: eqn_ddgp_joint

    \begin{pmatrix}
        \mathbf{y}^{DD} \\
        \mathbf{y}^{DD}_*
    \end{pmatrix}
    \sim \mathcal{N}\left(
    \mathbf{0},
    \begin{pmatrix}
        \boldsymbol{\Sigma}_{11}^{DD} & \boldsymbol{\Sigma}_{12}^{DD} \\
        \boldsymbol{\Sigma}_{21}^{DD} & \boldsymbol{\Sigma}_{22}^{DD}
    \end{pmatrix}
    \right)

The training covariance block :math:`\boldsymbol{\Sigma}_{11}^{DD}` is an :math:`n(1+q) \times n(1+q)` matrix with block structure:

.. math::
    :label: eqn_ddgp_S11

    \boldsymbol{\Sigma}_{11}^{DD} =
    \begin{pmatrix}
        K(\mathbf{X}, \mathbf{X}') & \frac{\partial K(\mathbf{X}, \mathbf{X}')}{\partial \mathbf{V}'} \\
        \frac{\partial K(\mathbf{X}, \mathbf{X}')}{\partial \mathbf{V}} & \frac{\partial^2 K(\mathbf{X}, \mathbf{X}')}{\partial \mathbf{V} \partial \mathbf{V}'}
    \end{pmatrix}

The upper-left block :math:`K(\mathbf{X}, \mathbf{X}')` is the standard :math:`n \times n` kernel matrix between function values. The off-diagonal blocks :math:`\frac{\partial K(\mathbf{X}, \mathbf{X}')}{\partial \mathbf{V}}` (size :math:`nq \times n`) contain covariances between directional derivatives and function values. The lower-right block :math:`\frac{\partial^2 K(\mathbf{X}, \mathbf{X}')}{\partial \mathbf{V} \partial \mathbf{V}'}` (size :math:`nq \times nq`) contains covariances between all pairs of directional derivatives across the training points.

Similarly, the training-test covariance block :math:`\boldsymbol{\Sigma}_{12}^{DD}` is an :math:`n(1+q) \times n_*(1+q_*)` matrix:

.. math::
    :label: eqn_ddgp_S12

    \boldsymbol{\Sigma}_{12}^{DD} =
    \begin{pmatrix}
        K(\mathbf{X}, \mathbf{X}_*) & \frac{\partial K(\mathbf{X}, \mathbf{X}_*)}{\partial \mathbf{V}_*'} \\
        \frac{\partial K(\mathbf{X}, \mathbf{X}_*)}{\partial \mathbf{V}} & \frac{\partial^2 K(\mathbf{X}, \mathbf{X}_*)}{\partial \mathbf{V} \partial \mathbf{V}_*'}
    \end{pmatrix}

where :math:`\boldsymbol{\Sigma}_{21}^{DD} = (\boldsymbol{\Sigma}_{12}^{DD})^T`, and :math:`\boldsymbol{\Sigma}_{22}^{DD}` has the same structure as :math:`\boldsymbol{\Sigma}_{11}^{DD}` but evaluated at test points. The posterior predictive distribution follows the standard GP conditioning formula:

.. math::
    :label: eqn_ddgp_posterior

    \boldsymbol{\mu}^{DD}_{*} &= \boldsymbol{\Sigma}_{21}^{DD} (\boldsymbol{\Sigma}_{11}^{DD})^{-1} \mathbf{y}^{DD}, \\
    \boldsymbol{\Sigma}^{DD}_{*} &=  \boldsymbol{\Sigma}_{22}^{DD} - \boldsymbol{\Sigma}_{21}^{DD} (\boldsymbol{\Sigma}_{11}^{DD})^{-1} \boldsymbol{\Sigma}_{12}^{DD}

Hyperparameter Optimization
----------------------------

Similar to other derivative-enhanced GPs, the kernel hyperparameters :math:`\boldsymbol{\psi}` are determined by maximizing the log marginal likelihood:

.. math::
    :label: eqn_ddgp_mll

    \log p(\mathbf{y}^{DD}|\mathbf{X}, \mathbf{V}, \boldsymbol{\psi}) = -\frac{1}{2} (\mathbf{y}^{DD})^\top (\boldsymbol{\Sigma}_{11}^{DD})^{-1} \mathbf{y}^{DD} - \frac{1}{2}\log|\boldsymbol{\Sigma}_{11}^{DD}| - \frac{n(1+q)}{2}\log 2\pi

Computational Advantages
------------------------

The primary advantage of DDGP is its favorable computational scaling. The matrix to be inverted has dimension :math:`M = n(1+q)`, resulting in :math:`\mathcal{O}((n(1+q))^3)` complexity. Since :math:`q` is typically chosen as a small constant (:math:`q \ll d`), this is substantially lower than the :math:`\mathcal{O}((n(d+1))^3)` cost of full GEK, making DDGP practical for high-dimensional problems where computing all partial derivatives would be prohibitive. The trade-off is that DDGP captures less geometric information than GEK, but with careful selection of directions (e.g., along principal curvature directions or gradient directions), it can still provide significant accuracy improvements over standard GP models.

References
----------

.. bibliography::
   :cited:
   :style: unsrt