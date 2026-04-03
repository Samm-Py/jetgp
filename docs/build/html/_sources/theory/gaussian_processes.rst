Gaussian Processes
==================

.. math::

   \DeclareMathOperator*{\argmax}{arg\,max}
   \DeclareMathOperator*{\argmin}{arg\,min}

.. bibliography:: GP_bib.bib
   :style: unsrt
   :all:

Gaussian Processes (GPs), or Kriging models, have become one of the most commonly used surrogate modeling techniques in many engineering design applications due to their ability to deliver both predictions and uncertainty estimates :cite:`10.1115/DETC2019-97499,georges1963principles,Krig,GP`. They are a non-parametric supervised learning method used to solve regression, optimization, and probabilistic classification problems :cite:`rasmussen2006gaussian`. In particular, a Gaussian process can be viewed as a collection of random variables, any finite number of which have a joint Gaussian distribution. A Gaussian Process model can be completely determined by its mean function :math:`m(\mathbf{x})` and covariance function :math:`k(\mathbf{x}, \mathbf{x}')` of a real process :math:`f(\mathbf{x})` as given below:

.. math::
    :label: eqn2

    m(\mathbf{x}) &= \mathbb{E}[f(\mathbf{x})], \\
    k(\mathbf{x}, \mathbf{x}') &= \mathbb{E}[(f(\mathbf{x}) - m(\mathbf{x}))(f(\mathbf{x}') - m(\mathbf{x}'))]

The Gaussian Process can then be written as:

.. math::
    :label: eqn3

    f(\mathbf{x}) \sim \mathcal{GP}(m(\mathbf{x}), k(\mathbf{x}, \mathbf{x}'))

The covariance function in Equation :eq:`eqn2` is often approximated by a kernel function which assigns a specific covariance between pairs of random variables so that :math:`\text{cov}\left(f(\mathbf{x}), f(\mathbf{x}')\right) = k(\mathbf{x}, \mathbf{x}')`. The kernel function can take on various forms, one of the most commonly used functions, the squared exponential, is given below:

.. math::
    :label: eqn4

    K_{SE}(\mathbf{x}, \mathbf{x}', \sigma_f^2, \boldsymbol{\theta}) = \sigma_f^2 \exp\left(-\sum\limits_{i = 1}^{d} \frac{(x_i - x'_i)^2}{2 \theta_i^2} \right)

In this function, the parameters :math:`\sigma_f^2` and :math:`\boldsymbol{\theta}` collectively denoted as :math:`\boldsymbol{\psi} = \{\sigma_f^2, \boldsymbol{\theta}\}` are the hyperparameters of the model, which are learned from the data:

* :math:`\sigma_f^2` is the **variance**, which is a scaling factor that controls the overall vertical variation of the function.
* :math:`\theta_i` is the **length scale** for each input dimension *i*. It determines how quickly the correlation between points decreases with distance. A small length scale means the function varies rapidly, while a large length scale indicates a smoother function.

Making Predictions
------------------

To make predictions using a Gaussian process model, we assume an observed training dataset :math:`\mathcal{D}` containing *n* input-output pairs:

.. math::
    :label: data_set

    \mathcal{D} = \left\{\mathbf{X}, \mathbf{Y} \right\}  = \left\{(\mathbf{x}_i, y_i) \mid i = 1, \ldots, n \right\}

where each input :math:`\mathbf{x}_i \in \mathbb{R}^d` has a corresponding observed output :math:`y_i = f(\mathbf{x}_i)`. For now, we will assume the output of *f* is scalar, though this framework can be generalized to vector outputs, as discussed in Section Multi-Output GPs.

To predict outputs at a new set of test locations, :math:`\mathbf{X}_*`, we model the joint distribution over the training and test outputs. It is common to assume a zero-mean function for the GP, such that :math:`m(\mathbf{x}) = 0`. This assumption simplifies the model but is not overly restrictive, as it only constrains the prior distribution; the posterior predictive mean will be non-zero and adapt to the data. With these assumptions, the joint distribution of the training outputs :math:`\mathbf{y} = f(\mathbf{X})` and the test outputs :math:`\mathbf{y}_* = f(\mathbf{X}_*)` is given by:

.. math::
    :label: eqn_joint_dist

    \begin{pmatrix}
        \mathbf{y} \\
        \mathbf{y}_*
    \end{pmatrix}
    \sim \mathcal{N}\left(
    \begin{pmatrix}
        \mathbf{0} \\
        \mathbf{0}
    \end{pmatrix},
    \begin{pmatrix}
        K(\mathbf{X}, \mathbf{X}') & K(\mathbf{X}, \mathbf{X}_*) \\
        K(\mathbf{X}_*, \mathbf{X}) & K(\mathbf{X}_*, \mathbf{X}_*')
    \end{pmatrix}
    \right)

Here, the prime notation is a notational convenience: :math:`\mathbf{X}'` represents the same set of training points as :math:`\mathbf{X}`, but as the second argument of the covariance function. Thus :math:`K(\mathbf{X}, \mathbf{X}')` evaluates the kernel between all pairs of training points, producing an *n* × *n* covariance matrix with entries :math:`[K(\mathbf{X}, \mathbf{X}')]_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)`. For notational simplicity, we partition this covariance matrix into blocks:

.. math::
    :label: eqn_cov_blocks

    \begin{pmatrix}
        \boldsymbol{\Sigma}_{11} & \boldsymbol{\Sigma}_{12} \\
        \boldsymbol{\Sigma}_{21} & \boldsymbol{\Sigma}_{22}
    \end{pmatrix}
    =
    \begin{pmatrix}
        K(\mathbf{X}, \mathbf{X}') & K(\mathbf{X}, \mathbf{X}_*) \\
        K(\mathbf{X}_*, \mathbf{X}) & K(\mathbf{X}_*, \mathbf{X}_*')
    \end{pmatrix}

where :math:`\boldsymbol{\Sigma}_{11}` is the covariance of training points with themselves, :math:`\boldsymbol{\Sigma}_{12}` is the covariance of training points with test points, :math:`\boldsymbol{\Sigma}_{22}` is the covariance of the test points with themselves, and :math:`\boldsymbol{\Sigma}_{21} = \boldsymbol{\Sigma}_{12}^T`.

Posterior Predictive Distribution
----------------------------------

From the multivariate Gaussian theorem, the conditional distribution of a Gaussian is itself Gaussian. This fact allows for making predictions at the test points :math:`\mathbf{X}_*` conditioned on the training data :math:`\mathcal{D}`. Using the block matrix notation defined in Equation :eq:`eqn_cov_blocks`, the posterior predictive distribution is given by:

.. math::
    :label: eqn_gp_posterior_block

    p(\mathbf{y}_* \mid \mathbf{y}) &= \mathcal{N}(\boldsymbol{\mu}_{*}, \boldsymbol{\Sigma}_{*}) \\
    \boldsymbol{\mu}_{*} &= \boldsymbol{\Sigma}_{21} \boldsymbol{\Sigma}_{11}^{-1} \mathbf{y} \\
    \boldsymbol{\Sigma}_{*} &=  \boldsymbol{\Sigma}_{22} - \boldsymbol{\Sigma}_{21} \boldsymbol{\Sigma}_{11}^{-1} \boldsymbol{\Sigma}_{12}

Here, :math:`\boldsymbol{\mu}_{*}` is the posterior predictive mean and :math:`\boldsymbol{\Sigma}_{*}` is the posterior predictive covariance.

Learning Hyperparameters
-------------------------

The kernel hyperparameters, :math:`\psi`, are not set manually but are instead learned from the training data :math:`\mathcal{D}`. A standard approach is to find the values that maximize the log marginal likelihood of the observations, :math:`p(\mathbf{y} \mid \mathbf{X}, \boldsymbol{\psi})` :cite:`rasmussen2006gaussian`. For a noise-free model, the log marginal likelihood is given by:

.. math::
    :label: MLL

    \log p(\mathbf{y}|\mathbf{X}, \boldsymbol{\psi}) = -\frac{1}{2} \mathbf{y}^\top \mathbf{K}_{\boldsymbol{\psi}}^{-1} \mathbf{y} - \frac{1}{2}\log|\mathbf{K}_{\boldsymbol{\psi}}| - \frac{n}{2}\log 2\pi

where :math:`\mathbf{K}_{\boldsymbol{\psi}} = K(\mathbf{X},\mathbf{X})`. Optimized hyperparameters, :math:`\boldsymbol{\psi}^*`, are then selected according to:

.. math::
    :label: maxMLL

    \boldsymbol{\psi}^* = \argmax_{\boldsymbol{\psi}} \left( \log p(\mathbf{y}|\mathbf{X}, \boldsymbol{\psi}) \right)

Computational Considerations
-----------------------------

Note that directly computing the expression in Equation :eq:`MLL` requires evaluating both the inverse and determinant of the matrix :math:`K = k(\mathbf{X}, \mathbf{X}')`, operations that are computationally expensive and numerically unstable for large *n*. To address these challenges, the Cholesky decomposition is typically employed :cite:`rasmussen2006gaussian`. For a symmetric positive definite matrix *K*, the Cholesky decomposition yields a lower triangular matrix *L* such that :math:`K = LL^\top`. This factorization enables efficient computation of both the matrix inverse and the log-determinant. Indeed, the Cholesky method is approximately twice as fast as Gaussian elimination for symmetric positive definite matrices :cite:`Allaire2008-by`, and it provides better numerical stability, especially for large-scale problems where the kernel matrix may be ill-conditioned.


.. bibliography::
   :cited:
   :style: unsrt