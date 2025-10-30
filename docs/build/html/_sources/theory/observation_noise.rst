Note on Observation Noise
=========================

In practice, it is typical that the observed outputs are contaminated with independent Gaussian noise, so that

.. math::

    y_i = f(\mathbf{x}_i) + \epsilon_i, 
    \quad \epsilon_i \sim \mathcal{N}(0, \sigma_n^2)

Using the block notation :math:`\Sigma_{ij}` for covariance matrices, where :math:`\Sigma_{11}` denotes the covariance of the training outputs with themselves, the presence of noise modifies only this block:

.. math::

    \Sigma_{11} \;\;\longrightarrow\;\; \Sigma_{11} + \sigma_n^2 I

All posterior and marginal-likelihood expressions follow from this replacement. This convention applies generally, including cases with derivatives or multi-output structures.

Noisy Joint Distribution
-------------------------

.. math::
    :label: eqn_noisy_joint

    \begin{pmatrix}
        \mathbf{y} \\
        f(\mathbf{X}_*)
    \end{pmatrix}
    \sim \mathcal{N}\left(
    \mathbf{0},
    \begin{pmatrix}
        \Sigma_{11} + \sigma_n^2 I & \Sigma_{12} \\
        \Sigma_{21} & \Sigma_{22}
    \end{pmatrix}
    \right)

Noisy Posterior
---------------

.. math::
    :label: eqn_noisy_posterior

    p(f(\mathbf{X}_*) \mid \mathbf{y}) &= \mathcal{N}(\boldsymbol{\mu}_*, \boldsymbol{\Sigma}_*) \\
    \boldsymbol{\mu}_* &= \Sigma_{21} \left(\Sigma_{11} + \sigma_n^2 I\right)^{-1} \mathbf{y}, \\
    \boldsymbol{\Sigma}_* &= \Sigma_{22} - \Sigma_{21} \left(\Sigma_{11} + \sigma_n^2 I\right)^{-1} \Sigma_{12}

Noisy Marginal Log-Likelihood
------------------------------

.. math::
    :label: eqn_noisy_mll

    \log p(\mathbf{y} \mid \mathbf{X}, \boldsymbol{\psi}, \sigma_n^2)
    = -\tfrac{1}{2}\mathbf{y}^\top\!\left(\Sigma_{11} + \sigma_n^2 I\right)^{-1}\mathbf{y}
    -\tfrac{1}{2}\log\!\left|\Sigma_{11} + \sigma_n^2 I\right|
    -\tfrac{n}{2}\log(2\pi)

Generalization
^^^^^^^^^^^^^^

For augmented problems such as derivative-enhanced GPs or multi-output GPs, the same substitution rule applies. If the training covariance block is, for example, :math:`\Sigma^{G}_{11}` or :math:`\Sigma^{H}_{11}` of size :math:`M \times M`, then

.. math::

    \Sigma^{G}_{11} \;\;\longrightarrow\;\; \Sigma^{G}_{11} + \sigma_n^2 I_{M}, 
    \qquad
    \Sigma^{H}_{11} \;\;\longrightarrow\;\; \Sigma^{H}_{11} + \sigma_n^2 I_{M}

All posterior and marginal-likelihood expressions remain valid under this substitution.

Heteroscedastic Noise
^^^^^^^^^^^^^^^^^^^^^

If distinct observation types have different noise levels (e.g., function values versus gradients, or different outputs), the diagonal correction is replaced with a block-diagonal noise matrix

.. math::

    N = \operatorname{diag}(\underbrace{\sigma_{f}^2 I_{n}}_{\text{function block}},\; 
                              \underbrace{\sigma_{g}^2 I_{n d}}_{\text{gradient block}},\; \ldots)

and the substitution becomes :math:`\Sigma_{11} \to \Sigma_{11} + N`.


.. bibliography::
   :cited:
   :style: unsrt