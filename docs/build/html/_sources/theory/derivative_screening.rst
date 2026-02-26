Derivative Screening
====================

One approach to reducing computational cost selectively incorporates partial derivative information, using only a subset of available gradients rather than all :math:`d` derivatives :cite:`Ulaganathan2016-xq,Ulaganathan2014-yh`. Partial gradient-enhanced kriging (PGEK) extends this concept by systematically identifying which gradients to include :cite:`CHEN201915`. PGEK employs a two-step process: first, a feature selection technique, such as Mutual Information (MI), ranks the influence of each input variable on the output; second, an empirical evaluation rule determines the optimal number of gradients to include, balancing modeling efficiency and accuracy.

Suppose that feature selection identifies a subset of :math:`m \leq d` input variables :math:`\mathbf{X}_A = \{x_1, x_2, \ldots, x_m\}` whose derivatives provide the best trade-off between accuracy and efficiency. The formulation follows the structure of the derivative-enhanced GPs section, but with a reduced observation vector. The observation vector :math:`\mathbf{y}` is augmented to include only the selected partial derivatives, forming :math:`\mathbf{y}^{\text{PGEK}}`, and similarly for test predictions :math:`\mathbf{y}^{\text{PGEK}}_*`:

.. math::
    :label: eqn_pgek_aug_vectors

    \mathbf{y}^{\text{PGEK}} = \begin{bmatrix} f(\mathbf{X}) \\ \frac{\partial f(\mathbf{X})}{\partial x_1} \\ \vdots \\ \frac{\partial f(\mathbf{X})}{\partial x_m} \end{bmatrix}, \quad
    \mathbf{y}^{\text{PGEK}}_* = \begin{bmatrix} f(\mathbf{X}_*) \\ \frac{\partial f(\mathbf{X}_*)}{\partial x_1} \\ \vdots \\ \frac{\partial f(\mathbf{X}_*)}{\partial x_m} \end{bmatrix}

The joint distribution between the augmented training observations and test predictions remains multivariate Gaussian:

.. math::
    :label: eqn_pgek_joint_dist

    \begin{pmatrix}
        \mathbf{y}^{\text{PGEK}} \\
        \mathbf{y}^{\text{PGEK}}_*
    \end{pmatrix}
    \sim \mathcal{N}\left(
    \mathbf{0},
    \begin{pmatrix}
        \boldsymbol{\Sigma}^{\text{PGEK}}_{11} & \boldsymbol{\Sigma}^{\text{PGEK}}_{12} \\
        \boldsymbol{\Sigma}^{\text{PGEK}}_{21} & \boldsymbol{\Sigma}^{\text{PGEK}}_{22}
    \end{pmatrix}
    \right)

The training covariance block :math:`\boldsymbol{\Sigma}^{\text{PGEK}}_{11}` is an :math:`n(m + 1) \times n(m + 1)` matrix constructed using only the selected derivatives:

.. math::
    :label: eqn_pgek_S11

    \boldsymbol{\Sigma}^{\text{PGEK}}_{11} =
    \begin{pmatrix}
        K(\mathbf{X}, \mathbf{X}') & \frac{\partial K(\mathbf{X}, \mathbf{X}')}{\partial \mathbf{X}_A'} \\
        \frac{\partial K(\mathbf{X}, \mathbf{X}')}{\partial \mathbf{X}_A} & \frac{\partial^2 K(\mathbf{X}, \mathbf{X}')}{\partial \mathbf{X}_A \partial \mathbf{X}_A'}
    \end{pmatrix}

where :math:`\mathbf{X}_A` denotes the subset of selected input dimensions. The cross-covariance blocks :math:`\boldsymbol{\Sigma}^{\text{PGEK}}_{12}` and :math:`\boldsymbol{\Sigma}^{\text{PGEK}}_{22}` are constructed analogously. The posterior predictive distribution for :math:`\mathbf{y}^{\text{PGEK}}_*` follows the standard GP conditioning formula (analogous to Equation :eq:`eqn_full_gek_posterior`), and hyperparameters are optimized by maximizing the marginal log-likelihood (analogous to Equation :eq:`eqn_gek_mll`).

The key advantage of PGEK is the reduction in covariance matrix size from :math: :math:`n(d + 1) \times n(d + 1)` to :math:`n(m + 1) \times n(m + 1)`, where :math:`m \ll d`, resulting in substantial computational savings while retaining the most informative derivative information. Studies have demonstrated that PGEK can reduce modeling time by 30-60% compared to full GEK while maintaining or even improving accuracy in some cases :cite:`CHEN201915`.

.. bibliography::
   :cited:
   :style: unsrt