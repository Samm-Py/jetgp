Weighted Gradient Enchanced Gaussian Processes
===============================================

.. math::

   \DeclareMathOperator*{\argmax}{arg\,max}
   \DeclareMathOperator*{\argmin}{arg\,min}


As previously outlined in the derivative-enhanced GPs section, operating with a large correlation matrix that includes all cross-correlations between function values and derivatives becomes computationally prohibitive for large datasets. An alternative strategy, known as weighted gradient-enhanced kriging (WGEK), alleviates this burden by decomposing the problem into multiple smaller submodels :cite:`HanWeightedGEK`. Rather than training a single GP on all derivative data, WGEK constructs submodels, each incorporating only a subset of the available derivatives. These submodels are then combined through a weighted sum that preserves the full set of interpolation conditions. The key advantage is that each submodel operates on a much smaller correlation matrix, avoiding the computational cost of decomposing the full gradient-enhanced covariance matrix.

Problem Formulation
-------------------

The WGEK formulation begins with an observed training dataset :math:`\mathcal{D}` containing :math:`n` input-output pairs with gradients:

.. math::

    \mathcal{D} = \left\{\mathbf{X}, f(\mathbf{X}), \nabla f(\mathbf{X})\right\} := \left\{(\mathbf{x}_i, f(\mathbf{x}_i), \nabla f(\mathbf{x}_i)) \mid i = 1, \ldots, n \right\}

The gradient information :math:`\nabla f(\mathbf{X})` is partitioned into :math:`M \leq n` disjoint sets:

.. math::

    D_1 &= \left\{\nabla f(\mathbf{x}_i) \mid  i = 1, \ldots, m_1\right\}, \\
    D_2 &= \left\{\nabla f(\mathbf{x}_i) \mid  i = m_1 + 1, \ldots, m_2\right\}, \\
    &\vdots \\
    D_M &= \left\{\nabla f(\mathbf{x}_i) \mid  i = m_{M-1} + 1, \ldots, n\right\},

where :math:`0 < m_1 < m_2 < \cdots < m_{M-1} < n` are cumulative partition boundaries. The training data for the :math:`k^{\text{th}}` submodel is then:

.. math::

    \mathcal{D}_k = \left\{(\mathbf{x}_i, f(\mathbf{x}_i)) \mid i = 1, \ldots, n \right\} \cup D_k

That is, each submodel contains all :math:`n` function values but only the gradient information from partition :math:`D_k`. Let :math:`\mathcal{I}_k = \{i \mid \nabla f(\mathbf{x}_i) \in D_k\}` denote the indices corresponding to partition *D*\ :sub:`k`, and let :math:`\mathbf{X}_k = \{\mathbf{x}_i \mid i \in \mathcal{I}_k\}` denote the locations of these gradients. The observation vector for the :math:`k^{\text{th}}` submodel is augmented to include only these selected gradients:

.. math::
    :label: eqn_wgek_aug_vector

    \mathbf{y}^{(k)} = \begin{bmatrix} 
        f(\mathbf{X}) \\
        \nabla f(\mathbf{X}_k)
    \end{bmatrix} \in \mathbb{R}^{n + d|\mathcal{I}_k|}

An important distinction of the WGEK formulation is that predictions are made only for function values :math:`f(\mathbf{X}_*)` at test locations :math:`\mathbf{X}_*`, not for derivatives. Because each submodel is trained on an incomplete set of gradient observations and the submodels are combined through weighted summation, the framework does not directly provide predictive distributions for derivative values or their associated uncertainties. This contrasts with the full GEK formulation and PGEK, which naturally support predictions for both function values and derivatives with uncertainty quantification.

Submodel Covariance Structure
------------------------------

For each submodel :math:`k`, the joint distribution between the augmented training observations :math:`\mathbf{y}^{(k)}` and the test predictions :math:`f(\mathbf{X}_*)` is multivariate Gaussian:

.. math::
    :label: eqn_wgek_joint_dist

    \begin{pmatrix}
        \mathbf{y}^{(k)} \\
        f(\mathbf{X}_*)
    \end{pmatrix}
    \sim \mathcal{N}\left(
    \mathbf{0},
    \begin{pmatrix}
        \boldsymbol{\Sigma}^{(k)}_{11} & \boldsymbol{\Sigma}^{(k)}_{12} \\
        \boldsymbol{\Sigma}^{(k)}_{21} & \boldsymbol{\Sigma}^{(k)}_{22}
    \end{pmatrix}
    \right)

The training covariance block :math:`\boldsymbol{\Sigma}^{(k)}_{11}` is an :math:`(n + d|\mathcal{I}_k|) \times (n + d|\mathcal{I}_k|)` matrix:

.. math::
    :label: eqn_wgek_S11

    \boldsymbol{\Sigma}^{(k)}_{11} =
    \begin{pmatrix}
        K(\mathbf{X}, \mathbf{X}') & \frac{\partial K(\mathbf{X}, \mathbf{X}_k')}{\partial \mathbf{X}_k'} \\
        \frac{\partial K(\mathbf{X}_k, \mathbf{X}')}{\partial \mathbf{X}_k} & \frac{\partial^2 K(\mathbf{X}_k, \mathbf{X}_k')}{\partial \mathbf{X}_k \partial \mathbf{X}_k'}
    \end{pmatrix}

The off-diagonal block :math:`\frac{\partial K(\mathbf{X}, \mathbf{X}_k')}{\partial \mathbf{X}_k'}` is an :math:`n \times d|\mathcal{I}_k|` matrix representing cross-covariances between function values at all :math:`n` training points and gradients at the :math:`|\mathcal{I}_k|` points in :math:`\mathbf{X}_k`. The lower-right block :math:`\frac{\partial^2 K(\mathbf{X}_k, \mathbf{X}_k')}{\partial \mathbf{X}_k \partial \mathbf{X}_k'}` is a :math:`d|\mathcal{I}_k| \times d|\mathcal{I}_k|` matrix representing cross-covariances between gradients at points in :math:`\mathbf{X}_k` only.

Since predictions of derivatives are not made at the test points :math:`\mathbf{X}_*`, the cross-covariance block :math:`\boldsymbol{\Sigma}^{(k)}_{12}` between training data and test points is a :math:`\left(n + d|\mathcal{I}_k|\right) \times n_*` matrix:

.. math::
    :label: eqn_wgek_S12

    \boldsymbol{\Sigma}^{(k)}_{12} =
    \begin{pmatrix}
        K(\mathbf{X}, \mathbf{X}_*) \\
        \frac{\partial K(\mathbf{X}_k, \mathbf{X}_*)}{\partial \mathbf{X}_k}
    \end{pmatrix}

and the test covariance block :math:`\boldsymbol{\Sigma}^{(k)}_{22} = K(\mathbf{X}_*, \mathbf{X}_*')`, an :math:`n_* \times n_*` matrix where :math:`n_* = |\mathbf{X}_*|` is the number of test points. The posterior predictive distribution for the :math:`k^{\text{th}}` submodel follows the standard GP conditioning formula (analogous to Equation :eq:`eqn_full_gek_posterior`):

.. math::
    :label: eqn_wgek_posterior

    \boldsymbol{\mu}^{(k)}_{*} &= \boldsymbol{\Sigma}^{(k)}_{21} \left(\boldsymbol{\Sigma}^{(k)}_{11}\right)^{-1} \mathbf{y}^{(k)}, \\
    \boldsymbol{\Sigma}^{(k)}_{*} &=  \boldsymbol{\Sigma}^{(k)}_{22} - \boldsymbol{\Sigma}^{(k)}_{21} \left(\boldsymbol{\Sigma}^{(k)}_{11}\right)^{-1} \boldsymbol{\Sigma}^{(k)}_{12}

Weighted Combination of Submodels
----------------------------------

By construction, each submodel prediction :math:`\boldsymbol{\mu}_*^{(k)}` exactly interpolates its training observations: :math:`\boldsymbol{\mu}_*^{(k)}(\mathbf{x}_i) = f(\mathbf{x}_i)` for all :math:`i`, and :math:`\nabla \boldsymbol{\mu}_*^{(k)}(\mathbf{x}_i) = \nabla f(\mathbf{x}_i)` for :math:`i \in \mathcal{I}_k`. The :math:`M` submodel predictions are then combined through a weighted summation to obtain the global prediction :math:`\boldsymbol{\mu}_*` at test locations :math:`\mathbf{X}_*`:

.. math::
    :label: eqn_wgek_weighted_prediction

    \boldsymbol{\mu}_* = \sum_{k=1}^{M} w_k \boldsymbol{\mu}_*^{(k)}

where :math:`w_k: \mathbb{R}^d \to \mathbb{R}` are weight functions. To ensure that the WGEK predictor preserves these interpolation conditions globally (i.e., :math:`\boldsymbol{\mu}_*(\mathbf{x}_i) = f(\mathbf{x}_i)` for all training points), the weights must satisfy:

.. math::
    :label: eqn_wgek_weight_constraints

    \sum_{k=1}^{M} w_k(\mathbf{x}) = 1 \quad \forall \mathbf{x} \in \mathbb{R}^d

and

.. math::

    w_k(\mathbf{x}_i) = \begin{cases}
        1 & \text{if } i \in \mathcal{I}_k, \\
        0 & \text{otherwise}.
    \end{cases}

That is, at each training point :math:`\mathbf{x}_i`, only the submodel containing that point's gradient information (submodel :math:`k` where :math:`i \in \mathcal{I}_k`) contributes to the prediction, thereby ensuring that both function and derivative interpolation conditions are satisfied.

Computing Weight Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^

To compute the weight functions satisfying these constraints, we follow the methodology outlined in :cite:`HanWeightedGEK`. For a given test set :math:`\mathbf{X}_*`, the weights are determined by solving the following constrained linear system:

.. math::
    :label: eqn_wgek_weight_system

    \begin{bmatrix}
        K(\mathbf{X}, \mathbf{X}') & \mathbf{f} \\ 
        \mathbf{f}^T & 0
    \end{bmatrix}
    \begin{bmatrix}
       \mathbf{W} \\ 
       \boldsymbol{\mu}
    \end{bmatrix} = 
    \begin{bmatrix}
        K(\mathbf{X}, \mathbf{X}_*)  \\ 
        \mathbf{1}_{1 \times n_*}
    \end{bmatrix}

where:

* :math:`\mathbf{W} \in \mathbb{R}^{n \times n_*}` contains weight contributions from each training point (rows) to each test point (columns)
* :math:`\mathbf{f} \in \mathbb{R}^{n \times 1}` is a vector of ones enforcing the partition-of-unity constraint
* :math:`\boldsymbol{\mu} \in \mathbb{R}^{1 \times n_*}` contains Lagrange multipliers for each test point
* :math:`\mathbf{1}_{1 \times n_*}` is a row vector of *n*\ :sub:`*` ones

The submodel weight coefficients :math:`w_k(\mathbf{x}_*^{(j)})` for the :math:`k`-th submodel at test point :math:`\mathbf{x}_*^{(j)}` are obtained by summing the weight contributions from all training points belonging to partition :math:`\mathcal{I}_k`:

.. math::
    :label: eqn_wgek_submodel_weights

    w_k(\mathbf{x}_*^{(j)}) = \sum_{i \in \mathcal{I}_k} W_{ij}

The predictive variance is computed following :cite:`HanWeightedGEK` by weighting the submodel standard deviations:

.. math::
    :label: eqn_wgek_weighted_variance

    \left[\boldsymbol{\Sigma}_{*}\right]_{jj} = \left(\sum_{k=1}^{M} w_k(\mathbf{x}_*^{(j)}) \sqrt{\left[\boldsymbol{\Sigma}^{(k)}_{*}\right]_{jj}}\right)^2

where :math:`\left[\boldsymbol{\Sigma}^{(k)}_{*}\right]_{jj}` is the predictive variance from the :math:`k`-th submodel at test point :math:`\mathbf{x}_*^{(j)}`.

Hyperparameter Optimization
----------------------------

As with the previous models, the kernel hyperparameters :math:`\boldsymbol{\psi}` must be determined. Since the submodels represent different views of the same underlying random process, they share the same hyperparameters. For the :math:`k`-th submodel, the marginal log-likelihood (MLL) of the augmented observations :math:`\mathbf{y}^{(k)}` is:

.. math::
    :label: eqn_wgek_mll

    \log p(\mathbf{y}^{(k)}|\mathbf{X}, \boldsymbol{\psi}) = -\frac{1}{2} \left(\mathbf{y}^{(k)}\right)^\top \left(\boldsymbol{\Sigma}^{(k)}_{11}\right)^{-1} \mathbf{y}^{(k)} - \frac{1}{2}\log|\boldsymbol{\Sigma}^{(k)}_{11}| - \frac{n + d|\mathcal{I}_k|}{2}\log 2\pi

Following :cite:`HanWeightedGEK`, the shared hyperparameters are determined by maximizing an aggregated log-likelihood function:

.. math::
    :label: eqn_wgek_jll

    \text{JLL}(\boldsymbol{\psi}) = \frac{1}{M}\sum_{k=1}^{M}\log p(\mathbf{y}^{(k)}|\mathbf{X}, \boldsymbol{\psi})

This uniform averaging provides a computationally efficient approximation to the full joint likelihood while ensuring all submodels contribute equally to the hyperparameter estimation. The optimal hyperparameters are obtained as:

.. math::
    :label: eqn_wgek_optimal_psi

    \boldsymbol{\psi}^* = \argmax_{\boldsymbol{\psi}} \text{JLL}(\boldsymbol{\psi})

Computational Trade-offs
-------------------------

This decomposition reduces the total training complexity from :math:`\mathcal{O}((n(d+1))^3)` for standard GEK to :math:`\mathcal{O}\left(\sum_{k=1}^M (n + d|\mathcal{I}_k|)^3\right)` for WGEK, providing substantial computational savings when the partitions distribute the gradient information appropriately. However, this computational efficiency comes at the cost of model accuracy. By partitioning the derivative data, WGEK disrupts the derivative cross-correlation structure that full GEK exploits. As the number of partitions :math:`M` increases, each submodel contains less derivative information, making the global predictor progressively less informative. The extreme case of :math:`M=n` (one gradient per submodel) provides maximum computational savings but minimal derivative cross-correlation information, resulting in the least accurate predictions. Conversely, reducing :math:`M` retains more of the derivative correlation structure, improving model accuracy at the expense of increased computational cost. Thus, the choice of :math:`M` represents a trade-off between computational efficiency and predictive fidelity.

References
----------

.. bibliography::
   :cited:
   :style: unsrt