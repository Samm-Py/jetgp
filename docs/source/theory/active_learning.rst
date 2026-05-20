Cost-Aware Active Learning with Directional Derivatives
========================================================

This chapter describes JetGP's active-learning framework for adaptively
choosing where to evaluate an expensive function and along which directions
to evaluate its derivatives. The framework's distinguishing feature is a
**single unified, cost-aware acquisition rule** that compares function and
directional-derivative candidates against each other in normalised
information-per-cost units, rather than treating direction selection as a
separate post-processing step after a function point has been chosen.

Notation follows the rest of the theory manual: :math:`f:\mathcal{X}\to\mathbb{R}`
is the expensive scalar response over a bounded input domain
:math:`\mathcal{X}\subset\mathbb{R}^d`, :math:`p(\mathbf{x})` is the input
density associated with the task (uniform if unspecified), and the GP model
follows the directional-derivative formulation summarised in
:doc:`directional_degp`.

---

Acquisition Quantities
----------------------

After the current GP has been refit, two families of candidate observations
are scored: a single new **function evaluation** somewhere in
:math:`\mathcal{X}`, and one or more new **directional derivatives** at the
existing training points.

Function candidate
~~~~~~~~~~~~~~~~~~

The next candidate function location is obtained by maximising the
PDF-weighted log-variance acquisition over the bounded domain:

.. math::

    \mathbf{x}_{\mathrm{new}} \;=\;
    \arg\max_{\mathbf{x}\in\mathcal{X}}\bigl[\;
      \log \sigma_f^2(\mathbf{x}) \;+\; \log p(\mathbf{x})
    \;\bigr],

which is equivalent to maximising :math:`\sigma_f^2(\mathbf{x})\,p(\mathbf{x})`
wherever the density is positive. The log form is used for numerical
stability when the posterior variance or density weight is small. When
:math:`p` is uniform this reduces to plain maximum posterior variance.

The optimisation is performed by Latin-hypercube multistart followed by
local bounded L-BFGS-B from each start. A large finite penalty is returned
for infeasible or degenerate evaluations, so the finite-difference gradient
needed by L-BFGS-B remains well-defined.

Directional-derivative candidates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At each existing training point :math:`\mathbf{x}_i`, the posterior gradient
covariance under the current GP is

.. math::

    \mathbf{C}_d(\mathbf{x}_i) \;=\; \mathrm{Cov}\bigl[\nabla f(\mathbf{x}_i)\bigr]
    \in \mathbb{R}^{d\times d}.

For any unit direction :math:`\mathbf{v}`, the posterior variance of the
directional derivative is the Rayleigh quotient

.. math::

    \sigma^2_{\partial_{\mathbf{v}} f}(\mathbf{x}_i)
    \;=\;
    \mathbf{v}^{\top}\mathbf{C}_d(\mathbf{x}_i)\mathbf{v}.

Maximising this over :math:`\|\mathbf{v}\|=1` yields the dominant eigenvector
of :math:`\mathbf{C}_d(\mathbf{x}_i)`, with the maximum value being the
largest eigenvalue (Rayleigh-Ritz). Subsequent directions, taken over the
orthogonal complement of previously chosen directions at the same anchor, are
the remaining eigenvectors in descending eigenvalue order.

Each candidate direction is normalised against a prior reference scale,

.. math::

    \rho_j \;=\;
    \frac{\lambda_j\!\bigl(\mathbf{C}_d(\mathbf{x}_i)\bigr)}
         {\lambda_{\max}\!\bigl(\mathbf{K}_{\nabla}^{\mathrm{prior}}\bigr)},

where :math:`\mathbf{K}_{\nabla}^{\mathrm{prior}} =
\mathbf{K}_{\nabla\nabla}(\mathbf{x},\mathbf{x})` is the gradient
covariance implied by the kernel before conditioning on data. The
ratio :math:`\rho_j\in[0,1]` measures the residual posterior uncertainty in
direction :math:`\mathbf{v}_j` as a fraction of the largest derivative
uncertainty the model can express.

Implementation note (Lanczos top-:math:`k`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Full eigendecomposition of :math:`\mathbf{C}_d(\mathbf{x}_i)` is avoided in
the implementation. The :math:`k` leading eigenpairs are obtained by Lanczos
iteration with full re-orthogonalisation, where
:math:`k = \min(\texttt{max\_directions},\, d) - q_i` and :math:`q_i` is the
number of directional observations already taken at :math:`\mathbf{x}_i`.
This is significantly cheaper for high-dimensional inputs and naturally
respects the per-anchor cap ``max_directions``.

---

Unified Cost-Aware Score
------------------------

The cost-aware framework departs from the classical two-stage rule (pick
:math:`\mathbf{x}` by MSE, then pick directions at :math:`\mathbf{x}`) by
ranking function and derivative candidates against each other under a
**single information-per-cost score**.

Let :math:`c_f` and :math:`c_d` denote the costs of one function evaluation
and one directional-derivative observation respectively (both in arbitrary
matching units; see the discussion below). The score for the function
candidate at :math:`\mathbf{x}_{\mathrm{new}}` is

.. math::

    s_f \;=\;
    \frac{\sigma_f^2(\mathbf{x}_{\mathrm{new}})}
         {\lambda_{\max}\!\bigl(\mathbf{K}_f^{\mathrm{prior}}\bigr)}
    \,\cdot\, p(\mathbf{x}_{\mathrm{new}})
    \,\big/\, c_f
    \;=\;
    \rho_f\, p(\mathbf{x}_{\mathrm{new}}) \,/\, c_f,

and the score for a directional-derivative candidate at anchor
:math:`\mathbf{x}_i` with direction :math:`\mathbf{v}_j` is

.. math::

    s_{d,ij} \;=\; \rho_j(\mathbf{x}_i)\, p(\mathbf{x}_i) \,/\, c_d.

Both scores are dimensionless (information ratio per unit cost), so they
are commensurable across modalities. The next observation is then chosen by
joint argmax,

.. math::

    \mathrm{candidate}^{*} \;=\;
    \arg\max\bigl\{\,s_f,\; \{s_{d,ij}\}_{i,j}\,\bigr\},

subject to the affordability constraint :math:`c(\mathrm{candidate}) \le
B - C_{\mathrm{spent}}` where :math:`B` is the total cost budget and
:math:`C_{\mathrm{spent}}` is the cumulative cost spent so far.

A relative tolerance :math:`\rho_{\mathrm{tol}}` discards candidates whose
underlying :math:`\rho` (before density and cost scaling) falls below the
gate; setting :math:`\rho_{\mathrm{tol}}=0` keeps every feasible candidate.

Interpretation
~~~~~~~~~~~~~~

- When :math:`c_d \ll c_f` (e.g. automatic differentiation or OTI), the
  derivative scores are upweighted by :math:`c_f/c_d`, and the policy
  naturally prefers directional derivatives at existing anchors over fresh
  function evaluations.
- When :math:`c_d \gg c_f` (e.g. central finite differences, which cost
  :math:`2c_f` per directional derivative), function evaluations dominate.
- The density factor :math:`p(\mathbf{x})` is what makes the framework
  *test-distribution-aware*: candidates at low-density inputs are
  deprioritised even if their posterior uncertainty is high. Without this
  factor the policy chases variance at the corners of :math:`\mathcal{X}`,
  which may have negligible weight under the task distribution.

---

Hyperparameter Optimisation
---------------------------

The kernel hyperparameters :math:`\boldsymbol{\psi}` are obtained by
maximising the marginal log-likelihood corresponding to the current mixed
observation set:

.. math::

    \boldsymbol{\psi}^* \;=\;
    \arg\max_{\boldsymbol{\psi}}\;
    \log p(\mathbf{y}^{DD}\mid \boldsymbol{\psi}),

where :math:`\mathbf{y}^{DD}` is the augmented vector of function values and
directional derivatives observed so far, and the likelihood is constructed
from the directional-derivative covariance described in
:doc:`directional_degp`. JetGP's hybrid global+local optimiser is used:
a JADE adaptive differential-evolution global search followed by periodic
L-BFGS-B refinement of the best population member. Each refit after the
first is warm-started from the previously optimised hyperparameters, which
is appropriate in sequential design where each model update only modestly
perturbs the observation set.

---

Algorithm Summary
-----------------

The complete adaptive procedure has two phases: an initial enrichment of the
LHS design with directional derivatives, and a budget-constrained
sequential design loop.

.. code-block:: text

    Inputs:  initial DOE size n0; cost budget B; per-modality costs c_f, c_d;
             max_directions cap m; tolerance rho_tol; input density p(x).
    Output:  a directional-derivative-enhanced GP surrogate.

    1. Sample initial DOE X_0 (LHS over the input bounds), evaluate f(X_0),
       and fit a function-only GP by maximising the MLL.

    2. Initial derivative enrichment.
       For each anchor x_i in X_0:
         a. Form C_d(x_i) under the current GP.
         b. Extract the top-k eigenpairs via Lanczos
            (k = min(m, d) - q_i).
         c. Compute prior-relative ratios rho_j.
         d. Retain directions with rho_j > rho_tol.
         e. Evaluate the retained directional derivatives and refit the GP.

    3. Sequential design (cost-budget loop).
       While the budget is not exhausted:
         a. Build the candidate set: one function candidate via wMPV, and
            all admissible directional-derivative candidates at existing
            anchors (Lanczos top-k, filtered by max_directions cap).
         b. Score every candidate as s = rho * p(x) / c.
         c. Discard candidates whose remaining cost would exceed the
            remaining budget.
         d. Pick the candidate with the largest score.
         e. Evaluate the chosen observation, update the training data,
            and refit the GP with warm-started hyperparameter optimisation.
         f. Update cumulative cost; stop if the budget is reached.

The loop also terminates early if no admissible candidate clears
:math:`\rho_{\mathrm{tol}}`. The number of new directional derivatives at any
anchor is capped by ``max_directions``, which prevents pathological stacking
of derivative observations at a single point.

---

Relationship to Classical Active Learning Criteria
--------------------------------------------------

The cost-aware framework is a strict generalisation of the classical
maximum-posterior-variance (MPV) and PDF-weighted MPV criteria:

- If :math:`c_d=\infty` (derivatives forbidden) and :math:`p` is uniform,
  the function-only score reduces to plain MPV.
- If :math:`c_d=\infty` and :math:`p\not\equiv\mathrm{const.}`, the
  function-only score reduces to weighted MPV (the pointwise greedy
  approximation of weighted IMSE).
- If :math:`c_f=\infty` (function observations forbidden), the policy
  becomes a pure-directional-derivative active learner, picking
  eigenvectors of :math:`\mathbf{C}_d` at existing anchors.
- For finite :math:`c_f, c_d`, the policy adaptively blends both modalities
  in proportion to their normalised information-per-cost ratio.

---

Limitations and Future Extensions
---------------------------------

The acquisition score is *local* and *greedy*. Each :math:`\rho` measures
posterior uncertainty reduction at the anchor point itself, not at the
distribution of test points where prediction accuracy will ultimately be
evaluated. This is mathematically equivalent to a one-step look-ahead in
normalised posterior variance per cost, but only an *approximation* of the
true objective (integrated test-distribution mean-squared error).

The gap between the two is small for well-conditioned input distributions
with a clear density gradient, but it can widen for mixed input
distributions with flat-density marginals, where the cost-aware framework
can over-commit to a small number of high-variance anchor points. A
rigorous fix would replace the local :math:`\rho\,p(\mathbf{x})` score by
the expected reduction in integrated posterior variance over a Monte-Carlo
representation of the test distribution (an IMSE / predictive-variance-
reduction criterion). This is a planned future extension.

---

References
----------

.. bibliography::
   :cited:
   :style: unsrt
