#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sparse Cholesky variant of the DEGP model.

Replaces the dense Cholesky factorisation in the NLML with a sparse
inverse-Cholesky factor U built using the paper's geometric sparsity
criterion: dist(x_P(i), x_P(j)) <= rho * l(j), where P is the MMD
(maximin) ordering and l(j) is the fill-distance at step j.

The sparsity pattern is computed once at model initialisation and is
independent of the kernel hyperparameters.
"""

