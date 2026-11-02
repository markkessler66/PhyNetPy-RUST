#! /usr/bin/env python
# -*- coding: utf-8 -*-

##############################################################################
##  -- PhyNetPy --
##  Library for the Development and use of Phylogenetic Network Methods
##
##  Copyright 2025 Mark Kessler, Luay Nakhleh.
##  All rights reserved.
##
##  See "LICENSE.txt" for terms and conditions of usage.
##
##  If you use this work or any portion thereof in published work,
##  please cite it as:
##
##     Mark Kessler, Luay Nakhleh. 2025.
##
##############################################################################

"""
Author : Mark Kessler
First Included in Version : 0.5.0

Shared proposal-math for MCMC over phylogenetic-network space.

This module is the **single source of truth** for the proposal densities,
Hastings ratios and reversible-jump Jacobians used by every PhyNetPy
network sampler.  Previously this math lived in two unrelated places --
:mod:`phynetpy._mcmc_seq` (correct, log-space, with RJMCMC terms) and
:mod:`phynetpy.ModelMove` (an SA-oriented shortcut that returned ``1.0``
for everything).  Maintaining the math twice meant the gene-tree / MPL
samplers silently drifted out of correctness.  Both stacks now delegate
here.

Unified move contract
----------------------

Every proposal, regardless of which stack drives it, conceptually returns

    (log_hastings_ratio, undo)

where ``log_hastings_ratio`` is in **log space** (``0.0`` for a symmetric
proposal) and already folds in any reversible-jump log-Jacobian.  A driver
accepts a proposal with

    log_alpha = (log_pi_proposed - log_pi_current) + log_hastings_ratio
    accept    = log_alpha >= 0 or log(U(0,1)) < log_alpha

The sequence sampler (:mod:`phynetpy._mcmc_seq`) realises this contract via
``op_*`` closures; the gene-tree / MPL sampler realises it via
:class:`phynetpy.ModelMove.Move` objects that cache the value computed in
``execute`` on ``self._log_hr``.  Both call the pure functions below so the
formulas can never disagree.

Functions here are deliberately pure (numbers in, numbers out) and have no
dependency on ``Network``, ``SeqState`` or ``Model`` so they can be unit
tested in isolation and reused from either stack.
"""

from __future__ import annotations

import math
import warnings

from . import _core

__all__ = [
    "add_reticulation_log_hastings",
    "remove_reticulation_log_hastings",
    "normal_log_pdf",
    "standard_normal_cdf",
    "truncated_normal_log_norm",
    "random_walk_truncated_log_hastings",
    "gaussian_delta_log_hastings",
    "weighted_choice_log_hastings",
    "LargeAnalysisWarning",
    "MCMC_TAXA_ADVISORY",
    "warn_if_large_mcmc",
]

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


# ======================================================================
# Tractability advisory for large MCMC analyses
# ======================================================================
#
# MCMC sampling of phylogenetic *networks* scales badly with the number of
# taxa: the topology space grows super-exponentially, chains mix poorly
# (low ESS / high Gelman-Rubin R-hat), and the per-iteration MSNC cost
# climbs steeply.  Past roughly this many taxa a converged posterior in any
# practical number of iterations is unlikely.  We do **not** forbid such a
# run -- a user who knows what they are doing may proceed -- but we emit a
# loud one-time warning advising against it.

MCMC_TAXA_ADVISORY = 20


class LargeAnalysisWarning(UserWarning):
    """Advisory that an MCMC analysis is large enough to be intractable.

    A subclass of :class:`UserWarning` so callers can silence or escalate it
    selectively, e.g. ``warnings.filterwarnings("ignore",
    category=phynetpy._network_moves.LargeAnalysisWarning)``.
    """


def warn_if_large_mcmc(n_taxa: int, *, method: str = "MCMC") -> bool:
    """Warn (without blocking) when an MCMC run is likely intractable.

    Args:
        n_taxa: Number of taxa (species) in the analysis.
        method: Sampler name used in the message (e.g. ``"MCMC_SEQ"``).

    Returns:
        ``True`` when a warning was emitted (``n_taxa >= MCMC_TAXA_ADVISORY``),
        ``False`` otherwise.  The run is never blocked either way.
    """
    if n_taxa < MCMC_TAXA_ADVISORY:
        return False

    warnings.warn(
        f"{method}: {n_taxa} taxa requested. MCMC sampling over phylogenetic "
        f"network space is generally intractable beyond ~{MCMC_TAXA_ADVISORY} "
        f"taxa -- the topology space grows super-exponentially, chains tend to "
        f"mix poorly (low ESS / high R-hat), and the per-iteration likelihood "
        f"cost climbs steeply. The analysis will still run, but it may not "
        f"converge in any practical number of iterations and the posterior "
        f"may be unreliable. Strongly consider: fewer taxa, a constrained or "
        f"fixed backbone topology, or a non-MCMC method (e.g. maximum "
        f"pseudo-likelihood via MPL / InferNetworkML).",
        category=LargeAnalysisWarning,
        stacklevel=3,
    )
    return True


# ======================================================================
# Reversible-jump reticulation moves (PhyloNet MCMC_SEQ formulation)
# ======================================================================
#
# These mirror PhyloNet's ``AddReticulation`` / ``DeleteReticulation``
# acceptance terms exactly.  The add and delete ratios are constructed to
# be exact negatives of each other (so an add immediately followed by the
# matching delete has a combined log-Hastings of ~0), which is what makes
# the dimension-changing pair reversible.  See
# ``phynetpy._mcmc_seq.op_add_reticulation`` for the canonical derivation
# and ``tests/test_mcmc_seq.py::TestReticulationMoves`` for the invariance
# check.


def add_reticulation_log_hastings(
    *, edge_count_pre: int, retic_count_pre: int, l1: float, l2: float
) -> float:
    r"""Log Hastings ratio (incl. RJ Jacobian) for adding a reticulation.

    ``log(pda * l1 * l2 * E (E - 1) / (2 (R + 1)))`` with ``E`` the pre-move
    edge count, ``R`` the pre-move reticulation count and ``pda = 0.5``
    only when adding the first reticulation.

    Raises:
        ValueError: If ``E < 2`` or either length is not positive.
    """
    return _core.add_reticulation_log_hastings_py(
        int(edge_count_pre), int(retic_count_pre), float(l1), float(l2)
    )


def remove_reticulation_log_hastings(
    *, edge_count_post: int, retic_count_pre: int, l1: float, l2: float
) -> float:
    r"""Log Hastings ratio (incl. RJ Jacobian) for deleting a reticulation.

    Exact negative of :func:`add_reticulation_log_hastings` evaluated at the
    reverse-add quantities: ``log(pad * 2 R / (l1 l2 E' (E' - 1)))`` with
    ``pad = 2`` only when removing the last reticulation.

    Raises:
        ValueError: If ``E' < 2``, ``R < 1`` or either length is not positive.
    """
    return _core.remove_reticulation_log_hastings_py(
        int(edge_count_post), int(retic_count_pre), float(l1), float(l2)
    )


def normal_log_pdf(x: float, mu: float, sigma: float) -> float:
    """Log density of ``N(mu, sigma^2)`` evaluated at ``x``."""
    return _core.normal_log_pdf_py(float(x), float(mu), float(sigma))


def standard_normal_cdf(x: float) -> float:
    """Standard-normal CDF ``Phi(x)``."""
    return _core.standard_normal_cdf_py(float(x))


def truncated_normal_log_norm(
    mu: float, sigma: float, lo: float, hi: float
) -> float:
    r"""Log normalising constant of ``N(mu, sigma^2)`` truncated to ``[lo, hi]``."""
    return _core.truncated_normal_log_norm_py(float(mu), float(sigma), float(lo), float(hi))


def random_walk_truncated_log_hastings(
    *, old: float, new: float, sigma: float, lo: float, hi: float
) -> float:
    r"""Log Hastings ratio for a truncated-Gaussian random walk on ``[lo, hi]``:
    ``log Z(old) - log Z(new)``."""
    return _core.random_walk_truncated_log_hastings_py(
        float(old), float(new), float(sigma), float(lo), float(hi)
    )


def gaussian_delta_log_hastings(
    *,
    delta: float,
    sigma_fwd: float,
    sigma_rev: float,
    lo_fwd: float,
    hi_fwd: float,
    lo_rev: float,
    hi_rev: float,
) -> float:
    r"""Log Hastings ratio for a symmetric-delta slide with state-dependent
    scale (the :class:`phynetpy.ModelMove.ChangeNodeHeight` proposal)."""
    return _core.gaussian_delta_log_hastings_py(
        float(delta), float(sigma_fwd), float(sigma_rev),
        float(lo_fwd), float(hi_fwd), float(lo_rev), float(hi_rev),
    )


def weighted_choice_log_hastings(
    *, fwd_weight: float, fwd_total: float, rev_weight: float, rev_total: float
) -> float:
    r"""Log Hastings ratio for an asymmetric weighted discrete choice:
    ``log((rev_weight / rev_total) / (fwd_weight / fwd_total))``."""
    return _core.weighted_choice_log_hastings_py(
        float(fwd_weight), float(fwd_total), float(rev_weight), float(rev_total)
    )
