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

Likelihood foundation for :class:`phynetpy._mcmc_seq.MCMC_SEQ`.

This module implements -- bit-for-bit against PhyloNet's ``MCMC_SEQ``
(Wen & Nakhleh 2018, *Systematic Biology* 67(3):439-457, "Coestimating
Reticulate Phylogenies and Gene Trees from Multilocus Sequence Data") --
the two factors of the per-locus likelihood that the co-estimation sampler
maximises:

1. **Phylogenetic (Felsenstein) likelihood** ``P(S_i | g_i)`` -- the
   probability of the locus-``i`` alignment given its (timed, ultrametric)
   gene tree under a time-reversible nucleotide substitution model.  This
   is Felsenstein's (1981) pruning algorithm, vectorised across all sites of
   the alignment at once with site-pattern compression (the same trick
   BEAGLE uses), so a single matrix multiply scores every column.

2. **Multispecies-network-coalescent (MSNC) density** ``P(g_i | Psi)`` --
   the probability *density* (not the topology-only mass function used by
   :mod:`phynetpy._mcmc_gt`) of a gene tree with branch lengths embedded in
   a species network with node heights, per-branch population-mutation
   rates ``theta`` and reticulation inheritance probabilities ``gamma``.
   The per-branch kernel is exactly PhyloNet's
   ``GeneTreeBrSpeciesNetDistribution.calculateProbability``::

       per coalescence c (with u lineages just below it):
           (2/theta) * exp(-(c - prev) * u*(u-1) / theta)
       final no-coalescence segment on a finite branch [.., tau_high]:
           exp(-(tau_high - prev) * u_f*(u_f-1) / theta)
       and a gamma^u factor for the u lineages tracking a reticulation edge.

   The sum is over all coalescent histories (ancestral configurations);
   for a species *tree* there is a single history and the density reduces
   to the standard Rannala-Yang (2003) multispecies-coalescent density.

Units (critical for matching PhyloNet): node heights / branch lengths are in
**expected substitutions per site**; ``theta`` is the population mutation
rate ``4 N mu`` (so the per-pair coalescent rate is ``2/theta`` and the total
rate among ``u`` lineages is ``u(u-1)/theta``).

Conventions (also critical): nucleotides are ordered ``A,C,G,T`` -> ``0,1,2,3``
(the BEAST/BEAGLE order PhyloNet uses), and ambiguity / gap characters expand
to the standard IUPAC tip-partial 0/1 vectors.

The MSNC density delegates to the shared ancestral-configurations DP in
:mod:`phynetpy._msnc_density` (same kernel as :mod:`phynetpy._mcmc_gt`),
with explicit ``theta`` scaling for substitution-unit branch lengths.

The numerical work -- rate-matrix normalisation and eigendecomposition,
``P(t)``, site-pattern compression and the pruning sweep itself -- runs in
the compiled core (``phynetpy._core``); the classes here are the public
faces of those kernels and keep the original constructor signatures.

Docs   - [x]
Tests  - [x]
Design - [x]
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from . import _core
from .Network import Network  # noqa: F401  (gene trees are Network objects)

# Re-exported: one MSNC density implementation, shared with the MCMC_GT stack.
from ._msnc_density import gene_tree_msnc_log_density  # noqa: F401


__all__ = [
    "SubstitutionModel",
    "JC69",
    "GTR",
    "HKY85",
    "FelsensteinCalculator",
    "gene_tree_msnc_log_density",
    "DNA_STATES",
    "tip_partials_for_char",
]


# ======================================================================
# Nucleotide encoding (A=0, C=1, G=2, T=3) with IUPAC ambiguity
# ======================================================================

DNA_STATES: str = "ACGT"
_STATE_INDEX: dict[str, int] = {"A": 0, "C": 1, "G": 2, "T": 3}

# IUPAC ambiguity codes -> the set of concrete nucleotides they cover.  Maps
# every character we might see in an aligned DNA matrix to a length-4 tip
# partial-likelihood vector with 1.0 in the allowed states and 0.0 elsewhere
# (gaps / N / ? / X are fully ambiguous -> all ones).  The compiled kernel
# carries the same table; this copy documents the convention.
_IUPAC: dict[str, str] = {
    "A": "A", "C": "C", "G": "G", "T": "T", "U": "T",
    "R": "AG", "Y": "CT", "S": "CG", "W": "AT", "K": "GT", "M": "AC",
    "B": "CGT", "D": "AGT", "H": "ACT", "V": "ACG",
    "N": "ACGT", "?": "ACGT", "-": "ACGT", "X": "ACGT", ".": "ACGT",
    "O": "ACGT",
}


def tip_partials_for_char(ch: str) -> np.ndarray:
    """Length-4 tip partial-likelihood vector for a DNA character.

    Args:
        ch: A single (case-insensitive) IUPAC nucleotide / ambiguity / gap
            character.

    Returns:
        ``float64`` array of length 4 (order ``A,C,G,T``) with ``1.0`` in
        every state the character is compatible with and ``0.0`` elsewhere.
        Unknown characters are treated as fully ambiguous (all ones), matching
        PhyloNet's permissive tip handling.
    """
    return _core.tip_partials_for_char(ch)


# ======================================================================
# Substitution models
# ======================================================================

def _native_p_matrix(method):
    """Mark a Python ``p_matrix`` override as equal to the compiled one.

    :meth:`FelsensteinCalculator.log_likelihood` evaluates ``P(t)`` inside the
    compiled pruning loop only when a model's ``p_matrix`` is the compiled
    implementation or carries this mark.  Any other override is called back
    into Python for every branch, so user subclasses keep their semantics.
    """
    method._native_p_matrix = True
    return method


class SubstitutionModel(_core.SubstitutionModel):
    """Time-reversible 4-state nucleotide substitution model.

    Subclasses build the (un-normalised) instantaneous rate matrix ``Q`` via
    :meth:`_build_Q` from their stationary base frequencies ``pi`` and the six
    exchangeability rates.  The base class then normalises ``Q`` so that the
    expected substitution rate is 1 (``-sum_i pi_i Q_ii = 1``) -- this makes a
    branch length the expected number of substitutions per site, exactly as in
    PhyloNet/BEAST -- and computes ``P(t) = exp(Q t)`` via a single
    eigendecomposition that is reused for every branch length.

    Attributes:
        pi: Stationary base frequency vector (length 4, ``A,C,G,T``).
    """

    def __init__(self, pi: Sequence[float], rates: Sequence[float]) -> None:
        """Build and normalise the model.

        Args:
            pi: Stationary base frequencies (length 4; ``A,C,G,T``); must sum
                to 1.
            rates: The six exchangeability parameters in the order
                ``AC, AG, AT, CG, CT, GT``.
        """
        self._set_params(pi, rates)
        self._build_and_decompose()

    # -- construction -------------------------------------------------

    def _build_Q(self) -> np.ndarray:
        """Assemble the un-normalised, un-row-summed rate matrix.

        Returns:
            A ``4x4`` matrix whose off-diagonal entries are
            ``Q[i,j] = rate(i,j) * pi[j]`` (the time-reversible GTR
            parameterisation).  Diagonal is filled in by the caller.
        """
        return super()._build_Q()

    def _build_and_decompose(self) -> None:
        """Normalise ``Q`` to unit mean rate and eigendecompose it."""
        # The compiled core fills the diagonal, scales Q to unit mean rate and
        # performs the symmetric eigendecomposition via the pi^{1/2}
        # similarity transform (numerically far more stable than a general
        # eig(), and exactly how BEAST/PhyloNet decompose a reversible Q).
        self._decompose_from(self._build_Q())

    # -- transition probabilities ------------------------------------
    # ``p_matrix(t)`` -- P(t) = exp(Q t) from the cached eigendecomposition --
    # is inherited from the compiled base class.


class JC69(SubstitutionModel):
    """Jukes-Cantor (1969): equal base frequencies, equal rates.

    PhyloNet's default model.  Uses the closed-form transition matrix
    ``P_ii(t) = 1/4 + 3/4 e^{-4t/3}``, ``P_ij(t) = 1/4 - 1/4 e^{-4t/3}``,
    bypassing the eigendecomposition entirely for speed and exactness.
    """

    def __init__(self) -> None:
        """Construct the parameter-free JC69 model."""
        super().__init__([0.25, 0.25, 0.25, 0.25], [1.0] * 6)
        self._use_jc69_closed_form(True)

    @_native_p_matrix
    def p_matrix(self, t: float) -> np.ndarray:
        """Closed-form JC69 transition matrix (see class docstring)."""
        return super().p_matrix(t)


class HKY85(SubstitutionModel):
    """Hasegawa-Kishino-Yano (1985): free ``pi``, one transition/transversion ratio.

    Exchangeabilities are ``[1, kappa, 1, 1, kappa, 1]`` in ``AC,AG,AT,CG,CT,GT``
    order (transitions ``A<->G`` and ``C<->T`` scaled by ``kappa``).
    """

    def __init__(self, kappa: float, pi: Sequence[float]) -> None:
        """Build an HKY85 model.

        Args:
            kappa: Transition/transversion rate ratio (``> 0``).
            pi: Stationary base frequencies (length 4, ``A,C,G,T``).
        """
        self.kappa = float(kappa)
        super().__init__(pi, [1.0, self.kappa, 1.0, 1.0, self.kappa, 1.0])


# ``GTR`` is just the general base class with all six rates free; alias it so
# user-facing code can name it explicitly.
class GTR(SubstitutionModel):
    """General time-reversible model: free ``pi`` and all six exchangeabilities.

    Args mirror PhyloNet's ``-gtr (piA piC piG piT, rAC rAG rAT rCG rCT rGT)``.
    """

    def __init__(self, pi: Sequence[float], rates: Sequence[float]) -> None:
        """Build a GTR model.

        Args:
            pi: Stationary base frequencies (length 4, ``A,C,G,T``).
            rates: Six exchangeabilities ``AC,AG,AT,CG,CT,GT``.
        """
        super().__init__(pi, rates)


# ======================================================================
# Felsenstein pruning likelihood (vectorised over sites)
# ======================================================================

class FelsensteinCalculator(_core.FelsensteinCalculator):
    """Per-locus phylogenetic likelihood ``P(S | g)`` via vectorised pruning.

    On construction the alignment for one locus is compressed to its distinct
    site patterns (columns), each carrying an integer weight (how many original
    sites share that pattern).  Scoring a gene tree then runs one post-order
    sweep in which every node's conditional likelihoods are an
    ``(n_patterns, 4)`` array, so a single ``partials @ P.T`` matrix product
    propagates all sites across a branch at once.  This is the standard
    BEAGLE-style batching and is the source of the "sequences are very
    vectorizable" speed-up.

    Attributes:
        taxa: Ordered list of leaf labels present in the alignment.
        n_patterns: Number of distinct site patterns after compression.
    """

    def __init__(self, alignment: dict[str, str]) -> None:
        """Compress a single-locus alignment.

        Args:
            alignment: Map from leaf label to its aligned nucleotide string
                (all strings the same length).  IUPAC ambiguity / gaps are
                allowed and expand to tip partials.
        """
        self._compress(alignment)

    # ``log_likelihood(gene_tree, model, *, site_rate=1.0)`` -- the post-order
    # pruning sweep -- is inherited from the compiled base class.  It reads
    # the gene tree through the core's node/edge tables and evaluates P(t) in
    # Rust for the models defined here (or any subclass that keeps the
    # inherited ``p_matrix``); a Python ``p_matrix`` override is honoured by
    # calling back into it for every branch.


# MSNC density: shared implementation in phynetpy._msnc_density (re-exported above).
