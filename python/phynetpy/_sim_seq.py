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

Coalescent simulation of multilocus sequence data under the multispecies
*network* coalescent (MSNC) -- the *generative* counterpart of the likelihood
in :mod:`phynetpy._seq_likelihood` and the sampler in
:mod:`phynetpy._mcmc_seq`.

Given a known (true) species network with node heights, a population mutation
rate ``theta``, and a sampling scheme (how many gene copies / alleles are drawn
per species), this module:

1. **Simulates a gene tree per locus** by running the coalescent *backward in
   time* through the branches of the species network.  Within a branch ``u``
   ancestral lineages coalesce at total rate ``u(u-1)/theta`` (each of the
   ``C(u,2)`` pairs at rate ``2/theta``); at a reticulation node every lineage
   independently follows one of the two parent edges with probability ``gamma``
   / ``1 - gamma``; at the root the remaining lineages coalesce on an infinite
   branch down to a single MRCA.  This is *exactly* the process whose density
   :func:`phynetpy._seq_likelihood.gene_tree_msnc_log_density` evaluates, so
   simulated data are a fair, self-consistent test bed for the sampler.

2. **Evolves an alignment down each gene tree** under any
   :class:`~phynetpy._seq_likelihood.SubstitutionModel` (JC69 / HKY85 / GTR):
   the root sequence is drawn from the model's stationary frequencies and each
   site is propagated across every branch with that branch's transition matrix
   ``P(t) = exp(Qt)`` (vectorised across all sites at once).

These are the primitives behind the public :func:`phynetpy.infer.simulate`
verb, which wraps them and returns data-axis objects that feed straight back
into :func:`phynetpy.infer.infer`.  Reach for this module directly only when
you want a single gene tree or a bare ``{label: sequence}`` dict.

The result is a :class:`SimulatedData` bundle that drops straight into
:class:`phynetpy._mcmc_seq.MCMC_SEQ` (``MCMC_SEQ(**data.to_mcmc_seq_kwargs())``),
so you can check *recovery* (does the sampler find the true topology /
divergence times?) and *calibration* (do the 95% HPD intervals cover the truth
at the nominal rate?) against a known ground truth.

Units match the rest of the MCMC_SEQ stack: heights / branch lengths are in
expected substitutions per site and ``theta = 4 N mu`` (per-pair coalescent
rate ``2/theta``).

The coalescent sweep, the sequence evolution and the per-locus loop run in
the compiled core (``phynetpy._core``).  Every random draw still comes from
the caller's ``numpy.random.Generator`` in the same order as before, so a
seeded run reproduces.

Docs   - [x]
Tests  - [x]
Design - [x]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from . import _core
from ._core import _Lineage
from .Network import Network, Node, Edge
from ._seq_likelihood import (
    SubstitutionModel,
    JC69,
    DNA_STATES,
)
from ._units import (
    BranchLengthUnit,
    BranchThetaKey,
    require_branch_length_unit,
    resolve_branch_theta,
    validate_branch_thetas,
)


__all__ = [
    "SimulatedData",
    "simulate_gene_tree",
    "simulate_sequences",
    "simulate_multilocus",
]


# ======================================================================
# Result bundle
# ======================================================================

@dataclass
class SimulatedData:
    """A simulated multilocus data set with its ground truth.

    The fields under "inputs to inference" are exactly what
    :class:`phynetpy._mcmc_seq.MCMC_SEQ` consumes; the "ground truth" fields are
    what a recovery / calibration check compares the posterior against.

    Attributes:
        loci: Per-locus alignments (allele label -> aligned sequence string).
        mapping: Species label -> list of allele (gene-copy) labels sampled.
        species_of: Allele label -> species label (inverse of ``mapping``).
        gene_trees: The *true* simulated gene tree for each locus.
        true_network: The species network the data were generated on.
        true_theta: The population mutation rate used.
        true_branch_thetas: Fixed branch-specific population rates used.
        model: The substitution model used.
        seq_length: Number of sites simulated per locus.
    """

    # -- inputs to inference -----------------------------------------
    loci: list[dict[str, str]]
    mapping: dict[str, list[str]]
    species_of: dict[str, str]
    # -- ground truth ------------------------------------------------
    gene_trees: list[Network]
    true_network: Network
    true_theta: float
    model: SubstitutionModel
    seq_length: int = 0
    true_branch_thetas: Optional[dict[BranchThetaKey, float]] = None

    def to_mcmc_seq_kwargs(self) -> dict[str, Any]:
        """Keyword arguments to construct an :class:`MCMC_SEQ` on this data.

        Returns:
            Constructor keywords for ``MCMC_SEQ``. The true network and gene
            trees are deliberately *not* passed so recovery starts from an
            independent UPGMA guess.
        """
        return {
            "loci": self.loci,
            "mapping": self.mapping,
            "model": self.model,
            "theta": self.true_theta,
            "branch_thetas": (
                dict(self.true_branch_thetas)
                if self.true_branch_thetas else None
            ),
        }


# ======================================================================
# Reticulation inheritance probabilities
# ======================================================================

def _reticulation_parent_gammas(
    net: Network, retic: Node
) -> list[tuple[Edge, float]]:
    """The two parent edges of a reticulation with normalised inheritance probs.

    Robust to networks parsed from extended Newick where only one of the two
    hybrid edges carried a ``[&gamma=...]`` annotation (the other defaulting to
    ``0``): if the two stored gammas do not already sum to ~1 we recover the
    complement, falling back to an even ``0.5 / 0.5`` split when neither edge
    was annotated.

    Args:
        net: The species network.
        retic: A reticulation node (in-degree 2).

    Returns:
        ``[(edge0, gamma0), (edge1, gamma1)]`` with ``gamma0 + gamma1 == 1``.
    """
    return _core.sim_reticulation_parent_gammas(net, retic)


# ======================================================================
# Gene-tree simulation (multispecies network coalescent)
# ======================================================================

# ``_Lineage`` -- one ancestral gene lineage during the coalescent sweep
# (attributes ``newick`` and ``height``) -- is the compiled class imported
# above.


def _coalesce_within(
    lineages: list[_Lineage],
    t_start: float,
    t_end: Optional[float],
    theta: float,
    rng: np.random.Generator,
    counter: list[int],
) -> list[_Lineage]:
    """Run the coalescent on ``lineages`` over ``[t_start, t_end)``.

    Each pair of lineages coalesces at rate ``2/theta``; with ``u`` lineages the
    total rate is ``u(u-1)/theta`` and the waiting time is
    ``Exponential(rate)``.  Coalescences that would occur at or beyond
    ``t_end`` do not happen on this (finite) branch; ``t_end = None`` denotes
    the infinite root branch, on which lineages coalesce until one remains.

    Args:
        lineages: Open lineages entering the bottom of the branch.
        t_start: Branch bottom height (younger endpoint).
        t_end: Branch top height, or ``None`` for the infinite root branch.
        theta: Population mutation rate on this branch.
        rng: Random generator.
        counter: One-element list holding the next internal-node id (mutated).

    Returns:
        The lineages exiting the top of the branch (length 1 when ``t_end`` is
        ``None``).
    """
    return _core.sim_coalesce_within(
        lineages, t_start, t_end, theta, rng, counter
    )


def simulate_gene_tree(
    species_net: Network,
    alleles_per_species: dict[str, list[str]],
    theta: float,
    rng: np.random.Generator,
    *,
    branch_thetas: Optional[dict[BranchThetaKey, float]] = None,
) -> Network:
    """Simulate one gene tree under the multispecies network coalescent.

    Performs a backward-in-time, height-ordered sweep of ``species_net``.
    Sampled alleles enter at the leaves (height 0); lineages coalesce within
    each branch (:func:`_coalesce_within`); at a reticulation node each
    surviving lineage independently ascends one of the two parent edges with
    probability ``gamma`` / ``1 - gamma``; at the root the remaining lineages
    coalesce to a single MRCA.

    Args:
        species_net: Timed species network (ultrametric, substitution units)
            with reticulation inheritance probabilities on hybrid edges.
        alleles_per_species: Map species label -> the allele (gene-copy) labels
            to sample; these become the gene-tree leaf labels.  Species absent
            from the map contribute a single lineage named after the species.
        theta: Constant population mutation rate (used for every branch unless
            ``branch_thetas`` overrides it).
        rng: Random generator.
        branch_thetas: Optional fixed per-population ``theta`` values.

    Returns:
        A rooted, ultrametric :class:`Network` (a tree) with branch lengths in
        substitution units and leaves labelled by the sampled alleles.
    """
    return _core.sim_gene_tree(
        species_net, alleles_per_species, theta, rng, branch_thetas
    )


# ======================================================================
# Sequence simulation (evolve an alignment down a gene tree)
# ======================================================================

def _evolve_states(
    parent_states: np.ndarray, P: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Sample child states for every site given parent states and ``P(t)``.

    Vectorised inverse-CDF sampling: for each site we draw the child state from
    the row of ``P`` indexed by that site's parent state.

    Args:
        parent_states: ``int`` array of length ``n_sites`` (values in 0..3).
        P: ``4x4`` row-stochastic transition matrix.
        rng: Random generator.

    Returns:
        ``int`` array of child states, length ``n_sites``.
    """
    return _core.sim_evolve_states(parent_states, P, rng)


def simulate_sequences(
    gene_tree: Network,
    model: SubstitutionModel,
    n_sites: int,
    rng: np.random.Generator,
) -> dict[str, str]:
    """Evolve an alignment of ``n_sites`` down ``gene_tree`` under ``model``.

    The root sequence is drawn i.i.d. from the model's stationary
    frequencies; each branch then mutates every site with that branch's
    transition matrix ``P(t)``.

    Args:
        gene_tree: Rooted gene tree with branch lengths in substitution units.
        model: Nucleotide substitution model.
        n_sites: Number of sites (alignment columns) to simulate.
        rng: Random generator.

    Returns:
        Map from leaf label -> simulated nucleotide string (length ``n_sites``,
        alphabet ``A,C,G,T``).
    """
    return _core.sim_sequences(gene_tree, model, n_sites, rng)


# ======================================================================
# End-to-end multilocus simulation
# ======================================================================

def simulate_multilocus(
    species_net: Network,
    mapping: dict[str, list[str]],
    n_loci: int,
    seq_length: int,
    *,
    theta: float = 0.02,
    model: Optional[SubstitutionModel] = None,
    branch_thetas: Optional[dict[Any, float]] = None,
    seed: Any = None,
) -> SimulatedData:
    """Simulate a full multilocus data set on a known species network.

    For each of ``n_loci`` independent loci a gene tree is drawn under the MSNC
    (:func:`simulate_gene_tree`) and an alignment is evolved down it
    (:func:`simulate_sequences`).

    Args:
        species_net: The *true* timed species network (with gammas on hybrid
            edges).
        mapping: Species label -> list of allele labels to sample at every
            locus (use one allele per species for the single-individual case,
            or several to exercise multi-allele coalescence).
        n_loci: Number of independent loci to simulate.
        seq_length: Sites per locus.
        theta: Population mutation rate (constant across branches unless
            ``branch_thetas`` is given).
        model: Substitution model (default :class:`JC69`).
        branch_thetas: Optional fixed per-population ``theta`` values.
        seed: Seed for all randomness.

    Returns:
        A :class:`SimulatedData` bundle ready for
        ``MCMC_SEQ(**data.to_mcmc_seq_kwargs())``.
    """
    if n_loci <= 0:
        raise ValueError("n_loci must be positive.")
    validate_branch_thetas(branch_thetas)
    rng = np.random.default_rng(seed)
    model = model if model is not None else JC69()
    species_of = {a: sp for sp, alleles in mapping.items() for a in alleles}

    loci, gene_trees = _core.sim_multilocus(
        species_net,
        mapping,
        n_loci,
        seq_length,
        theta,
        model,
        branch_thetas,
        rng,
    )

    return SimulatedData(
        loci=loci,
        mapping=mapping,
        species_of=species_of,
        gene_trees=gene_trees,
        true_network=species_net,
        true_theta=theta,
        model=model,
        seq_length=seq_length,
        true_branch_thetas=(
            dict(branch_thetas) if branch_thetas else None
        ),
    )
