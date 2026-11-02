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
Last Stable Edit : 4/24/26
First Included in Version : 0.3.2

Docs   - [x]
Tests  - [ ]
Design - [x]

Maximum pseudo-likelihood (MPL) for phylogenetic network inference.

Implements the scoring function of

    Yu, Y. & Nakhleh, L. (2015). "A maximum pseudo-likelihood approach
    for phylogenetic networks." BMC Genomics, 16(S10), S10.

whose objective (to maximise) is

    log L(Psi, gamma | G)
        = sum_{X,Y,Z} [
              rho(XY|Z) * log P(XY|Z | Psi, gamma)
            + rho(XZ|Y) * log P(XZ|Y | Psi, gamma)
            + rho(YZ|X) * log P(YZ|X | Psi, gamma)
          ]

where Psi is the candidate species network, gamma is the vector of
reticulation inheritance probabilities, G is the input gene-tree set,
rho(·|·) are empirical triplet frequencies computed from G, and
P(·|·) are expected triplet probabilities under the coalescent model
on Psi.

Pipeline overview
-----------------
End-to-end control flow when a user runs ``MPL.search(...)``:

    1. Enumerate all 3-taxon species triplets from the species labels.

    2. Precompute rho (triplet frequencies from gene trees) once.
       Driver:  ``compute_gene_tree_triplets`` -> ``_compute_all_rhos_fast``
                -> ``_process_tree_batch`` (optionally parallel via
                ``ProcessPoolExecutor``).  Uses ``_GeneTreeLCAIndex``
                for O(1) topology queries per triplet per tree.

    3. Build an initial ``Model`` wrapping the starting species network.
       Attach an ``MPLScorer`` so the search driver can call
       ``score(model)`` to get log-pseudo-likelihood.

    4. Run the search driver (``HillClimbing`` or ``SimulatedAnnealing``)
       using ``MPLKernel`` to propose moves.  Each accepted proposal
       mutates the model's network in place; the kernel adapts move
       weights and continuous proposal sigmas from rolling per-move
       statistics.

    5. After the search, report final score, move-kernel stats, and
       (optionally) a reference-network comparison.

Scoring path (per call to ``MPLScorer.__call__``):

    Model.network  ->  _TripleDPEngine(network)  ->  either
      - _score_with_cython(engine, triplets, rho)  (fast path), or
      - Python fallback: engine.calculate_triple_probability(triplet)
        summed over all active triplets.

The Cython path and the Python path compute the same quantity; the
Python path is the reference implementation kept for portability and
debugging.  ``_HAS_CYTHON_MPL`` gates which one is used.

Module layout
-------------
Sections are marked with banner comments.  Top-to-bottom:

    (A) Result containers                  (GeneTreeTripletResult, ...)
    (B) Reference-comparison helpers       (format_mpl_reference_comparison)
    (C) Rho precomputation                 (_GeneTreeLCAIndex, _compute_all_rhos_fast)
    (D) Subnetwork probability DP engine   (_TripleDPEngine)
    (E) Cython bridge                      (_extract_topology_for_cython)
    (F) Public triplet API                 (compute_gene_tree_triplets, score_species_network_triplets)
    (G) Scorer + adaptive proposal kernel  (MPLScorer, _AdaptiveConfig, MPLKernel)
    (H) Orchestration class                (MPL)
"""
 
from __future__ import annotations

import copy
import math
import os
import warnings
from collections import deque
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np

from .Network import Network, Node
from .GeneTrees import GeneTrees
from . import IO as io
from .ModelGraph import Model
from .GraphUtils import count_reticulations, _hardwired_clusters_by_label
from .MetropolisHastings import ProposalKernel, HillClimbing, SimulatedAnnealing
from .ModelMove import (
    Move,
    SPR,
    ChangeNodeHeight,
    ChangeInheritanceProb,
    AddReticulation,
    RemoveReticulation,
    FlipReticulation,
    ChangeReticSource,
    ChangeReticDest,
    RelocateReticulation,
)
from ._optimize import optimize_network_parameters
from ._search_flags import (
    resolve_move_types, make_level_validator, make_containment_validator,
    CONTINUOUS_MOVES, resolve_search_preset,
)
 
# log(p) for zero-probability triplet outcomes is replaced by this
# floor to keep the score finite.  math.log(1e-200) ≈ -460; any triplet
# the DP engine assigns probability 0 is effectively penalised by this
# amount per unit of observed rho mass.  Chosen large enough that the
# search is strongly deterred from such topologies but not so large
# that it single-handedly dominates the score.
_LOG_FLOOR = math.log(1e-200)

# Compiled backend.  The triplet DP and the rho precomputation run in the
# Rust core (``phynetpy._core``); ``_HAS_CYTHON_MPL`` keeps its historical
# name for callers that consulted it, and is always True.
from ._core import _MplTopology, _TripletIndex, _compute_rhos

_HAS_CYTHON_MPL = True


# ======================================================================
# (A) Result containers
# ======================================================================

class GeneTreeTripletResult:
    """Container for precomputed gene-tree triplet frequencies (rho values).

    Returned by :func:`compute_gene_tree_triplets` and consumed by
    :func:`score_species_network_triplets` (or stored on :class:`MPL`
    for repeated scoring against varying species networks).  Encodes
    the "gene-tree side" of the MPL inputs: all rho values are
    functions of the observed gene trees only and never change during
    a network search.

    Attributes:
        triplets: Canonical ordered species triplets, e.g.
            ``[('A', 'B', 'C'), ('A', 'B', 'D'), ...]``.
        rho_by_triplet: For each triplet ``(X, Y, Z)``, the empirical
            frequencies of the three rooted resolutions as a tuple
            ``(rho_XY|Z, rho_XZ|Y, rho_YZ|X)``.  Values sum to
            approximately the number of gene trees that covered all
            three taxa (ties contribute 1/3 to each resolution).
    """

    def __init__(
        self,
        triplets: list[tuple[str, str, str]],
        rho_by_triplet: dict[tuple[str, str, str], tuple[float, float, float]],
    ) -> None:
        """Initialise the container.

        Args:
            triplets: Canonical species triplets.
            rho_by_triplet: Mapping ``triplet -> (rho_XY|Z, rho_XZ|Y,
                rho_YZ|X)``.
        """
        self.triplets: list[tuple[str, str, str]] = triplets
        self.rho_by_triplet: dict[
            tuple[str, str, str], tuple[float, float, float]
        ] = rho_by_triplet


class SpeciesNetworkTripletResult:
    """Container for species-network triplet probabilities and final score.

    Returned by :func:`score_species_network_triplets`.  Holds the
    "species-network side" of the scoring call: per-triplet predicted
    probabilities plus the summed log-pseudo-likelihood.

    Attributes:
        triplets: Canonical ordered species triplets (matching the
            order used during scoring).
        probs_by_triplet: For each triplet, predicted probabilities
            ``(P(XY|Z), P(XZ|Y), P(YZ|X))`` under the DP model.
        log_pseudo_likelihood: Total MPL score (higher is better;
            always <= 0 since it is a sum of ``rho * log(p)`` terms).
    """

    def __init__(
        self,
        triplets: list[tuple[str, str, str]],
        probs_by_triplet: dict[tuple[str, str, str], tuple[float, float, float]],
        log_pseudo_likelihood: float,
    ) -> None:
        """Initialise the container.

        Args:
            triplets: Canonical species triplets.
            probs_by_triplet: Mapping ``triplet -> (P(XY|Z), P(XZ|Y),
                P(YZ|X))`` under the DP model.
            log_pseudo_likelihood: Final pseudo-likelihood score.
        """
        self.triplets: list[tuple[str, str, str]] = triplets
        self.probs_by_triplet: dict[
            tuple[str, str, str], tuple[float, float, float]
        ] = probs_by_triplet
        self.log_pseudo_likelihood: float = log_pseudo_likelihood


# ======================================================================
# (B) Reference-comparison helpers
#
# These are only used when ``MPL.search(reference_network=...)`` is set
# (e.g. when benchmarking against a simulator ground truth).  They
# don't participate in the scoring path.
# ======================================================================

def format_mpl_reference_comparison(
    found: Network,
    reference: Network,
    rho: dict[tuple[str, str, str], tuple[float, float, float]],
    triplets: list[tuple[str, str, str]],
    *,
    top_k: int = 25,
) -> str:
    """Build a text report comparing a found network against a reference.

    The report covers four sections:

      1. Final MPL scores for both networks and the signed gap.
      2. Per-network reticulation summary (parent labels, subtree
         leaves) to eyeball where each reticulation sits.
      3. Clade-level Venn counts: shared, only-in-found, only-in-reference.
      4. Top ``top_k`` triplets where the reference outperforms the
         found network, ranked by log-likelihood gap -- useful for
         diagnosing *why* the found topology under-fits.

    Uses the same ``rho`` and ``triplets`` passed to :class:`MPL` so
    the reference is scored against the same observed gene-tree
    statistics.

    Args:
        found: Best network returned by the search.
        reference: Ground-truth or baseline network.
        rho: Triplet frequency table (gene-tree side).
        triplets: Canonical triplets to iterate over.
        top_k: How many worst-performing triplets to list.

    Returns:
        Multi-line formatted report string (no trailing newline).
    """
    scorer = MPLScorer(rho, triplets)
    m_found = Model()
    m_found.network = found
    m_ref = Model()
    m_ref.network = reference
    s_found = scorer(m_found)
    s_ref = scorer(m_ref)
    gap = s_ref - s_found

    lines: list[str] = []
    lines.append("=== MPL reference comparison ===")
    lines.append(f"Found score:     {s_found:.4f}")
    lines.append(f"Reference score: {s_ref:.4f}")
    lines.append(f"Gap (ref - found): {gap:.4f}  (positive => reference is better)")
    lines.append("")

    def _retic_lines(net: Network, label: str) -> None:
        """Append one line per reticulation describing parents + subtree."""
        retics = [n for n in net.V() if net.in_degree(n) > 1]
        lines.append(f"{label} reticulations: {len(retics)}")
        for r in retics:
            sub = sorted(n.label for n in net.leaf_descendants(r))
            pars = [p.label for p in net.get_parents(r)]
            lines.append(f"  {r.label}: parents={pars}, subtree_leaves={sub}")

    _retic_lines(found, "Found")
    _retic_lines(reference, "Reference")
    lines.append("")

    cf = _hardwired_clusters_by_label(found)
    cr = _hardwired_clusters_by_label(reference)
    shared = cf & cr
    only_f = cf - cr
    only_r = cr - cf
    lines.append(f"Clades: shared={len(shared)}, only_found={len(only_f)}, only_reference={len(only_r)}")
    if only_r:
        lines.append("Clades only in reference (sample up to 12):")
        for c in sorted(only_r, key=len)[:12]:
            lines.append(f"  {sorted(c)}")
    if only_f:
        lines.append("Clades only in found (sample up to 12):")
        for c in sorted(only_f, key=len)[:12]:
            lines.append(f"  {sorted(c)}")
    lines.append("")

    eng_f = _TripleDPEngine(found)
    eng_r = _TripleDPEngine(reference)
    trip_list = [t for t in triplets if any(rho[t][i] > 0.0 for i in range(3))]
    diffs: list[tuple[tuple[str, str, str], float]] = []
    for trip in trip_list:
        x, y, z = trip
        r_xy, r_xz, r_yz = rho[trip]
        pf_xy = eng_f.calculate_triple_probability((x, y, z))
        pf_xz = eng_f.calculate_triple_probability((x, z, y))
        pf_yz = max(0.0, 1.0 - pf_xy - pf_xz)
        pr_xy = eng_r.calculate_triple_probability((x, y, z))
        pr_xz = eng_r.calculate_triple_probability((x, z, y))
        pr_yz = max(0.0, 1.0 - pr_xy - pr_xz)
        ll_f = ll_r = 0.0
        for rr, a, b in [(r_xy, pf_xy, pr_xy), (r_xz, pf_xz, pr_xz), (r_yz, pf_yz, pr_yz)]:
            if rr > 0:
                ll_f += rr * math.log(max(a, 1e-300))
                ll_r += rr * math.log(max(b, 1e-300))
        diffs.append((trip, ll_r - ll_f))
    diffs.sort(key=lambda x: x[1], reverse=True)
    lines.append(f"Top {top_k} triplets where reference beats found (largest LL gap):")
    lines.append(f"{'triplet':>28} {'gap':>10}")
    for trip, d in diffs[:top_k]:
        lines.append(f"{','.join(trip):>28} {d:>10.2f}")
    lines.append("=== end comparison ===")
    return "\n".join(lines)


def save_mpl_network_newick(net: Network, path: str | os.PathLike[str]) -> None:
    """Serialize ``net`` as Newick to ``path`` (UTF-8, newline-terminated).

    Parent directories are created if they don't exist.

    Args:
        net: Network to serialise.
        path: Output path (str or ``os.PathLike``).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(net.newick() + "\n", encoding="utf-8")


# ======================================================================
# (C) Rho precomputation
#
# "rho" = triplet frequency statistic derived from the gene-tree set.
# For each species triplet (X, Y, Z), we walk every gene tree that
# covers all three taxa, determine the induced rooted topology via
# LCA depths, and accumulate a weight into one of the three
# resolution buckets (XY|Z, XZ|Y, YZ|X).  Each tree contributes a
# unit of mass (split 1/3 across ties if the triplet is a soft
# polytomy).  The resulting rho table is constant for the duration
# of a search, so it's computed once up front.
# ======================================================================

class _GeneTreeLCAIndex:
    """Precomputed LCA index for O(1) triplet topology queries on a gene tree.

    Construction is O(L^2) where L is the number of relevant leaves
    (BFS to build depth + parent maps, then pairwise LCA via parent
    climbing).  After construction, each triplet topology query is
    three dict lookups plus a few comparisons -- no BFS per query.

    For a dataset with T trees and K triplets, naive re-walking costs
    O(T * K * L).  Indexing trades a one-time O(T * L^2) construction
    for O(T * K) queries, which is a big win when K >> L.

    Attributes:
        _pair_depth: ``dict[(label_a, label_b)] -> lca_depth``, stored
            symmetrically in both orderings.
        _leaf_labels: Relevant leaves actually present in this tree
            (the tree may be missing taxa compared to the full set).
    """

    __slots__ = ('_pair_depth', '_leaf_labels')

    def __init__(self, tree: Network, relevant_labels: frozenset[str]) -> None:
        """Index ``tree`` for fast triplet queries over ``relevant_labels``.

        Args:
            tree: A gene tree.
            relevant_labels: Frozen set of allele labels that will
                actually be queried (typically the union of alleles
                across all species in the mapping).  Leaves whose
                labels fall outside this set are skipped, so a tree
                missing some taxa still indexes cleanly.
        """
        root = tree.root()
        depth: dict[Node, int] = {}
        parent: dict[Node, Node | None] = {}
        label_to_node: dict[str, Node] = {}

        # Single BFS to build depth + parent maps.
        q: deque[tuple[Node, Node | None, int]] = deque()
        q.append((root, None, 0))
        while q:
            node, par, d = q.popleft()
            depth[node] = d
            parent[node] = par
            for child in tree.get_children(node):
                q.append((child, node, d + 1))

        for leaf in tree.get_leaves():
            if leaf.label in relevant_labels:
                label_to_node[leaf.label] = leaf

        self._leaf_labels: frozenset[str] = frozenset(label_to_node)

        # Precompute pairwise LCA depths for all relevant leaves.
        sorted_labels = sorted(label_to_node)
        pair_depth: dict[tuple[str, str], int] = {}
        for i in range(len(sorted_labels)):
            a_label = sorted_labels[i]
            na_orig = label_to_node[a_label]
            da_orig = depth[na_orig]
            for j in range(i + 1, len(sorted_labels)):
                b_label = sorted_labels[j]
                na, nb = na_orig, label_to_node[b_label]
                da, db = da_orig, depth[nb]
                while da > db:
                    na = parent[na]
                    da -= 1
                while db > da:
                    nb = parent[nb]
                    db -= 1
                while na is not nb:
                    na = parent[na]
                    nb = parent[nb]
                    da -= 1
                pair_depth[(a_label, b_label)] = da
                pair_depth[(b_label, a_label)] = da

        self._pair_depth = pair_depth

    def induced_triple_fast(self, x: str, y: str, z: str) -> str:
        """Classify the induced topology of ``{x, y, z}`` via cached LCA depths.

        The deepest pairwise LCA names the two taxa that are siblings
        in the induced triplet; the third is the outgroup.  Ties in
        all three pair depths indicate a soft polytomy.

        Args:
            x: Taxon label.
            y: Taxon label.
            z: Taxon label.

        Returns:
            One of ``"xy|z"``, ``"xz|y"``, ``"yz|x"`` (rooted resolution
            strings) or ``"star"`` (polytomy).
        """
        pd = self._pair_depth
        d_xy = pd[(x, y)]
        d_xz = pd[(x, z)]
        d_yz = pd[(y, z)]

        if d_xy == d_xz == d_yz:
            return "star"
        max_d = d_xy
        if d_xz > max_d:
            max_d = d_xz
        if d_yz > max_d:
            max_d = d_yz
        if d_xy == max_d:
            return "xy|z"
        if d_xz == max_d:
            return "xz|y"
        return "yz|x"


def _process_tree_batch(
    batch: list[Network],
    relevant: frozenset[str],
    mapping: dict[str, list[str]],
    triplets: list[tuple[str, str, str]],
    identity_mapping: bool,
) -> dict[tuple[str, str, str], list[float]]:
    """Accumulate partial rho counts over a batch of gene trees.

    Separated from :func:`_compute_all_rhos_fast` so it can be
    dispatched to a ``concurrent.futures`` worker without closing
    over non-picklable state.

    Two code paths are kept for performance:

      * ``identity_mapping=True``: each species maps to exactly one
        allele (no within-species ambiguity), and triplet lookup is a
        straight dict read.  This is the common case on empirical
        datasets that haven't been resampled.

      * ``identity_mapping=False``: species can map to multiple
        alleles, so each species triplet expands to the Cartesian
        product of allele triplets.  Counts are weighted by
        ``1 / (|ax| * |ay| * |az|)`` to give each species triplet
        unit mass regardless of allele count.

    Args:
        batch: Gene trees to process.
        relevant: Flattened set of allele labels used anywhere in the
            mapping (for :class:`_GeneTreeLCAIndex` filtering).
        mapping: Species -> list of allele labels.
        triplets: Canonical species triplets to accumulate over.
        identity_mapping: True iff all species map to a single allele.

    Returns:
        ``dict`` mapping each triplet to a 3-element list
        ``[count_XY|Z, count_XZ|Y, count_YZ|X]``.  The caller merges
        partials across batches.
    """
    partial: dict[tuple[str, str, str], list[float]] = {
        t: [0.0, 0.0, 0.0] for t in triplets
    }

    if identity_mapping:
        _ONE_THIRD = 1.0 / 3.0
        for tree in batch:
            idx = _GeneTreeLCAIndex(tree, relevant)
            present = idx._leaf_labels
            pd = idx._pair_depth

            for triplet in triplets:
                X, Y, Z = triplet
                if X not in present or Y not in present or Z not in present:
                    continue

                d_xy = pd[(X, Y)]
                d_xz = pd[(X, Z)]
                d_yz = pd[(Y, Z)]

                acc = partial[triplet]
                if d_xy == d_xz == d_yz:
                    acc[0] += _ONE_THIRD
                    acc[1] += _ONE_THIRD
                    acc[2] += _ONE_THIRD
                elif d_xy >= d_xz and d_xy >= d_yz:
                    acc[0] += 1.0
                elif d_xz >= d_yz:
                    acc[1] += 1.0
                else:
                    acc[2] += 1.0
    else:
        for tree in batch:
            idx = _GeneTreeLCAIndex(tree, relevant)
            present = idx._leaf_labels

            for triplet in triplets:
                X, Y, Z = triplet
                ax = [a for a in mapping[X] if a in present]
                ay = [a for a in mapping[Y] if a in present]
                az = [a for a in mapping[Z] if a in present]

                denom = len(ax) * len(ay) * len(az)
                if denom == 0:
                    continue

                cnt_xy, cnt_xz, cnt_yz = 0.0, 0.0, 0.0
                for xi in ax:
                    for yj in ay:
                        for zk in az:
                            topo = idx.induced_triple_fast(xi, yj, zk)
                            if topo == "xy|z":
                                cnt_xy += 1.0
                            elif topo == "xz|y":
                                cnt_xz += 1.0
                            elif topo == "yz|x":
                                cnt_yz += 1.0
                            else:
                                cnt_xy += 1.0 / 3.0
                                cnt_xz += 1.0 / 3.0
                                cnt_yz += 1.0 / 3.0

                inv = 1.0 / denom
                acc = partial[triplet]
                acc[0] += cnt_xy * inv
                acc[1] += cnt_xz * inv
                acc[2] += cnt_yz * inv

    return partial


def _compute_all_rhos_fast(
    gene_trees: GeneTrees,
    mapping: dict[str, list[str]],
    triplets: list[tuple[str, str, str]],
    n_workers: int = 1,
) -> dict[tuple[str, str, str], tuple[float, float, float]]:
    """Compute the full rho table in one pass over ``gene_trees``.

    Each gene tree is indexed exactly once (pairwise LCA depths of the
    relevant leaves) and the index is reused for all triplets -- a large
    win when the triplet count (cubic in species count) dominates the
    tree count.  The work runs in the compiled core; ``n_workers > 1``
    splits the trees across native threads.

    Args:
        gene_trees: Collection of gene trees.
        mapping: Species -> list of allele labels.
        triplets: Canonical species triplets.
        n_workers: Thread count for the parallel path.  ``1`` (default)
            runs single-threaded.

    Returns:
        ``dict`` mapping each triplet to a tuple
        ``(rho_XY|Z, rho_XZ|Y, rho_YZ|X)``.
    """
    pairs = [(str(sp), [str(a) for a in alleles]) for sp, alleles in mapping.items()]
    trips = [tuple(t) for t in triplets]
    rho = _compute_rhos(list(gene_trees.trees), pairs, trips, int(n_workers))
    return {t: tuple(r) for t, r in zip(triplets, rho)}


# ======================================================================
# (D) Subnetwork probability DP engine
#
# Computes P(resolution | Psi, gamma) for a single rooted triplet
# against a fixed species network Psi.  The algorithm mirrors
# PhyloNet's pseudo-likelihood code path, which itself follows Yu &
# Nakhleh (2015).  The DP itself lives in the compiled core
# (``phynetpy._core._MplTopology``); this class is the Python handle.
#
# State legend (triplet lineages labelled 1, 2, 3):
#    0  nothing                             1  lineage 1 only
#    2  lineage 2 only                      3  lineage 3 only
#    4  lineages 1+2                        5  lineages 1+3
#    6  lineages 2+3                        7  lineages 1+2+3
#    8, 9, 10  intermediate states used when 3 lineages have partially
#              coalesced; differ in how many common ancestors are still
#              distinct.  State 10 (single lineage from triplet) is a
#              terminal articulation state.
# ======================================================================

class _TripleDPEngine:
    """Dynamic-programming engine for 3-taxon triplet probabilities.

    Each instance is bound to one fixed species network ``Psi``.
    Construction extracts the topology (leaf-to-root order, adjacency,
    articulation nodes, branch lengths and inheritance probabilities)
    into the compiled engine, so the per-triplet DP never touches
    :class:`Network` again.

    Attributes:
        network: The bound species network.
        net_node_num: Number of reticulation nodes.
        articulation_nodes: Tree nodes whose removal disconnects leaves
            from the root (plus leaves and the root).
        lowest_articulation_nodes: Articulation nodes at which the DP
            collapses compatible configurations.
    """

    # The transition tables, kept for reference; the compiled DP holds
    # its own copies.
    _MERGING_MAP = {
        (1, 2): 4,
        (1, 3): 5,
        (1, 6): 7,
        (2, 3): 6,
        (2, 5): 7,
        (3, 4): 7,
        (3, 8): 9,
    }
    _SPLITTING_MAP = {
        0: [(0, 0)],
        1: [(0, 1)],
        2: [(0, 2)],
        3: [(0, 3)],
        4: [(0, 4), (1, 2)],
        5: [(0, 5), (1, 3)],
        6: [(0, 6), (2, 3)],
        7: [(0, 7), (1, 6), (2, 5), (3, 4)],
        8: [(0, 8)],
        9: [(0, 9), (3, 8)],
        10: [(0, 10)],
    }
    _COALESCING_MAP = {
        0: [(0, (0, 0))],
        1: [(1, (1, 1))],
        2: [(2, (1, 1))],
        3: [(3, (1, 1))],
        4: [(4, (2, 2)), (8, (2, 1))],
        5: [(5, (2, 2))],
        6: [(6, (2, 2))],
        7: [(7, (3, 3)), (9, (3, 2)), (10, (3, 1))],
        8: [(8, (1, 1))],
        9: [(9, (2, 2)), (10, (2, 1))],
        10: [(10, (1, 1))],
    }

    def __init__(self, network: Network) -> None:
        """Bind the engine to a fixed species network.

        Args:
            network: Species network to be scored.  Must already
                have valid branch lengths and gamma values on its
                edges -- the engine does not touch :class:`Network`
                for any of that data after ``__init__`` returns.
        """
        self.network = network
        self._rs = _MplTopology(network)
        self.net_node_num: int = self._rs.net_node_num
        nodes = self._rs.nodes()
        self._topo_leaf_to_root: list[Node] = nodes
        self.articulation_nodes: set[Node] = {
            nodes[i] for i in self._rs.articulation_indices()
        }
        self.lowest_articulation_nodes: set[Node] = {
            nodes[i] for i in self._rs.lowest_articulation_indices()
        }

    def refresh_params(self) -> None:
        """Re-read branch lengths / gammas from the bound network in place.

        Fast path when only continuous parameters changed: the cached
        adjacency / articulation structure is still valid, so only the
        per-edge values are refreshed.
        """
        self._rs.refresh_params(self.network)

    @staticmethod
    def _fact(start: int, end: int) -> float:
        """Return the product ``start * (start+1) * ... * end`` (1.0 if empty)."""
        result = 1.0
        for i in range(start, end + 1):
            result *= i
        return result

    @staticmethod
    def _gij_raw(length: float, i: int, j: int) -> float:
        """Closed-form coalescent transition probability g_{ij}(t).

        ``length is None`` or ``-1`` is an infinite branch (1.0 iff
        ``j == 1``); ``length == 0`` allows no coalescence (1.0 iff
        ``i == j``); ``i == 0`` is a no-op.
        """
        if length is None or length == -1:
            return 1.0 if j == 1 else 0.0
        if length == 0:
            return 1.0 if i == j else 0.0
        if i == 0:
            return 1.0
        _fact = _TripleDPEngine._fact
        result = 0.0
        for k in range(j, i + 1):
            temp = (
                math.exp(0.5 * k * (1.0 - k) * length)
                * (2.0 * k - 1.0)
                * ((-1.0) ** (k - j))
                * _fact(j, j + k - 2)
                * _fact(i - k + 1, i)
            )
            denom = _fact(1, j) * _fact(1, k - j) * _fact(i, i + k - 1)
            result += temp / denom
        return result

    def calculate_triple_probability(self, triple: tuple[str, str, str]) -> float:
        """Compute ``P(xy|z)`` for one ordered triplet against this network.

        The triplet is interpreted as "x and y are siblings, z is the
        outgroup".  To get ``P(xz|y)`` or ``P(yz|x)`` call this method
        again with the taxa reordered.  A label that is not a leaf of
        the network contributes no lineage, so the probability is 0.

        Args:
            triple: Ordered species labels ``(x, y, z)``.

        Returns:
            ``P(xy|z)`` under the species network.
        """
        x, y, z = triple
        return self._rs.triple_probability(str(x), str(y), str(z))


# ======================================================================
# (E) Compiled bridge
#
# The names below predate the Rust core (they used to shuttle numpy arrays
# into a Cython kernel).  They are kept so the scorer and DIMPLE's subset
# scorer keep working unchanged; each is now a thin call into the engine.
# ======================================================================

_CY_MAX_CHILDREN = 4
_CY_MAX_PARENTS = 2


def _extract_topology_for_cython(engine: _TripleDPEngine) -> "_MplTopology":
    """The compiled topology bound to ``engine`` (cached on the engine)."""
    return engine._rs


def _build_cython_triplet_index(
    topo: "_MplTopology",
    triplets: list[tuple[str, str, str]],
    rho: dict[tuple[str, str, str], tuple[float, float, float]],
) -> tuple["_TripletIndex", None]:
    """Translate species triplets into leaf indices for the compiled DP.

    Triplets whose taxa are not all present in the network are silently
    dropped (mirrors the Python path).  Depends only on the topology, so
    it is stable across branch-length / gamma edits.

    Returns:
        ``(index, None)``: the compiled index and a placeholder for the
        former rho list, which now lives inside the index.
    """
    return topo.build_triplet_index([tuple(t) for t in triplets], rho), None


def _refresh_cython_params(engine: _TripleDPEngine, topo: "_MplTopology") -> None:
    """Re-read branch lengths / gammas into the compiled topology."""
    topo.refresh_params(engine.network)


def _cy_score_from_topo(
    topo: "_MplTopology",
    trip_idx: "_TripletIndex",
    rho_vals=None,
) -> float:
    """Run the compiled DP over a prebuilt triplet index."""
    return topo.score(trip_idx)


def _score_with_cython(
    engine: _TripleDPEngine,
    triplets: list[tuple[str, str, str]],
    rho: dict[tuple[str, str, str], tuple[float, float, float]],
) -> float:
    """Score a network against all triplets via the compiled DP engine.

    Triplets whose taxa are not all present in the engine's network
    are silently dropped -- this mirrors the Python path's behaviour
    and keeps the pruned-taxa case working.

    Args:
        engine: DP engine already bound to the target network.
        triplets: Canonical species triplets to evaluate.
        rho: Gene-tree triplet frequency table (constant for the run).

    Returns:
        Log pseudo-likelihood across all evaluable triplets.
    """
    topo = engine._rs
    index = topo.build_triplet_index([tuple(t) for t in triplets], rho)
    return topo.score(index)


# ======================================================================
# (F) Public triplet API
#
# These two functions are the entry points for users who want to
# bring-their-own-search-driver: enumerate triplets, precompute rho,
# then call score_species_network_triplets against any number of
# candidate species networks.  :class:`MPL` wraps both for the
# common case.
# ======================================================================

def compute_gene_tree_triplets(
    gene_trees: GeneTrees,
    mapping: dict[str, list[str]],
    species_labels: Optional[list[str]] = None,
) -> GeneTreeTripletResult:
    """Enumerate triplets and precompute rho from ``gene_trees``.

    Gene-tree statistics are constant across a network search, so
    this is called exactly once up front.

    Args:
        gene_trees: Collection of gene trees.
        mapping: Species -> list of allele labels.
        species_labels: Optional explicit species list; defaults to
            ``sorted(mapping.keys())``.  Provide this when scoring
            against a reduced taxon set (e.g. sub-sample experiments).

    Returns:
        :class:`GeneTreeTripletResult` carrying triplets and rho.

    Raises:
        ValueError: If fewer than three species labels are available
            (triplets require at least three taxa).
    """
    if species_labels is None:
        species_labels = sorted(mapping.keys())
    else:
        species_labels = sorted(species_labels)

    if len(species_labels) < 3:
        raise ValueError("Need at least 3 species labels to form triplets")

    triplets = list(combinations(species_labels, 3))
    rho_by_triplet = _compute_all_rhos_fast(gene_trees, mapping, triplets)

    return GeneTreeTripletResult(triplets=triplets, rho_by_triplet=rho_by_triplet)


def score_species_network_triplets(
    species_net: Network,
    gene_triplet_result: GeneTreeTripletResult,
) -> SpeciesNetworkTripletResult:
    """Score a species network against precomputed triplet frequencies.

    Builds a fresh :class:`_TripleDPEngine` bound to ``species_net``
    and walks every triplet, computing both predicted probabilities
    and the MPL log-sum.

    Args:
        species_net: Species network to evaluate.
        gene_triplet_result: Output of
            :func:`compute_gene_tree_triplets` (same gene-tree set).

    Returns:
        :class:`SpeciesNetworkTripletResult` with per-triplet
        probabilities and the final log pseudo-likelihood.

    Raises:
        ValueError: If a network triplet isn't present in
            ``gene_triplet_result`` (which indicates the rho table was
            built over a different species set than the network's
            leaves).
    """
    # Use network leaves as the scoring universe.
    triplets = list(combinations(sorted(n.label for n in species_net.get_leaves()), 3))
    triple_engine = _TripleDPEngine(species_net)
    probs_by_triplet: dict[tuple[str, str, str], tuple[float, float, float]] = {}

    total = 0.0
    for triplet in triplets:
        # For triplet (x, y, z), engine computes P(xy|z) and P(xz|y).
        # P(yz|x) is derived by normalization.
        x, y, z = triplet
        p_xy = triple_engine.calculate_triple_probability((x, y, z))
        p_xz = triple_engine.calculate_triple_probability((x, z, y))
        probs = (p_xy, p_xz, max(1.0 - p_xy - p_xz, 0.0))
        probs_by_triplet[triplet] = probs

        rho = gene_triplet_result.rho_by_triplet.get(triplet)
        if rho is None:
            raise ValueError(
                f"Missing gene-tree triplet frequencies for species triplet {triplet}"
            )

        # MPL contribution for one species triplet.
        for rho_i, p_i in zip(rho, probs):
            if rho_i > 0.0:
                total += rho_i * (math.log(p_i) if p_i > 0.0 else _LOG_FLOOR)

    return SpeciesNetworkTripletResult(
        triplets=triplets,
        probs_by_triplet=probs_by_triplet,
        log_pseudo_likelihood=total,
    )


# ======================================================================
# (G) Scorer + adaptive proposal kernel
#
# :class:`MPLScorer` is the objective function (a thin callable that
# rebuilds the DP engine per call).  :class:`MPLKernel` is the
# proposal distribution driving the search (phased move selection +
# adaptive weights + adaptive sigmas for continuous proposals).
# :class:`_AdaptiveConfig` is a private dataclass-style container for
# the empirically-chosen tuning scalars that the kernel reads.
# ======================================================================


class MPLScorer:
    """Callable likelihood evaluator for use with :class:`Model`.

    Registered via ``model.set_likelihood_calculator(scorer)``.  On
    each call, rebuilds a fresh :class:`_TripleDPEngine` for the
    current network (topology moves invalidate the previous engine's
    cached articulation sets) and returns the log pseudo-likelihood.

    Gene-tree rho values never change during a search, so they are
    stored once on the instance.  Triplets whose gene-tree rho is
    identically zero contribute nothing to the score and are pruned
    at construction time to avoid redundant DP calls.

    Attributes:
        _rho: Full triplet -> rho table (kept even for pruned
            triplets in case the scorer is reused against a network
            with a different triplet mask).
        _triplets: Active (non-zero-rho) triplet list used by
            ``__call__``.
    """

    def __init__(
        self,
        rho: dict[tuple[str, str, str], tuple[float, float, float]],
        triplets: list[tuple[str, str, str]],
    ) -> None:
        """Initialise the scorer from precomputed gene-tree statistics.

        Args:
            rho: Mapping ``triplet -> (rho_XY|Z, rho_XZ|Y, rho_YZ|X)``.
            triplets: Canonical species triplets to iterate over.
        """
        self._rho = rho
        self._triplets = [
            t for t in triplets if any(rho[t][i] > 0.0 for i in range(3))
        ]

        # ---- Lever 3: incremental engine cache --------------------------
        # When consecutive scoring calls touch the *same* network object
        # and only its branch lengths / gammas changed (the dirty-set
        # protocol reports a non-``None`` touched-node set), we reuse the
        # cached engine + extracted Cython topology and refresh only the
        # parameter arrays, skipping the (dominant) topology rebuild.
        self._engine: _TripleDPEngine | None = None
        self._engine_net = None              # network object identity guard
        self._cy_topo: dict | None = None
        self._cy_trip_idx = None
        self._cy_rho_vals = None

    def _needs_rebuild(self, model: Model) -> bool:
        """Decide whether the cached engine must be rebuilt for ``model``.

        Rebuild when there is no cache yet, the network object changed
        (a different proposal / committed copy), or the move reported a
        full ("topology changed") invalidation via ``_dirty_nodes is
        None``.  A non-``None`` dirty set means parameters-only -> reuse.
        """
        if self._engine is None or self._engine_net is not model.network:
            return True
        dirty = getattr(model, "_dirty_nodes", None)
        return dirty is None

    def __call__(self, model: Model) -> float:
        """Return log pseudo-likelihood of ``model.network``.

        Uses the Cython DP path when available, otherwise falls back
        to the Python DP.  Both compute the same score.  Reuses a cached
        engine across calls on the same network when only continuous
        parameters changed (lever 3).

        Args:
            model: :class:`Model` whose ``network`` attribute is the
                current species network.

        Returns:
            Log pseudo-likelihood (higher is better; always <= 0).
        """
        net = model.network
        if self._needs_rebuild(model):
            self._engine = _TripleDPEngine(net)
            self._engine_net = net
            if _HAS_CYTHON_MPL:
                self._cy_topo = _extract_topology_for_cython(self._engine)
                self._cy_trip_idx, self._cy_rho_vals = (
                    _build_cython_triplet_index(
                        self._cy_topo, self._triplets, self._rho,
                    )
                )
        else:
            # Parameters-only change: refresh edge values on the cached
            # engine / topology rather than rebuilding the topology.
            if _HAS_CYTHON_MPL:
                _refresh_cython_params(self._engine, self._cy_topo)
            else:
                self._engine.refresh_params()

        # The scorer has consumed the dirty set; reset it so a subsequent
        # parameters-only call is recognised as such.
        clear = getattr(model, "clear_dirty_nodes", None)
        if callable(clear):
            clear()

        if _HAS_CYTHON_MPL:
            return _cy_score_from_topo(
                self._cy_topo, self._cy_trip_idx, self._cy_rho_vals,
            )

        engine = self._engine
        total = 0.0
        for triplet in self._triplets:
            x, y, z = triplet
            p_xy = engine.calculate_triple_probability((x, y, z))
            p_xz = engine.calculate_triple_probability((x, z, y))
            probs = (p_xy, p_xz, max(1.0 - p_xy - p_xz, 0.0))
            rho = self._rho[triplet]
            for rho_i, p_i in zip(rho, probs):
                if rho_i > 0.0:
                    total += rho_i * (math.log(p_i) if p_i > 0.0 else _LOG_FLOOR)
        return total


class _AdaptiveConfig:
    """Internal adaptive-tuning constants for :class:`MPLKernel`.

    Grouped here so the kernel's class-level surface stays focused on
    its structural pieces (phase order, base-weight maps) rather than
    buried under fifteen tuning scalars.  None of these values are
    user-facing knobs; they're empirically-chosen defaults for the
    adaptive machinery.  Moving them here is a pure refactor -- the
    numbers are identical to the previous ``MPLKernel`` class
    constants.  When you want to tune the adaptive layer, this is the
    single place to look.
    """

    # ── Robbins-Monro continuous-sigma tuning ──────────────────────
    # Applied to ChangeNodeHeight and ChangeInheritanceProb.  The
    # scheme nudges ``log(sigma)`` toward the target acceptance rate;
    # step size shrinks as ``1 / (n + shift)**exp`` so later updates
    # make smaller corrections.
    SIGMA_TARGET_ACCEPT = 0.35           # Roberts-Rosenthal heuristic
    SIGMA_TUNE_DELAY = 20                # warm-up observations per move
    SIGMA_TUNE_EXPONENT = 0.6            # step decay rate (1/n^exp)
    SIGMA_TUNE_DENOM_SHIFT = 50          # dampen very early updates

    # ── ChangeNodeHeight sigma_frac bounds ─────────────────────────
    # sigma_frac is a *fraction of the feasible half-range*, so these
    # bounds are dimensionless.  0.4 gives a tighter proposal than the
    # prior uniform draw (which was effectively ~0.58).
    NH_SIGMA_INIT = 0.4
    NH_SIGMA_MIN = 0.02
    NH_SIGMA_MAX = 1.5

    # ── ChangeInheritanceProb sigma bounds ─────────────────────────
    # sigma acts directly on the gamma scale (inheritance probability).
    # Bounds keep proposals from degenerating to delta spikes or flat
    # uniforms.
    CIP_SIGMA_INIT = 0.1
    CIP_SIGMA_MIN = 0.005
    CIP_SIGMA_MAX = 0.4

    # ── SPR adaptive regraft radius ────────────────────────────────
    # ``SPR`` weights each candidate regraft edge by
    # ``1 / d**distance_decay``.  MAX keeps the move strongly local
    # (exploit current basin); MIN makes it near-flat (broad hops
    # across the network).  Decay interpolates from MAX -> MIN as
    # ``stagnation / phase_patience`` climbs toward 1.
    SPR_DECAY_MAX = 2.5
    SPR_DECAY_MIN = 0.5

    # ── Post-reheat wide-SPR window ────────────────────────────────
    # After each reheat notification, the next N SPR proposals are
    # forced to ``SPR_DECAY_MIN`` regardless of phase switches or
    # stagnation resets.  Without this, the phase switch triggered
    # by the reheat hook zeroes ``_stagnation`` before a single
    # broad SPR fires, cancelling the reheat's exploration intent
    # entirely.  Sized to roughly cover one post-reheat SA cooling
    # window (~250 iters at ~8% SPR density).
    SPR_BROAD_MODE_PROPOSALS = 20

    # ── efficiency-aware weight scaler blend ───────────────────────
    # Composite score = ACC_WEIGHT * acc_rate + EFF_WEIGHT * efficiency
    # where efficiency = imp_rate / max(acc_rate, EFF_EPSILON).  The
    # 0.4/0.6 split weights exploitation slightly over exploration --
    # SA itself handles exploration via temperature-driven uphill
    # accepts, so the adaptive layer focuses on "which move is
    # actually moving the needle".
    SCALER_ACC_WEIGHT = 0.4
    SCALER_EFF_WEIGHT = 0.6
    SCALER_EFF_EPSILON = 0.01


class MPLKernel(ProposalKernel):
    """Phase-aware proposal kernel for MPL network search.

    Cycles between two complementary search phases:

      **TOPOLOGY** -- Explores tree space via SPR and branch-length
      adjustments (ChangeNodeHeight), with ChangeInheritanceProb
      mixed in for gamma tuning.

      **RETICULATION** -- Searches for hybridization events via
      Add/Remove/Flip reticulation and source/dest moves, with
      ChangeInheritanceProb and ChangeNodeHeight for local
      refinement after reticulation changes.

    Both phases keep a small bleed of the complementary phase's moves
    active (~10-15% weight) so the chain stays ergodic across phase
    flips and the adaptive scaler always has fresh observations for
    every class.

    Phase transitions fire after ``phase_patience`` consecutive
    proposals without accepted improvement.  Within each phase,
    selection weights adapt from phase-specific base distributions
    using a sliding window of recent acceptance rates.

    When explicit *weights* are supplied, the kernel falls back to
    flat (non-phased) mode for backward compatibility.
    """

    TOPOLOGY = "topology"
    RETICULATION = "reticulation"

    # Phase base weights.  Each phase keeps a low-rate bleed of the
    # complementary phase's moves so the chain is never strictly
    # topology-only or strictly reticulation-only.
    _TOPOLOGY_BASE: dict[type, float] = {
        SPR: 5.0,
        ChangeNodeHeight: 3.0,
        ChangeInheritanceProb: 1.0,
        # Bleed: ~10% aggregate weight on retic moves keeps Add/Remove
        # warm and lets small retic corrections happen mid-topology-phase.
        AddReticulation: 0.30,
        RemoveReticulation: 0.20,
        RelocateReticulation: 0.25,
        ChangeReticSource: 0.10,
        ChangeReticDest: 0.10,
        # FlipReticulation is deliberately weighted tiny.  Empirically
        # (see P4 50k run) it has a ~0.5% strict-improvement rate --
        # ~25x lower than other retic moves -- while its ~27% accept
        # rate (mostly uphill SA accepts) trips the adaptive scaler
        # into over-proposing it.  Keep it alive (not zero) so the
        # reticulation-orientation corner case is still reachable, but
        # don't let it consume a large slice of the proposal budget.
        FlipReticulation: 0.03,
    }

    _RETICULATION_BASE: dict[type, float] = {
        AddReticulation: 3.0,
        RemoveReticulation: 2.0,
        # FlipReticulation base weight: see TOPOLOGY_BASE comment.
        # Previously 1.5 (equal to source/dest moves); dropped to 0.5
        # based on observed improvement rate ~25x lower than peers.
        FlipReticulation: 0.5,
        ChangeReticSource: 1.5,
        ChangeReticDest: 1.5,
        RelocateReticulation: 2.5,
        ChangeInheritanceProb: 1.5,
        ChangeNodeHeight: 1.0,
        # Bleed: keep SPR alive in retic phase so local topology
        # adjustments can accompany reticulation placement.
        SPR: 1.5,
    }

    _PHASE_ORDER = [TOPOLOGY, RETICULATION]

    # All empirically-chosen adaptive-tuning scalars live on
    # ``_AdaptiveConfig`` (defined below this class) so the kernel's
    # visible surface stays focused on structural design decisions
    # (phases, base weights) rather than buried under a dozen magic
    # numbers.  Instance attributes below pull initial values from
    # that config at construction time.

    def __init__(self,
                 move_types: list[type[Move]] | None = None,
                 weights: list[float] | None = None,
                 max_reticulations: int | None = None,
                 max_level: int | None = None,
                 adaptive: bool = True,
                 window_size: int = 30,
                 min_weight: float = 0.05,
                 phase_patience: int = 150,
                 warmup: int = 8,
                 stagnation_reset_delta: float = 1.0,
                 rng: np.random.Generator | None = None) -> None:
        """
        Args:
            move_types: Move classes available to the kernel.  When
                ``None`` the full default set is used.
            weights: Fixed per-move selection weights.  Supplying this
                disables phased cycling and adaptive tuning.
            max_reticulations: Cap on reticulation nodes.  When the
                current network is at the cap, ``AddReticulation`` is
                dropped from the active move set so we don't waste
                proposals on guaranteed no-ops.
            max_level: Cap on network level (``max_lvl`` search flag).
                Passed to every level-raising move (Add / Relocate /
                ChangeReticSource / ChangeReticDest / Flip) so they
                self-reject proposals that would exceed the cap.
                ``None`` disables the per-move check (the accept-path
                level guard remains authoritative).
            adaptive: Enable within-phase adaptive weight scaling.
            window_size: Sliding-window length for acceptance stats.
            min_weight: Minimum scale factor (fraction of base weight)
                to prevent any move from being fully starved.
            phase_patience: Consecutive non-improving proposals before
                the kernel switches to the next phase.  Default 150
                keeps phase flips roughly aligned with the SA
                ``steps_per_temp`` cadence.
            warmup: Minimum observations per move class before the
                adaptive scaling activates for that class.
            stagnation_reset_delta: Minimum strict-improvement
                magnitude (in log-PL units) that counts as "real
                progress" for stagnation-reset purposes.  Any
                ``delta > stagnation_reset_delta`` resets the
                stagnation counter; smaller wiggles are ignored.
                Default 1.0 filters out the O(0.01-0.1) tweaks that
                late-stage NH/CIP moves generate, so the counter
                actually climbs during genuine plateaus.  Setting to
                0.0 restores the prior "any strict improvement resets"
                behaviour.  Required for phase cycling and adaptive
                SPR radius to track true plateaus rather than
                micro-noise.
            rng: ``numpy.random.Generator`` used to sample the move
                class.  When ``None``, a fresh generator is created
                from OS entropy; callers that care about
                reproducibility (e.g. ``MPL.search``) should pass one
                seeded from the search-wide ``SeedSequence``.
        """
        super().__init__()
        self._max_retics: int | None = max_reticulations
        self._max_level: int | None = max_level
        self._all_moves: list[type[Move]] = move_types or [
            SPR,
            ChangeNodeHeight,
            ChangeInheritanceProb,
            AddReticulation,
            RemoveReticulation,
            FlipReticulation,
            ChangeReticSource,
            ChangeReticDest,
            RelocateReticulation,
        ]

        self._fixed_weights: list[float] | None = weights
        self._phased: bool = (weights is None)
        self._adaptive: bool = adaptive and (weights is None)

        self._phase: str = self.TOPOLOGY
        self._phase_patience: int = phase_patience
        self._stagnation: int = 0
        self._phase_switches: int = 0

        self._window_size: int = window_size
        self._min_weight: float = min_weight
        self._warmup: int = warmup
        self._stagnation_reset_delta: float = float(stagnation_reset_delta)

        # Diagnostics for phase/SPR plumbing -- filled in as the run
        # progresses; surfaced via ``format_stats`` at the end.
        self._reheat_signals: int = 0
        self._stagnation_peak: int = 0

        self.rng: np.random.Generator = (
            rng if rng is not None else np.random.default_rng()
        )

        from collections import deque
        self._history: dict[type[Move], deque] = {
            cls: deque(maxlen=window_size) for cls in self._all_moves
        }
        self._last_cls: type[Move] | None = None

        # Lifetime counters for end-of-run diagnostics.  ``proposed`` counts
        # every draw of a move class (regardless of validity).  ``accepted``
        # counts every draw that the SA/HC driver later committed to the
        # chain (strict improvement OR uphill-accepted).  ``improved``
        # counts strict-improvement commits (delta > 0 in maximisation).
        self._proposed: dict[type[Move], int] = {cls: 0 for cls in self._all_moves}
        self._accepted: dict[type[Move], int] = {cls: 0 for cls in self._all_moves}
        self._improved: dict[type[Move], int] = {cls: 0 for cls in self._all_moves}

        # Adaptive sigmas for the two continuous proposals.  They are
        # driven toward ``_SIGMA_TARGET_ACCEPT`` by a Robbins-Monro-style
        # update in ``report_outcome``.  ``_sigma_obs`` counts the
        # observations each move class has contributed so the step size
        # decays like 1/n^exponent.
        self._nh_sigma: float = _AdaptiveConfig.NH_SIGMA_INIT
        self._cip_sigma: float = _AdaptiveConfig.CIP_SIGMA_INIT
        self._sigma_obs: dict[type[Move], int] = {
            ChangeNodeHeight: 0,
            ChangeInheritanceProb: 0,
        }

        # SPR decay tracking.  ``_last`` holds the decay used on the most
        # recent SPR draw; ``_min_seen``/``_max_seen`` record the
        # realised range so ``format_stats`` can show how often the
        # kernel actually opened the SPR radius during the run.
        self._spr_decay_last: float = _AdaptiveConfig.SPR_DECAY_MAX
        self._spr_decay_min_seen: float = _AdaptiveConfig.SPR_DECAY_MAX
        self._spr_decay_max_seen: float = _AdaptiveConfig.SPR_DECAY_MAX

        # Post-reheat wide-SPR window.  ``on_reheat`` sets this to
        # ``SPR_BROAD_MODE_PROPOSALS``; each SPR draw decrements it.
        # While positive, ``_current_spr_decay`` returns ``SPR_DECAY_MIN``
        # regardless of the stagnation-derived decay.  This decouples
        # SPR radius adaptation from phase-switch stagnation resets:
        # even if the reheat triggers a phase switch (which zeroes
        # ``_stagnation``), the broad-mode counter still forces broad
        # regrafts for the next N SPR calls.  See ``_AdaptiveConfig``.
        self._spr_broad_remaining: int = 0
        self._spr_broad_activations: int = 0
        self._spr_broad_proposals: int = 0

    # ── phase helpers ───────────────────────────────────────────

    def _phase_base_map(self) -> dict[type, float]:
        """Base weight map for the current phase."""
        if self._phase == self.TOPOLOGY:
            return self._TOPOLOGY_BASE
        return self._RETICULATION_BASE

    def _at_retic_cap(self, network: Network | None) -> bool:
        """True when the current network already holds ``max_reticulations``."""
        if self._max_retics is None or network is None:
            return False
        return count_reticulations(network) >= self._max_retics

    def _active_moves_and_base(
        self, network: Network | None = None,
    ) -> tuple[list[type[Move]], list[float]]:
        """Active move classes and base weights for the current phase.

        When the caller supplies a ``network`` and we're at the
        reticulation cap, ``AddReticulation`` is omitted entirely so
        proposals aren't wasted on a guaranteed no-op.
        """
        base = self._phase_base_map()
        at_cap = self._at_retic_cap(network)
        moves: list[type[Move]] = []
        weights: list[float] = []
        for cls in self._all_moves:
            if cls not in base:
                continue
            if at_cap and cls is AddReticulation:
                continue
            moves.append(cls)
            weights.append(base[cls])
        return moves, weights

    def phase_switch_due(self) -> bool:
        """Whether the next :meth:`generate` will change phase.

        Hill climbing decides to stop before it asks for the next
        move, and the phase change happens inside that request. When
        both counters reach the same threshold, the stop wins and the
        reticulation phase is never sampled. The driver calls this so
        a switch that is already due still gets one proposal.

        Returns:
            bool: True when this kernel cycles phases and stagnation
            has reached ``phase_patience``.
        """
        return self._phased and self._stagnation >= self._phase_patience

    def _maybe_switch_phase(self) -> None:
        """Advance to the next phase when stagnation exceeds patience."""
        if self._stagnation < self._phase_patience:
            return
        idx = self._PHASE_ORDER.index(self._phase)
        for offset in range(1, len(self._PHASE_ORDER) + 1):
            candidate = self._PHASE_ORDER[
                (idx + offset) % len(self._PHASE_ORDER)
            ]
            base = (self._TOPOLOGY_BASE if candidate == self.TOPOLOGY
                    else self._RETICULATION_BASE)
            if any(cls in base for cls in self._all_moves):
                self._phase = candidate
                self._phase_switches += 1
                break
        self._stagnation = 0

    # ── adaptive SPR regraft radius ─────────────────────────────

    def _current_spr_decay(self) -> float:
        """SPR distance-decay for the current stagnation level.

        Returns a value in ``[SPR_DECAY_MIN, SPR_DECAY_MAX]``.  Normal
        path: decay drops linearly from MAX (just improved / fresh)
        toward MIN (fully stuck) as ``stagnation / phase_patience``
        climbs.  Post-reheat path: while ``_spr_broad_remaining > 0``
        (set by :meth:`on_reheat`), decay is forced to MIN so the
        next ``SPR_BROAD_MODE_PROPOSALS`` SPR draws get broad hops
        regardless of intervening phase switches.  This is the fix
        for the "reheat triggers phase switch, phase switch zeroes
        stagnation, next SPR sees stuckness=0, wide mode never fires"
        pathology seen in long runs.
        """
        if self._spr_broad_remaining > 0:
            return _AdaptiveConfig.SPR_DECAY_MIN
        if self._phase_patience <= 0:
            return _AdaptiveConfig.SPR_DECAY_MAX
        stuckness = min(1.0, self._stagnation / float(self._phase_patience))
        span = _AdaptiveConfig.SPR_DECAY_MAX - _AdaptiveConfig.SPR_DECAY_MIN
        return _AdaptiveConfig.SPR_DECAY_MAX - stuckness * span

    # ── adaptive sigma (continuous proposals) ───────────────────

    def _tune_sigma(self,
                    current: float,
                    n_obs: int,
                    accepted: bool,
                    sigma_min: float,
                    sigma_max: float) -> float:
        """One Robbins-Monro step toward the acceptance-rate target.

        ``log(sigma)`` drifts by ``step * (accept_indicator - target)``,
        where ``step`` shrinks as ``1/(n + shift)^exponent``.  This is
        the standard adaptive Metropolis update (Roberts & Rosenthal,
        2009): accept > target widens proposals, accept < target
        tightens them.  Because SA also cools over time, ``sigma``
        tends to contract alongside temperature -- which is exactly
        what we want in the exploitation phase.
        """
        if n_obs < _AdaptiveConfig.SIGMA_TUNE_DELAY:
            return current
        step = 1.0 / ((n_obs + _AdaptiveConfig.SIGMA_TUNE_DENOM_SHIFT)
                      ** _AdaptiveConfig.SIGMA_TUNE_EXPONENT)
        indicator = 1.0 if accepted else 0.0
        log_adjust = step * (indicator - _AdaptiveConfig.SIGMA_TARGET_ACCEPT)
        new_sigma = current * math.exp(log_adjust)
        return max(sigma_min, min(sigma_max, new_sigma))

    # ── adaptive tuning ─────────────────────────────────────────

    def _adapt_weights(self,
                       moves: list[type[Move]],
                       base_weights: list[float]) -> list[float]:
        """Scale base weights by an efficiency-aware accept/improve blend.

        A pure acceptance-rate scaler over-rewards moves that get
        accepted often via uphill SA accepts but rarely make strict
        progress (e.g. ``RelocateReticulation`` or ``FlipReticulation``
        at cold T).  Meanwhile it starves moves that accept rarely but
        whose accepts are almost always strict improvements (e.g.
        ``SPR`` at cold T, ``AddReticulation`` when below cap).  That
        is the opposite of what we want for an MPL *optimiser*.

        The formula below reads two signals from each move's rolling
        ``window_size`` history:

            acc_rate        = accepts / window                # explore credit
            imp_rate        = strict_improvements / window    # exploit credit
            efficiency      = imp_rate / max(acc_rate, 0.01)  # per-accept quality

        ``efficiency`` is the probability that, given the move
        accepted, the accept was a strict improvement.  It's capped at
        1.0 to keep the math bounded when a warmup-era window happens
        to have only improvements.

        The composite score

            composite = 0.4 * acc_rate + 0.6 * efficiency

        then drives the same ``floor .. 2x`` multiplicative scale the
        old formula used -- so the knob count is unchanged; only the
        signal feeding it is different.  The 0.4 / 0.6 split
        deliberately weights exploitation (progress) more heavily than
        exploration (acceptance): SA itself already handles exploration
        through its temperature-driven uphill accepts.  We want the
        adaptive layer focused on "which move is actually moving the
        needle right now".

        Moves with fewer than ``warmup`` observations keep their full
        base weight (avoids early-window noise swinging the scaler).
        """
        floor = self._min_weight
        weights: list[float] = []
        for cls, base_w in zip(moves, base_weights):
            hist = self._history[cls]
            if len(hist) < self._warmup:
                weights.append(base_w)
                continue
            n = len(hist)
            n_acc = sum(1 for accepted, _ in hist if accepted)
            n_imp = sum(1 for accepted, delta in hist
                        if accepted and delta > 0.0)
            acc_rate = n_acc / n
            imp_rate = n_imp / n
            efficiency = min(
                1.0,
                imp_rate / max(acc_rate, _AdaptiveConfig.SCALER_EFF_EPSILON),
            )
            composite = (
                _AdaptiveConfig.SCALER_ACC_WEIGHT * acc_rate
                + _AdaptiveConfig.SCALER_EFF_WEIGHT * efficiency
            )
            scale = floor + (2.0 - floor) * composite
            weights.append(base_w * scale)
        return weights

    # ── public interface ────────────────────────────────────────

    def generate(self, model: "Model | None" = None) -> Move:
        """Sample a move from the current phase's weighted distribution.

        Args:
            model: Optional current model.  Used only for cap-aware
                filtering (``AddReticulation`` is dropped when the
                network is already at ``max_reticulations``).  Callers
                from ``SimulatedAnnealing``/``HillClimbing`` pass the
                live model; the abstract base signature keeps the
                parameter optional for back-compat.
        """
        network = model.network if model is not None else None

        # --- Decide which moves are eligible and their weights -----
        if self._phased:
            self._maybe_switch_phase()
            moves, base_weights = self._active_moves_and_base(network)
            weights = (self._adapt_weights(moves, base_weights)
                       if self._adaptive else base_weights)
        else:
            moves = self._all_moves
            weights = self._fixed_weights

        if not moves:
            # Defensive fallback: with cap filtering active and a weird
            # phase base, we might have no eligible moves.  Fall back to
            # the full move list so the search keeps progressing.
            moves = list(self._all_moves)
            weights = [1.0] * len(moves)

        # --- Sample a move class from the effective distribution ---
        w = np.asarray(weights, dtype=float)
        total = float(w.sum())
        if not np.isfinite(total) or total <= 0.0:
            # Degenerate weights (all zero or NaN): fall back to
            # uniform sampling so the chain doesn't stall.
            idx = int(self.rng.integers(0, len(moves)))
        else:
            idx = int(self.rng.choice(len(moves), p=w / total))
        cls = moves[idx]
        self._last_cls = cls
        self._proposed[cls] = self._proposed.get(cls, 0) + 1

        # --- Instantiate the move with its current tuning knobs ----
        if cls is AddReticulation:
            return AddReticulation(
                max_reticulations=self._max_retics,
                max_level=self._max_level,
            )
        if cls in (FlipReticulation, ChangeReticSource,
                   ChangeReticDest, RelocateReticulation):
            # Level-raising relocation / endpoint moves: pass the level
            # cap so they can self-reject early.
            return cls(max_level=self._max_level)
        if cls is ChangeNodeHeight:
            return ChangeNodeHeight(sigma_frac=self._nh_sigma)
        if cls is ChangeInheritanceProb:
            return ChangeInheritanceProb(sigma=self._cip_sigma)
        if cls is SPR:
            decay = self._current_spr_decay()
            # Decrement the broad-mode window.  A single SPR draw
            # consumes one slot of post-reheat broad-radius credit
            # regardless of whether the move ultimately commits.
            # Tracking consumed slots separately from activations
            # lets format_stats distinguish "broad mode fired N
            # times" vs "M reheats activated it".
            if self._spr_broad_remaining > 0:
                self._spr_broad_remaining -= 1
                self._spr_broad_proposals += 1
            # Keep a trailing record so end-of-run stats can show the
            # realised decay range without sampling every tick.
            self._spr_decay_last = decay
            self._spr_decay_min_seen = min(self._spr_decay_min_seen, decay)
            self._spr_decay_max_seen = max(self._spr_decay_max_seen, decay)
            return SPR(distance_decay=decay)
        return cls()

    def report_outcome(self, accepted: bool, delta: float = 0.0) -> None:
        """Record a move outcome and update adaptive state.

        Called by the search driver (HC / SA) immediately after each
        ``generate()`` -> propose -> score pair, regardless of
        accept/reject.  Updates:

          * The rolling per-move history window (fed back to
            :meth:`_adapt_weights`).
          * Lifetime accepted/improved counters (for
            :meth:`format_stats`).
          * The Robbins-Monro sigma for the last move class when it
            was a continuous proposal (NH / CIP).
          * The magnitude-aware stagnation counter driving phase
            switching and adaptive SPR radius.

        Args:
            accepted: True iff the driver committed this move.
            delta: Score change ``new - old``; positive means strict
                improvement on the search objective (maximisation).
        """
        cls = self._last_cls
        if cls is not None and cls in self._history:
            self._history[cls].append((accepted, delta))
        if cls is not None:
            if accepted:
                self._accepted[cls] = self._accepted.get(cls, 0) + 1
            if accepted and delta > 0:
                self._improved[cls] = self._improved.get(cls, 0) + 1

            if cls is ChangeNodeHeight:
                self._sigma_obs[cls] += 1
                self._nh_sigma = self._tune_sigma(
                    self._nh_sigma,
                    self._sigma_obs[cls],
                    accepted,
                    _AdaptiveConfig.NH_SIGMA_MIN,
                    _AdaptiveConfig.NH_SIGMA_MAX,
                )
            elif cls is ChangeInheritanceProb:
                self._sigma_obs[cls] += 1
                self._cip_sigma = self._tune_sigma(
                    self._cip_sigma,
                    self._sigma_obs[cls],
                    accepted,
                    _AdaptiveConfig.CIP_SIGMA_MIN,
                    _AdaptiveConfig.CIP_SIGMA_MAX,
                )

        # Magnitude-aware reset: only genuine improvements reset
        # stagnation.  Tiny +0.01 tweaks from late-stage NH/CIP moves
        # don't count, so the counter actually climbs when the chain
        # is truly plateauing (which is what phase switching and
        # adaptive SPR radius need to see).
        if accepted and delta > self._stagnation_reset_delta:
            self._stagnation = 0
        else:
            self._stagnation += 1
            if self._stagnation > self._stagnation_peak:
                self._stagnation_peak = self._stagnation

    def on_reheat(self) -> None:
        """Notify the kernel that the driver just fired a reheat.

        Reheats are the SA layer's explicit "we're stuck" signal --
        they're computed from rolling improvement over a window, which
        is a *stronger* plateau signal than anything the kernel's own
        stagnation counter can see.  On notification we:

        1. Force ``_stagnation`` up to ``phase_patience`` so the next
           ``generate()`` call triggers a phase switch (if we aren't
           in the opposite phase already).  This flips to retic-focused
           moves to try a structurally different escape route.
        2. Arm the post-reheat broad-SPR window: the next
           ``SPR_BROAD_MODE_PROPOSALS`` SPR draws are forced to
           ``SPR_DECAY_MIN`` (near-flat, large hops) regardless of
           intervening phase switches or stagnation resets.

        The broad-SPR window is the critical half: without it, the
        phase switch from (1) resets ``_stagnation`` to 0 before a
        single broad SPR fires, reverting the move to local-only mode
        and cancelling the reheat's entire exploration intent.  With
        it, reheats actually reach the basin-escape topology moves
        they were fired to attempt.

        No-ops for the non-phased kernel (custom ``weights`` path).
        """
        self._reheat_signals += 1
        if not self._phased:
            return
        self._stagnation = max(self._stagnation, self._phase_patience)
        if self._stagnation > self._stagnation_peak:
            self._stagnation_peak = self._stagnation
        # Arm the broad-SPR window independently of the stagnation
        # counter.  If a prior reheat's window hasn't fully drained,
        # refresh to the full budget rather than accumulating -- the
        # cascade guard already limits how often this can happen.
        self._spr_broad_remaining = _AdaptiveConfig.SPR_BROAD_MODE_PROPOSALS
        self._spr_broad_activations += 1

    def get_weights(self) -> dict[str, float]:
        """Return the current effective selection probabilities.

        Honours the active phase (when phased) and the adaptive
        scaler (when enabled); falls back to the fixed weight list
        otherwise.  Useful for quick introspection or logging.

        Returns:
            ``dict`` mapping move class name to probability in [0, 1]
            (summing to 1).
        """
        if self._phased:
            moves, base = self._active_moves_and_base()
            weights = (self._adapt_weights(moves, base)
                       if self._adaptive else base)
        else:
            moves = self._all_moves
            weights = self._fixed_weights or [1.0] * len(moves)
        total = sum(weights)
        return {cls.__name__: w / total for cls, w in
                zip(moves, weights)}

    def get_stats(self) -> dict[str, dict[str, int | float]]:
        """Lifetime per-move statistics across the search.

        Returns:
            Mapping ``move_class_name -> {proposed, accepted, improved,
            acceptance_rate, improvement_rate}``.  ``acceptance_rate``
            is accepted / proposed (fraction of draws that were
            committed to the chain, including uphill-accepted moves).
            ``improvement_rate`` is improved / proposed (strict
            log-PL gain).  Both rates default to 0 when a move was
            never sampled.
        """
        stats: dict[str, dict[str, int | float]] = {}
        for cls in self._all_moves:
            n_prop = self._proposed.get(cls, 0)
            n_acc = self._accepted.get(cls, 0)
            n_imp = self._improved.get(cls, 0)
            stats[cls.__name__] = {
                "proposed": n_prop,
                "accepted": n_acc,
                "improved": n_imp,
                "acceptance_rate": (n_acc / n_prop) if n_prop else 0.0,
                "improvement_rate": (n_imp / n_prop) if n_prop else 0.0,
            }
        return stats

    def format_stats(self) -> str:
        """Return a human-readable summary of lifetime kernel statistics.

        Intended for printing at the end of a search.  Includes
        per-move proposal / accept / improve counts, phase-switch
        summary, adaptive sigma end-state, SPR radius range, and the
        post-reheat broad-SPR window activation count.
        """
        stats = self.get_stats()
        name_width = max((len(k) for k in stats), default=8)
        lines = [
            f"{'move':<{name_width}}  {'proposed':>9}  {'accepted':>9}  "
            f"{'improved':>9}  {'accept%':>8}  {'improve%':>9}"
        ]
        total_prop = sum(s["proposed"] for s in stats.values())
        for name, s in sorted(
            stats.items(), key=lambda kv: -kv[1]["proposed"],
        ):
            lines.append(
                f"{name:<{name_width}}  "
                f"{s['proposed']:>9d}  "
                f"{s['accepted']:>9d}  "
                f"{s['improved']:>9d}  "
                f"{100.0 * s['acceptance_rate']:>7.2f}%  "
                f"{100.0 * s['improvement_rate']:>8.3f}%"
            )
        lines.append(f"  total proposals: {total_prop}")
        lines.append(
            f"  phase switches: {self._phase_switches}"
            f" (reheat notifications: {self._reheat_signals};"
            f" stagnation peak: {self._stagnation_peak}/{self._phase_patience})"
        )
        lines.append(
            "  adaptive sigma -- "
            f"ChangeNodeHeight.sigma_frac={self._nh_sigma:.4f} "
            f"(n={self._sigma_obs.get(ChangeNodeHeight, 0)}), "
            f"ChangeInheritanceProb.sigma={self._cip_sigma:.4f} "
            f"(n={self._sigma_obs.get(ChangeInheritanceProb, 0)})"
        )
        lines.append(
            "  adaptive SPR radius -- "
            f"distance_decay last={self._spr_decay_last:.3f}, "
            f"range=[{self._spr_decay_min_seen:.3f}, "
            f"{self._spr_decay_max_seen:.3f}] "
            f"(max={_AdaptiveConfig.SPR_DECAY_MAX}, "
            f"min={_AdaptiveConfig.SPR_DECAY_MIN}; "
            "low=broad, high=local)"
        )
        lines.append(
            "  post-reheat broad-SPR window -- "
            f"activated={self._spr_broad_activations}x, "
            f"SPR draws in broad mode={self._spr_broad_proposals} "
            f"(window size={_AdaptiveConfig.SPR_BROAD_MODE_PROPOSALS})"
        )
        return "\n".join(lines)



# ======================================================================
# (H) Orchestration class
#
# :class:`MPL` is the user-facing entry point.  It couples the
# gene-tree precomputation, the scorer, and a search driver behind a
# small and consistent API: ``MPL(net, gts, mapping).score()`` /
# ``.search(...)``.  The static-method wrappers below simply forward
# to the module-level public functions so existing callers don't have
# to import them separately.
# ======================================================================


class MPL:
    """Maximum pseudo-likelihood scorer and search driver for a species network.

    At construction time, rho values are precomputed from the input
    gene trees (constant for the run) and a DP engine is built for
    the initial species network.  Subsequent :meth:`score` calls
    reuse the cached rho values and rebuild only the network-side DP.
    :meth:`search` wires the scorer into a search driver (HC or SA)
    with an :class:`MPLKernel` and returns the best log-PL found.

    Typical use::

        mpl = MPL(species_net, gene_trees, mapping)
        log_pl = mpl.score()                    # one-off score
        best_log_pl = mpl.search(method="sa", num_iter=20000)

    Attributes:
        net: Current species network.  Updated in place by
            :meth:`search` to the best network found.
        gene_trees: Input gene-tree collection.
        mapping: Species -> allele labels mapping.
    """

    def __init__(
        self,
        species_net: Network,
        gene_trees: GeneTrees,
        mapping: dict[str, list[str]],
    ) -> None:
        """Initialise rho precomputation and the initial DP engine.

        Gene-tree-side work (rho table, active-triplet filter) runs
        exactly once here.  Network-side DP is rebuilt on every
        :meth:`score` call because topology moves invalidate the
        engine's cached articulation sets.

        Args:
            species_net: Initial species network topology.  Branch
                lengths and gammas should already be populated.
            gene_trees: Gene-tree collection to score against.
            mapping: Species -> list of allele labels (identity
                mappings, i.e. ``{species: [species]}``, are fine).
        """
        self.net = species_net
        self.gene_trees = gene_trees
        self.mapping = mapping

        # Precompute rho (constant across any search that reuses this
        # object).  Triplets missing from the gene trees will still
        # appear in self._triplets but have all-zero rho, so the
        # active-triplet filter below skips them for scoring.
        precomputed = compute_gene_tree_triplets(
            gene_trees=self.gene_trees,
            mapping=self.mapping,
            species_labels=[n.label for n in self.net.get_leaves()],
        )
        self._triplets = precomputed.triplets
        self._rho = precomputed.rho_by_triplet
        self._active_triplets = [
            t for t in self._triplets
            if any(self._rho[t][i] > 0.0 for i in range(3))
        ]

        # Initial DP engine for the starting network.  Replaced by
        # :meth:`search` once the driver finishes (so ``self.net`` and
        # this engine always agree).
        self._triple_engine = _TripleDPEngine(self.net)

    @classmethod
    def from_nexus(
        cls,
        gt_file: str,
        st_file: str,
        mapping: dict[str, list[str]],
    ) -> "MPL":
        """Construct an :class:`MPL` from two NEXUS files.

        Args:
            gt_file: Path to a NEXUS file holding the gene trees.
            st_file: Path to a NEXUS file holding the starting species
                network.
            mapping: Species -> list of allele labels.

        Returns:
            A fully initialised :class:`MPL` ready to be scored or
            searched.
        """
        st: Network = io.read_nexus(st_file, return_type="networks")
        gts: GeneTrees = io.read_nexus(gt_file, return_type="genetrees")
        gts.species_gene_mapping(mapping)
        return cls(st, gts, mapping)

    @staticmethod
    def compute_gene_tree_triplets(
        gene_trees: GeneTrees,
        mapping: dict[str, list[str]],
        species_labels: Optional[list[str]] = None,
    ) -> GeneTreeTripletResult:
        """Thin wrapper around module-level :func:`compute_gene_tree_triplets`.

        Provided so callers that only have the :class:`MPL` class
        handy don't have to separately import the module-level
        function.  See that function's docstring for parameter
        details.
        """
        return compute_gene_tree_triplets(
            gene_trees=gene_trees,
            mapping=mapping,
            species_labels=species_labels,
        )

    @staticmethod
    def score_species_network_triplets(
        species_net: Network,
        gene_triplet_result: GeneTreeTripletResult,
    ) -> SpeciesNetworkTripletResult:
        """Thin wrapper around module-level :func:`score_species_network_triplets`.

        See that function's docstring for parameter details.
        """
        return score_species_network_triplets(
            species_net=species_net,
            gene_triplet_result=gene_triplet_result,
        )

    # ── Scoring ───────────────────────────────────────────────────

    def score(self) -> float:
        """Compute the total log pseudo-likelihood for ``self.net``.

        Returns:
            Log pseudo-likelihood (always <= 0, higher is better).
        """
        if _HAS_CYTHON_MPL:
            return _score_with_cython(
                self._triple_engine, self._active_triplets, self._rho,
            )

        total = 0.0
        for triplet in self._active_triplets:
            total += self._score_triplet(triplet)
        return total

    # ── Search ────────────────────────────────────────────────────

    def search(
        self,
        method: str = "hc",
        num_iter: int = 700,
        kernel: MPLKernel | None = None,
        max_reticulations: int | None = None,
        *,
        preset: str = "default",
        opt_bl: bool | None = None,
        fix_st: bool | None = None,
        max_lvl: int | None = None,
        pseudo: bool = True,
        backbone: Network | None = None,
        optimize_params: bool | None = None,
        optimize_scope: str | None = None,
        optimize_band: float = 5.0,
        phase_patience: int | None = None,
        save_best_path: str | os.PathLike[str] | None = None,
        reference_network: Network | None = None,
        comparison_report_path: str | os.PathLike[str] | None = None,
        comparison_top_k: int = 25,
        print_comparison: bool = True,
        **kwargs,
    ) -> float:
        """Search the network space for the highest-scoring topology.

        Builds an :class:`MPLScorer`, wires RNG seeds through the
        model/kernel/driver trio for reproducibility, and runs either
        a Hill Climbing or Simulated Annealing driver against the
        starting network ``self.net``.  ``self.net`` and
        ``self._triple_engine`` are updated in place to the best
        network found.

        End-of-run diagnostics (move kernel stats) are printed
        unconditionally.

        Args:
            preset: One-word search profile (see
                :data:`phynetpy._search_flags.SEARCH_PRESETS`):
                ``"default"`` (recommended -- accurate r>=1 results at
                ~baseline speed via near-free gamma optimisation),
                ``"fast"`` (raw climb), ``"accurate"`` (per-topology
                optimisation of gammas + incident branches), or
                ``"phylonet"`` (reproduce PhyloNet's optimise-everything
                behaviour).  Any flag below passed explicitly overrides the
                preset.
            method: Search driver -- ``"hc"`` (Hill Climbing, default)
                or ``"sa"`` (Simulated Annealing).
            num_iter: Number of proposed moves to evaluate.
            kernel: Custom :class:`MPLKernel` instance.  When
                ``None`` a default one is built with ``max_reticulations``.
            max_reticulations: Upper bound on reticulation count.
                ``None`` means unlimited.  Ignored when a custom
                ``kernel`` is supplied.
            opt_bl: Optimise branch lengths (``opt_bl`` flag).  When
                ``True`` the proposal kernel drops the continuous
                parameter moves (``ChangeNodeHeight`` /
                ``ChangeInheritanceProb``) during the topology search,
                and the best network's branch lengths + gammas are
                optimised once at the end via Brent coordinate ascent
                against the pseudo-likelihood objective.  Ignored when a
                custom ``kernel`` is supplied (except the final
                optimisation, which still runs).
            fix_st: Fix the starting-tree backbone (``fix_st`` flag).
                When ``True`` the kernel drops backbone-changing moves
                (``SPR``) so only reticulation add/remove/relocate/
                endpoint and gamma moves are proposed.  Ignored when a
                custom ``kernel`` is supplied.
            max_lvl: Maximum network level (``max_lvl`` flag).  When set
                (e.g. ``1`` for level-1), proposals that would push the
                network level above this are rejected by an accept-path
                guard, and level-raising moves self-reject early.
                ``None`` disables the cap.
            pseudo: Pseudo-likelihood scoring (``pseudo`` flag).  MPL is
                a pseudo-likelihood method, so this is always effectively
                ``True``; passing ``False`` emits a warning and is
                ignored (use ``MCMC_GT`` / ``InferNetwork_ML`` for full
                likelihood).
            backbone: Network the result must contain as a subgraph.  Drops
                every move that could destroy existing structure and
                enforces containment in the accept path.  Set by
                ``infer(start=Start(net, mode=StartMode.AUGMENT))``; leave
                ``None`` for an unconstrained search.
            optimize_params: When ``True`` (Hill-Climbing only), refine a
                proposal's continuous parameters with a Brent coordinate
                ascent *before* the accept/reject decision, so each
                reticulation topology is judged at (near) its parameter
                optimum -- matching PhyloNet's per-round behaviour.  This
                is the fix for systematically-worse r>=1 inferences: with
                it off, a new reticulation is only ever scored at its
                birth gamma (0.5) during the climb.  Defaults to ``None``
                -> taken from ``preset`` (``True`` for ``"default"``).
                Ignored for ``method="sa"`` and when a custom ``kernel``
                drops the relevant moves.
            optimize_scope: Which parameters the per-topology optimisation
                touches (see :func:`phynetpy._optimize.optimize_network_parameters`).
                ``"gamma"`` optimises only the inheritance probabilities
                -- the cheapest scope, a no-op for trees (r=0), and the
                direct fix for the "reticulation judged at gamma=0.5"
                bias.  ``"reticulation"`` also optimises the branches
                incident to reticulations; ``"all"`` optimises every
                identifiable branch length too (closest to a full
                per-topology optimisation, but markedly slower).  Branch
                lengths skipped here are still tuned by the one-time final
                optimisation.  Defaults to ``None`` -> taken from
                ``preset`` (``"gamma"`` for ``"default"``).  Only used
                when ``optimize_params`` resolves to ``True``.
            optimize_band: Lazy-gating tolerance (in log-pseudo-likelihood
                units).  A reticulation-bearing proposal is optimised only
                when its raw score is within ``optimize_band`` of the
                current score (i.e. ``raw >= current - optimize_band``), so
                clearly-bad random proposals never pay the optimisation
                cost.  :class:`~phynetpy.ModelMove.AddReticulation` is
                exempt: a new hybrid is born at gamma 0.5, and that raw
                score is not evidence the insertion is bad.  Larger values
                optimise more of the remaining proposals (more accurate,
                slower).  Only used when ``optimize_params`` is ``True``.
            phase_patience: Consecutive non-improving proposals before
                :class:`MPLKernel` switches phase.  ``None`` keeps the
                kernel default (150), which lines up with a long annealing
                schedule.  Ignored when a custom ``kernel`` is supplied.
            save_best_path: Optional Newick output path for the best
                network found.
            reference_network: Optional ground-truth / baseline
                network.  When provided, a comparison report is
                printed after the search.
            comparison_report_path: Optional path to also write the
                comparison report as UTF-8 text.
            comparison_top_k: Number of worst-performing triplets
                listed in the comparison report.
            print_comparison: When ``reference_network`` is set,
                whether to echo the report to stdout.
            **kwargs: Forwarded to the chosen search driver.  For SA
                these include ``t_start``, ``t_end``, ``n_restarts``,
                ``seed``, ``plateau_frac``, ``progress_every`` (print
                every *n* iterations; ``0`` disables), and
                ``schedule``: ``"cool"`` (high-to-low T), ``"heat"``
                (low-to-high T), or ``"geometric_reheat"`` (geometric
                cooling every ``steps_per_temp`` steps down to
                ``t_min``, with multi-signal reheats: rate-based
                plateau -- ``reheat_window`` + ``reheat_min_improve``;
                strict-stall backstop -- ``reheat_threshold``;
                optional frozen-chain detection --
                ``reheat_on_no_uphill``).  A cascade guard
                (``reheat_max_consecutive``) suspends further reheats
                after N fire without any strict run-best improvement;
                the counter resets on improvement.  Also see
                ``cooling_alpha``, ``steps_per_temp``,
                ``reheat_factor``, ``reheat_cap_mult``, ``t_min``.

        Returns:
            Log pseudo-likelihood of the best network found.

        Raises:
            ValueError: If ``method`` is not ``"hc"`` or ``"sa"``.
        """
        if not pseudo:
            warnings.warn(
                "MPL only supports pseudo-likelihood scoring; the "
                "pseudo=False flag is ignored.  Use MCMC_GT or "
                "InferNetwork_ML for full-likelihood inference.",
                stacklevel=2,
            )

        # Resolve the preset into concrete behaviour; explicit flags win.
        settings = resolve_search_preset(
            preset,
            optimize_params=optimize_params,
            optimize_scope=optimize_scope,
            opt_bl=opt_bl,
            fix_st=fix_st,
        )
        opt_bl = settings.opt_bl
        fix_st = settings.fix_st
        optimize_params = settings.optimize_params
        optimize_scope = settings.optimize_scope

        scorer = MPLScorer(self._rho, self._active_triplets)

        # ----------------------------------------------------------------
        # RNG plumbing.
        #
        # One SeedSequence drives three independent generators:
        #   * SA accept/reject (``sa_ss``),
        #   * proposal-class sampling inside the kernel (``kernel_ss``),
        #   * proposal *content* inside each move (``model_ss``, read
        #     via ``model.rng``).
        # This is the only way to make seeded MPL runs actually
        # reproducible; every tuning study depends on it.  HC does not
        # consume its own accept/reject RNG so we only inject ``seed``
        # into kwargs for the SA path.
        # ----------------------------------------------------------------
        seed_in = kwargs.pop("seed", None)
        if isinstance(seed_in, np.random.SeedSequence):
            root_ss = seed_in
        else:
            root_ss = np.random.SeedSequence(seed_in)
        sa_ss, kernel_ss, model_ss = root_ss.spawn(3)
        if method == "sa":
            kwargs["seed"] = sa_ss

        # ----------------------------------------------------------------
        # Build a Model around a deep copy of the starting network so
        # the driver's in-place mutations don't touch self.net until
        # the run completes successfully.
        # ----------------------------------------------------------------
        model = Model(rng=model_ss)
        model.network = self.net.copy()[0]
        model.set_likelihood_calculator(scorer)

        # ----------------------------------------------------------------
        # Build or fix up the proposal kernel.  If the caller supplied
        # their own kernel but didn't wire in an RNG, seed it from the
        # same SeedSequence so reproducibility still holds.
        # ----------------------------------------------------------------
        if kernel is None:
            # phase_patience is a kernel setting. Leaving it None keeps
            # the long-run default; a short hill climb passes a smaller
            # value so the reticulation phase arrives inside the budget.
            kernel_kwargs: dict = dict(
                move_types=resolve_move_types(
                    opt_bl=opt_bl, fix_st=fix_st,
                    augment_only=backbone is not None,
                ),
                max_reticulations=max_reticulations,
                max_level=max_lvl,
                rng=np.random.default_rng(kernel_ss),
            )
            if phase_patience is not None:
                kernel_kwargs["phase_patience"] = phase_patience
            kernel = MPLKernel(**kernel_kwargs)
        elif getattr(kernel, "rng", None) is None:
            kernel.rng = np.random.default_rng(kernel_ss)

        # ----------------------------------------------------------------
        # Dispatch to the chosen search driver.  The level validator is
        # the authoritative ``max_lvl`` guard: it level-checks every
        # accepted proposal regardless of which move produced it.  When a
        # backbone is required, containment is enforced in the same place
        # and for the same reason.
        # ----------------------------------------------------------------
        validate = make_level_validator(
            max_lvl, base=make_containment_validator(backbone),
        )

        # ----------------------------------------------------------------
        # Per-topology continuous-parameter optimisation (levers 1 + 2).
        # Lever 1: optimise only the reticulation parameters (gammas +
        # incident branches) so reticulation topologies are judged near
        # their optimum -- the fix for systematically-worse r>=1 results.
        # Lever 2: lazy gating -- optimise a reticulation-bearing
        # proposal when its raw score is already within ``optimize_band``
        # of the incumbent. AddReticulation is the exception: its birth
        # gamma is 0.5, so the raw score of a correct insertion is
        # typically far outside that band and has to be fit first.
        # ----------------------------------------------------------------
        optimize_proposal = None
        should_optimize = None
        if optimize_params:
            if method != "hc":
                warnings.warn(
                    "optimize_params is only supported for method='hc'; "
                    "ignoring it for this run.",
                    stacklevel=2,
                )
            else:
                def optimize_proposal(opt_m: Model) -> float:
                    """Optimise ``opt_m``'s continuous parameters in place and return its score."""
                    return optimize_network_parameters(
                        opt_m, scorer, self.mapping, scope=optimize_scope,
                    )

                def should_optimize(
                    opt_m: Model, raw_proposed: float, current: float, move,
                ) -> bool:
                    """Gate: fit gamma on a new hybrid; otherwise only near the incumbent."""
                    # Only re-optimise after a *topology* move (matching
                    # PhyloNet's per-round behaviour); pure continuous moves
                    # already explored their parameter, so re-optimising the
                    # whole reticulation neighbourhood after them is wasted
                    # work.
                    if isinstance(move, CONTINUOUS_MOVES):
                        return False
                    net = opt_m.network
                    if not any(v.is_reticulation() for v in net.V()):
                        return False
                    # Gamma 0.5 is not a fitted value. Judging the
                    # insertion there rejects hybrids whose optimised
                    # gamma beats the current tree.
                    if isinstance(move, AddReticulation):
                        return True
                    return raw_proposed >= current - optimize_band

        if method == "hc":
            searcher = HillClimbing(
                pkernel=kernel,
                model=model,
                num_iter=num_iter,
                validate=validate,
                optimize_proposal=optimize_proposal,
                should_optimize=should_optimize,
                **kwargs,
            )
        elif method == "sa":
            searcher = SimulatedAnnealing(
                pkernel=kernel,
                model=model,
                num_iter=num_iter,
                validate=validate,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown search method {method!r}; use 'hc' or 'sa'")

        end_state = searcher.run()

        # ----------------------------------------------------------------
        # End-of-run diagnostics.  Cheap, one-time, and the single
        # best tool for tuning kernel weights.  Callers that want to
        # silence it can patch ``MPLKernel.format_stats`` or capture
        # stdout.
        # ----------------------------------------------------------------
        if hasattr(kernel, "format_stats"):
            print("\nMove kernel stats (lifetime):", flush=True)
            print(kernel.format_stats(), flush=True)

        # ----------------------------------------------------------------
        # Adopt the best network into self.net and rebuild the DP
        # engine so subsequent :meth:`score` calls see the new
        # topology.  Optional Newick / comparison-report artifacts.
        # ----------------------------------------------------------------
        self.net = end_state.current_model.network
        self._triple_engine = _TripleDPEngine(self.net)

        # ----------------------------------------------------------------
        # Final Brent coordinate-ascent over *all* branch lengths and
        # gammas of the best network, against the pseudo-likelihood
        # objective (the same scorer the search used).  Runs whenever the
        # resolved preset requests a final optimisation (always for the
        # shipped presets) or ``opt_bl`` dropped the continuous moves.
        # ----------------------------------------------------------------
        final_score: float | None = None
        if settings.final_optimize or opt_bl:
            opt_model = Model(rng=np.random.default_rng())
            opt_model.network = self.net
            opt_model.set_likelihood_calculator(scorer)
            final_score = optimize_network_parameters(
                opt_model, scorer, self.mapping,
            )
            self.net = opt_model.network
            self._triple_engine = _TripleDPEngine(self.net)

        if save_best_path is not None:
            save_mpl_network_newick(self.net, save_best_path)

        if reference_network is not None:
            report = format_mpl_reference_comparison(
                self.net,
                reference_network,
                self._rho,
                self._active_triplets,
                top_k=comparison_top_k,
            )
            if print_comparison:
                print(report, flush=True)
            if comparison_report_path is not None:
                Path(comparison_report_path).write_text(report + "\n", encoding="utf-8")

        if final_score is not None:
            return final_score
        return end_state.likelihood()

    # ── Internal ──────────────────────────────────────────────────

    def _score_triplet(self, triplet: tuple[str, str, str]) -> float:
        """Contribution of one species triplet to the log-PL sum.

        Only used by the Python scoring fallback in :meth:`score`;
        the Cython path handles every triplet in a single call.

        Args:
            triplet: Canonical species triplet.

        Returns:
            ``sum_i rho_i * log(p_i)`` across the three resolutions,
            with ``p_i == 0`` terms replaced by ``_LOG_FLOOR`` and
            ``rho_i == 0`` terms skipped entirely.
        """
        x, y, z = triplet
        p_xy = self._triple_engine.calculate_triple_probability((x, y, z))
        p_xz = self._triple_engine.calculate_triple_probability((x, z, y))
        probs = (p_xy, p_xz, max(1.0 - p_xy - p_xz, 0.0))
        rho = self._rho[triplet]

        contribution = 0.0
        for rho_i, p_i in zip(rho, probs):
            if rho_i > 0.0:
                contribution += rho_i * (math.log(p_i) if p_i > 0.0 else _LOG_FLOOR)
        return contribution
