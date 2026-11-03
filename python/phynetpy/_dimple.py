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
Last Stable Edit : 10/30/26
First Included in Version : 0.7.0

Docs   - [x]
Tests  - [x]
Design - [x]

DIMPLE: divide a tree of blobs, infer each piece, stitch.

The scaffold's polytomies (out-degree greater than 2) are the blobs. This is
not :func:`phynetpy.GraphUtils.blobs`, which is a biconnected-component
decomposition. Each blob is cut into subsets of at most ``max_subset_size``
leaves, each subset is inferred with the existing MSC triplet
pseudo-likelihood engine, and the pieces are grafted back onto the scaffold.

Neighbor-joining of the overlapping subset trees follows NJMerge
(Molloy and Warnow, 2018, *Algorithms for Molecular Biology*). Each join
minimizes the Saitou and Nei (1987, *Molecular Biology and Evolution*)
criterion Q(i, j) = (n - 2) d(i, j) - r_i - r_j. The subset search itself
is Yu and Nakhleh's triplet pseudo-likelihood, already implemented by
:func:`phynetpy.infer.infer`.

The entry point is :func:`infer_dimple`. It is not a cell of the
``(data, model, criterion)`` registry: the partition and the stitch are
specific to this pipeline.

The graph, combinatorial and numerical steps (partitioning, gene-tree
restriction, displayed-tree selection, the compatibility filter,
constrained neighbor-joining, reticulation records and placement, prune
reattachment, grafting and the per-blob merge) run in the compiled core
(``phynetpy._core``, the ``_dp_*`` functions). This module keeps the
records, the input checks, the subset-search orchestration and the
driver, and every name below keeps its signature.
"""

from __future__ import annotations

import math
import random
import time
import warnings
from dataclasses import dataclass, field, replace
from typing import Callable, Optional, Sequence

import numpy as np

from .criteria import PseudoLikelihood
from .data import GeneTrees
from .GraphUtils import count_reticulations, is_tree
from .models import MSC
from .ModelSelection import reticulation_sweep
from .Network import Edge, Network, Node
from ._mcmc_gt import _populate_default_branch_lengths
from ._units import BranchLengthUnit
from . import _core

# A near-duplicate of a reticulation already accepted has to clear this
# relative gain. Any strict gain is enough for a placement that is not near
# one. Relative gain is (new - current) / |current| on the log
# pseudo-likelihood (higher is better).
REDUNDANCY_THRESHOLD = 0.02

# A blob taxon's distance to the virtual outgroup is this factor times its
# mean gene-tree distance to the scaffold siblings. DIMPLE's merger uses 2.0
# (``compute_outgroup_distances``): on estimated gene trees the raw mean let
# NJ slip the outgroup between blob leaves. 1.0 is exact on an additive tree
# metric (``test_dimple_nj`` pins that), but on DIMPLE's own subnetworks the
# merger reproduced its network for 79% of blobs at 2.0 against 69% at 1.0
# (``reports/dimple_merger_data_final_after_fix*.md``).
OUTGROUP_SCALE = 2.0

# DIMPLE fills missing divisions from the full partition enumeration only
# when a blob has at most this many items; beyond it, the seeded packings
# are all there is.
ENUM_MAX_ITEMS = 12

# Per-subset model selection. Each reticulation is charged this many
# parameters (one inheritance probability and two branch lengths) in the
# AIC/BIC rules, both when a subset's reticulation count is chosen and
# when the merger decides whether a candidate reticulation earns its place.
SELECTION_PARAMS_PER_RETICULATION = 3
# The ``gain`` rule: a reticulation is kept only when it raises the log
# pseudo-likelihood by more than this. Calibrated on 86 DIMPLE subsets of
# the 50-taxon replicates (reports/dimple_retic_calibration.md): true trees
# gain a median of 0 and a 75th percentile of 1.6, true networks a median of
# 347 and a lower quartile of 134; AIC's 3 let one true tree in five
# through, while 50 to 100 kept 40 to 41 of 44 trees and found 34 to 35 of
# 42 networks. This is DIMPLE's rule only; nothing else in PhyNetPy uses it.
SELECTION_GAIN_THRESHOLD = 50.0
SELECTION_RULES = ("gain", "aic", "bic", "elbow")

# Subset searches are simulated annealing, three independent runs, and
# the network with the higher log pseudo-likelihood is kept. The
# schedule is the 7-taxon geometric-reheat harness. 5000 iterations
# cools from 125 to about 4; it does not reach the 0.05 floor, and the
# run does not stop early. Callers override any of these by passing the
# same names to infer_dimple (num_iter, n_runs, method, t_start, ...).
SUBSET_ITERS = 5000
SUBSET_RUNS = 3
_SUBSET_ANNEALING = {
    "schedule": "geometric_reheat",
    "t_start": 125.0,
    "t_end": 0.05,
    "cooling_alpha": 0.95,
    "steps_per_temp": 75,
    "reheat_threshold": 600,
    "reheat_factor": 1.8,
    "reheat_window": 250,
    "reheat_min_improve": 1.0,
    "reheat_on_no_uphill": True,
    "reheat_cap_mult": 1.0,
    "reheat_max_consecutive": 4,
}

# Only used when a caller forces method="hc" on a reticulate subset.
# The kernel default of 150 non-improvements before the reticulation
# phase was chosen for a long annealing run.
SUBSET_PHASE_PATIENCE = 40

# Network.clean selects routines by index: 0 drops degree-0 floaters,
# 1 drops a single-child root, 2 contracts in-degree-1/out-degree-1
# chains. The stitch still uses the root as an attachment point, so
# only the contraction runs.
_CONTRACT_DEGREE2 = [False, False, True]

_OUT = "OUT"


class DimpleError(ValueError):
    """The DIMPLE inputs or a stitch step cannot be carried out."""


# ── Records ───────────────────────────────────────────────────────────


@dataclass
class PrunedSubtree:
    """A non-blob clade cut off a blob, and where it has to be put back.

    Attributes:
        leaves: Taxa in the cut clade.
        anchor_leaves: Taxa that were siblings of the cut at the moment it
            was made, on the residual tree (earlier cuts already removed).
        parent_leaves: Every taxon under the cut's parent at that moment,
            including the cut. Identifies the parent on the scaffold.
        subtree: The cut clade as its own network.
        source_type: ``blob_item`` when the anchor still sits in the blob,
            ``prune`` when the anchor is another pruned clade.
        index: Stable tie-break for attachment order.
    """

    leaves: frozenset[str]
    anchor_leaves: frozenset[str]
    parent_leaves: frozenset[str]
    subtree: Network
    source_type: str = "blob_item"
    index: int = 0


@dataclass
class BlobJob:
    """One mega-blob's partition.

    Attributes:
        name: Member labels joined by ``+``.
        tob_leaves: Scaffold leaves of this blob, including nested blobs
            and clades that were pruned.
        external_siblings: Scaffold leaves outside the blob, under the
            blob's parent. Empty when the blob is the root.
        divisions: ``n_divisions`` partitions. Each partition is a list of
            taxon sets, every one of size at most ``max_subset_size``.
        pruned: Clades cut off before inference, in the order they were cut.
        is_root: The blob node is the scaffold root.
    """

    name: str
    tob_leaves: frozenset[str]
    external_siblings: frozenset[str]
    divisions: list[list[frozenset[str]]]
    pruned: list[PrunedSubtree] = field(default_factory=list)
    is_root: bool = False


@dataclass
class SubsetRecord:
    """One subset search.

    Attributes:
        blob_name: Blob the subset belongs to (empty until the pipeline
            files the record).
        taxa: Leaves that were inferred.
        network: The chosen network.
        score: Log pseudo-likelihood of ``network`` on the subset gene
            trees, re-scored after the search.
        bound: The reticulation cap the search was given (the true count
            when a ground-truth network was supplied).
        k_selected: Reticulation cap the chosen network came from. Equals
            ``bound`` without model selection.
        selection: ``gain``, ``aic``, ``bic``, ``elbow``, or None.
        sweep: One dict per cap tried (``k``, ``best_log_lik``, ``n_params``,
            ``aic``, ``bic``, ``delta_log_lik``, ``n_retics``), or None.
        chains: One dict per search chain at the chosen cap (``method``,
            ``num_iter``, ``seed``, ``score``, ``seconds``, ``chosen``), or
            None when the record was built without them.
    """

    blob_name: str
    taxa: frozenset[str]
    network: Network
    score: float
    bound: int = 1
    k_selected: Optional[int] = None
    selection: Optional[str] = None
    sweep: Optional[list[dict]] = None
    chains: Optional[list[dict]] = None


@dataclass
class ReticTrial:
    """One attempted reticulation insertion.

    Attributes:
        child_leaves: Leaves below the reticulation.
        edge_leaves: Leaves below the edge the minor parent was placed on.
            Empty when no edge was a candidate.
        accepted: Whether the trial replaced the current network.
        score_before: Log pseudo-likelihood before the trial.
        score_after: Log pseudo-likelihood of the trial (or ``score_before``
            when there was nothing to try).
        reason: Why a trial was not scored. ``child_not_clade``,
            ``no_candidate_edge``, or ``no_destination``. Empty when the
            candidate was scored.
    """

    child_leaves: frozenset[str]
    edge_leaves: frozenset[str]
    accepted: bool
    score_before: float
    score_after: float
    reason: str = ""


@dataclass
class BlobResult:
    """One blob after merging and reattachment.

    Attributes:
        job: The partition this blob was inferred from.
        subsets: Subset inferences, in division order.
        merged: Blob network with pruned clades reattached.
        nj_relaxations: How many neighbor-joining steps ignored a constraint.
        trials: Reticulation insertions, in the order they were tried.
    """

    job: BlobJob
    subsets: list[SubsetRecord]
    merged: Network
    nj_relaxations: int = 0
    trials: list[ReticTrial] = field(default_factory=list)


@dataclass
class DimpleTrace:
    """What :func:`infer_dimple` did, attached as ``InferenceResult.raw``.

    Attributes:
        blobs: One entry per mega-blob, scaffold order.
        final: The full-taxon network.
        score: Log pseudo-likelihood of ``final`` on the gene trees, or
            None when the caller skipped it.
    """

    blobs: list[BlobResult]
    final: Network
    score: Optional[float]


@dataclass
class GraftPiece:
    """A combined blob ready to replace its scaffold clade.

    Attributes:
        tob_leaves: Scaffold leaves this piece replaces.
        network: Combined blob network.
        anchor_leaves: Where to hang this piece if an outer graft has to
            reattach it. The blob's external siblings.
    """

    tob_leaves: frozenset[str]
    network: Network
    anchor_leaves: frozenset[str]


# ── Small graph helpers ───────────────────────────────────────────────


def _labels(net: Network, node: Optional[Node] = None) -> frozenset[str]:
    """Leaf labels of ``net``, or of the clade under ``node``."""
    return _core._dp_labels(net, node)


def _label_map(net: Network) -> dict[Node, frozenset[str]]:
    """Leaf labels under every node, from one pass over ``net``.

    The clade lookups, compatibility checks and candidate-edge scans each
    need the leaf set of every node; one ``leaf_descendants_all`` is far
    cheaper than a ``leaf_descendants`` per node.
    """
    return _core._dp_label_map(net)


def _copy(net: Network) -> Network:
    """A structural copy of ``net``."""
    return _core._dp_copy(net)


def _blobs(net: Network) -> list[Node]:
    """Polytomies: nodes with more than two children. Not biconnected components."""
    return _core._dp_blobs(net)


def _is_binary_tree(net: Network) -> bool:
    """A rooted tree in which every internal node has exactly two children."""
    return _core._dp_is_binary_tree(net)


def _node_with_leaves(net: Network, leaves: frozenset[str]) -> Optional[Node]:
    """The node whose descendant leaf set is exactly ``leaves``, if there is one."""
    return _core._dp_node_with_leaves(net, leaves)


def _below_retics(net: Network, node: Node) -> Node:
    """Walk down through stacked reticulations to the first ordinary node.

    A pruned clade attached at a reticulation has to land inside the hybrid
    clade. Each reticulation has one child; stop when the node is not one.
    """
    return _core._dp_below_retics(net, node)


def _detach(net: Network, parent: Node, child: Node) -> Network:
    """Cut ``parent -> child`` and return the clade under ``child``.

    ``net`` loses that clade. The returned network is an induced copy, so
    it does not share nodes with ``net``. Degree-2 nodes left on ``net``
    are contracted; the root is left in place.

    Args:
        net (Network): Network mutated by the cut.
        parent (Node): Source of the cut edge.
        child (Node): Root of the clade being removed.

    Returns:
        Network: The removed clade on its own leaves.

    Raises:
        NetworkError: If a leaf under ``child`` cannot be induced.
    """
    return _core._dp_detach(net, parent, child)


def strip_outgroup(net: Network, outgroup: Optional[str]) -> Network:
    """Copy ``net`` without the outgroup leaf.

    The label is only removed from the scaffold. It is not added to gene
    trees. An absent label, or ``None``, returns a plain copy.

    Args:
        net (Network): Rooted scaffold.
        outgroup (str | None): Leaf label to drop.

    Returns:
        Network: Scaffold on the remaining leaves.
    """
    return _core._dp_strip_outgroup(net, outgroup)


def _ensure_internal_labels(net: Network) -> None:
    """Give every unlabeled node a name so blob identity survives copies.

    ``induced_subnetwork_by_taxa`` keeps labels and drops nodes that have
    none. ``Network._fresh_label`` is the same collision check the graft
    path uses.

    Args:
        net (Network): Scaffold or blob piece, mutated in place.
    """
    _core._dp_ensure_internal_labels(net)


# ── Input checks ──────────────────────────────────────────────────────


def validate_inputs(
    gene_trees: GeneTrees,
    tree_of_blobs: Network,
    start_tree: Network,
    *,
    max_subset_size: int = 12,
    n_divisions: int = 15,
    outgroup: Optional[str] = None,
) -> None:
    """Reject a scaffold, start tree, or gene-tree set DIMPLE cannot run on.

    Incomplete gene trees are allowed: a locus may simply lack a taxon.
    A reticulation on the scaffold is not. A blob here is a polytomy on a
    tree.

    Args:
        gene_trees (GeneTrees): One rooted tree per locus.
        tree_of_blobs (Network): Rooted scaffold whose polytomies are blobs.
        start_tree (Network): Binary tree on the scaffold's taxa.
        max_subset_size (int): Largest subset. Must be at least 4.
        n_divisions (int): Alternative partitions per blob. Must be at least 1.
        outgroup (str | None): Scaffold leaf to ignore. May be absent.

    Raises:
        DimpleError: If a size is illegal, the scaffold is reticulate, the
            start tree is not a binary tree on the scaffold taxa, or a gene
            tree is itself a network.
    """
    if max_subset_size < 4:
        raise DimpleError("max_subset_size must be at least 4.")
    if n_divisions < 1:
        raise DimpleError("n_divisions must be at least 1.")
    if not isinstance(gene_trees, GeneTrees):
        raise DimpleError("gene_trees must be a GeneTrees object.")

    for node in tree_of_blobs.V():
        if node.is_reticulation() or tree_of_blobs.in_degree(node) > 1:
            raise DimpleError(
                "The tree of blobs contains a reticulation. A blob is a "
                "polytomy (out-degree > 2) on a tree, not a biconnected "
                "component."
            )
    for tree in gene_trees.trees:
        if not is_tree(tree) or any(tree.in_degree(node) > 1 for node in tree.V()):
            raise DimpleError("A gene tree is a network; gene trees must be trees.")

    scaffold_leaves = _labels(tree_of_blobs)
    if outgroup is not None:
        scaffold_leaves = frozenset(leaf for leaf in scaffold_leaves if leaf != outgroup)
    start_leaves = _labels(start_tree)
    missing = scaffold_leaves - start_leaves
    if missing:
        sample = ", ".join(sorted(missing)[:5])
        raise DimpleError(f"The start tree is missing scaffold taxa: {sample}.")
    if not _is_binary_tree(start_tree):
        raise DimpleError(
            "The start tree is not binary. A polytomous tree of blobs is "
            "not a legal pseudo-likelihood start."
        )


# ── Partition ─────────────────────────────────────────────────────────


def _forbidden(net: Network, blobs: Sequence[Node]) -> set[tuple[str, str]]:
    """Original out-edges of every polytomy, as label pairs. Never cut these."""
    return _core._dp_forbidden(net, blobs)


def _mega_groups(net: Network, blobs: Sequence[Node]) -> list[list[Node]]:
    """Union blobs that are joined by a direct parent-child edge.

    A polytomy nested inside another through an ordinary node stays its own
    mega-blob. Only an edge from one polytomy to another merges them.
    """
    return _core._dp_mega_groups(net, blobs)


def _top_member(net: Network, members: Sequence[Node]) -> Node:
    """The member no other member is an ancestor of."""
    return _core._dp_top_member(net, members)


def _candidate_cuts(
    net: Network,
    forbidden: set[tuple[str, str]],
    blob_set: set[Node],
    max_subset_size: int,
) -> list[tuple[int, Edge]]:
    """Legal cuts: not a polytomy edge, no blob below, both sides at least 3, size at most S."""
    return _core._dp_candidate_cuts(net, forbidden, blob_set, max_subset_size)


def _oversized(net: Network, blobs: Sequence[Node], max_subset_size: int) -> list[Node]:
    """Blobs whose children's leaf counts still sum to more than S."""
    return _core._dp_oversized(net, blobs, max_subset_size)


def _prune_piece(
    net: Network,
    blobs: Sequence[Node],
    forbidden: set[tuple[str, str]],
    max_subset_size: int,
) -> list[PrunedSubtree]:
    """Cut the largest legal clade until every blob in ``net`` has size at most S.

    Mutates ``net``. Each recorded anchor is the sibling set on the tree as
    it stood when that cut was made, so an earlier cut is already gone.
    Stops when no blob is oversized or no legal cut remains, even if a blob
    is still larger than S.

    Args:
        net (Network): Working tree, mutated.
        blobs (Sequence[Node]): Polytomies whose child-leaf totals must fit in S.
        forbidden (set[tuple[str, str]]): Polytomy out-edges, as label pairs,
            that must not be cut.
        max_subset_size (int): S.

    Returns:
        list[PrunedSubtree]: Cuts in the order they were made.
    """
    return _core._dp_prune_piece(net, blobs, forbidden, max_subset_size)


def _split_clade(net: Network, leaves: frozenset[str], max_subset_size: int) -> list[frozenset[str]]:
    """Split a leaf set larger than S by the same size rules, with no blob edges to protect."""
    return _core._dp_split_clade(net, leaves, max_subset_size)


def _enumerate_partitions(weights: list[int], max_size: int, min_size: int, cap: int = 20000) -> list[list[list[int]]]:
    """Unique bin partitions in restricted-growth order, each bin in ``[min_size, max_size]``.

    DIMPLE enumerates without a cap; ``cap`` is a safety valve for a blob
    whose items are many and tiny, where the partition count explodes.
    """
    return _core._dp_enumerate_partitions(weights, max_size, min_size, cap)


def _random_pack(
    weights: list[int],
    max_size: int,
    min_size: int,
    rng: random.Random,
    items: Optional[Sequence[int]] = None,
) -> list[list[int]]:
    """One seeded first-fit packing, with DIMPLE's repair of an undersized last bin.

    ``items`` (default: every index) are shuffled with ``rng`` and placed
    first-fit. When more than one bin results and the last is under
    ``min_size``, it is merged into the first earlier bin with room. Failing
    that, items move into it from bins that stay at least ``min_size``,
    chosen with ``rng``, until it reaches ``min_size`` or nothing can move.
    This mirrors ``bin_pack_items`` in DIMPLE's divider: ``min_size`` is
    soft, ``max_size`` is hard.
    """
    return _core._dp_random_pack(weights, max_size, min_size, rng, items)


def _smallest_first_pack(weights: list[int], max_size: int, min_size: int, rng: random.Random) -> list[list[int]]:
    """DIMPLE's ``smallest_first`` strategy, used for a blob's first division.

    The lightest items fill one bin while they fit; the rest are first-fit
    packed by :func:`_random_pack`. When the first bin would be under
    ``min_size``, every item is packed by :func:`_random_pack` instead.
    ``rng`` only orders the remaining items.
    """
    return _core._dp_smallest_first_pack(weights, max_size, min_size, rng)


def _bins_to_division(bins: list[list[int]], items: list[tuple[str, frozenset[str]]]) -> list[frozenset[str]]:
    """Turn item-index bins into taxon sets."""
    return _core._dp_bins_to_division(bins, items)


def _allocations(
    items: list[tuple[str, frozenset[str]]],
    max_subset_size: int,
    n_divisions: int,
    seed: int,
) -> list[list[frozenset[str]]]:
    """Up to ``n_divisions`` distinct partitions of ``items``. No duplicates.

    Follows DIMPLE's divider (``generate_k_divisions``). When the items
    already fit in one subset, that one subset is the only division.
    Otherwise division 0 is the deterministic smallest-first packing and
    divisions ``1..K-1`` are seeded first-fit random packings (seed
    ``seed + i``), deduplicated. Only when fewer than K distinct packings
    result, and the blob has at most :data:`ENUM_MAX_ITEMS` items, is the
    remainder filled from the enumeration of all partitions with subsets
    in ``[3, S]``, shuffled with ``seed``. If no partition can keep every
    subset at 3 or more, the minimum is relaxed to 1, as DIMPLE does.
    """
    return _core._dp_allocations(items, max_subset_size, n_divisions, seed)


def _items_of(
    net: Network,
    members: set[Node],
    other_blob_labels: set[str],
    max_subset_size: int,
) -> list[tuple[str, frozenset[str]]]:
    """Allocation items: each non-member child clade of a member blob.

    Leaves that belong to a nested blob outside this mega-blob are subtracted.
    A clade still larger than S is split.
    """
    return _core._dp_items_of(net, members, other_blob_labels, max_subset_size)


def _classify_prunes(pruned: list[PrunedSubtree], residual: frozenset[str]) -> None:
    """Mark a prune ``prune`` when its anchor has already left the residual blob.

    ``source_type`` defaults to ``blob_item``. An empty anchor stays that
    way: there is no earlier prune to wait for.

    Args:
        pruned (list[PrunedSubtree]): Cuts just made, mutated in place.
        residual (frozenset[str]): Leaves still in the blob after the cuts.
    """
    _core._dp_classify_prunes(pruned, residual)


def partition_scaffold(
    net: Network,
    *,
    max_subset_size: int = 12,
    n_divisions: int = 15,
    seed: int = 0,
    outgroup: Optional[str] = None,
) -> list[BlobJob]:
    """Cut a tree of blobs into subset taxon sets.

    Forbidden edges are the original out-edges of each polytomy. The prune
    removes the largest legal clade (both sides at least 3 leaves, the cut
    itself at most S) until every blob's children sum to at most S. Blobs
    joined by a direct edge are one mega-blob and are packed together.
    ``seed`` seeds the random packings and the enumeration shuffle; the
    first division of every blob is deterministic.

    Args:
        net (Network): Rooted tree of blobs.
        max_subset_size (int): S, at least 4.
        n_divisions (int): K, at least 1.
        seed (int): Seed for the partition shuffle.
        outgroup (str | None): Scaffold leaf to drop before cutting.

    Returns:
        list[BlobJob]: One job per mega-blob, sorted by name.

    Raises:
        DimpleError: If S or K is illegal, the scaffold is reticulate, or a
            clade cannot be split under the size rules.
    """
    return _core._dp_partition_scaffold(
        net, max_subset_size, n_divisions, seed, outgroup,
    )


# ── Subset restriction and inference ──────────────────────────────────


def restrict_gene_trees(gene_trees: GeneTrees, taxa: Sequence[str]) -> GeneTrees:
    """Gene trees induced on ``taxa``.

    A tree that shares fewer than three labels with ``taxa`` is omitted: it
    has no rooted triple on the subset. A tree whose leaves are exactly
    ``taxa`` is kept as the same object, not copied: every consumer only
    reads gene trees. The mapping is the parent mapping restricted to
    alleles that remain.

    Args:
        gene_trees (GeneTrees): Full gene trees. Individual trees may lack
            taxa.
        taxa (Sequence[str]): Subset labels.

    Returns:
        GeneTrees: The restricted collection.

    Raises:
        DimpleError: If every gene tree misses the subset.
    """
    return _core._dp_restrict_gene_trees(gene_trees, taxa)


def restrict_start(start_tree: Network, taxa: Sequence[str]) -> Network:
    """Binary start tree induced on ``taxa``.

    Args:
        start_tree (Network): Binary tree that contains every label in ``taxa``.
        taxa (Sequence[str]): Subset labels.

    Returns:
        Network: The induced tree, with coalescent branch-length units set
        and missing lengths filled in.

    Raises:
        DimpleError: If a label is missing, or the induced tree is not binary.
    """
    return _core._dp_restrict_start(start_tree, taxa)


def true_reticulation_bound(true_network: Network, taxa: Sequence[str]) -> int:
    """Reticulations of the true network restricted to ``taxa`` that triplets can see.

    This is the protocol of the DIMPLE paper: each subset search is given
    the reticulation count of the ground-truth subnetwork on its taxa as
    its bound. Restriction can collapse a reticulation's cycle to three
    nodes (its two parents become parent and grandparent); no rooted
    triple distinguishes that from a tree, so such reticulations are not
    counted (:func:`phynetpy.GraphUtils.count_reticulations` with
    ``min_cycle_size=4``).

    Args:
        true_network (Network): Ground-truth network on the full taxon set.
        taxa (Sequence[str]): Subset labels.

    Returns:
        int: Detectable reticulations in the induced true subnetwork.

    Raises:
        DimpleError: If a label is not in the true network.
    """
    return _core._dp_true_reticulation_bound(true_network, taxa)


def _selection_data_size(subset_trees: GeneTrees) -> int:
    """Sample size ``n`` for the BIC rule: the number of gene trees.

    This is the module convention of :mod:`phynetpy.ModelSelection`. The
    log pseudo-likelihood is a sum over triplet counts, so ``n`` is a
    convention rather than a derivation; it lives here so one line can
    change it.
    """
    return len(subset_trees.trees)


def selection_min_gain(selection: Optional[str], data_size: int) -> float:
    """Log pseudo-likelihood a reticulation must add to earn its parameters.

    ``gain``: :data:`SELECTION_GAIN_THRESHOLD`. ``aic``:
    :data:`SELECTION_PARAMS_PER_RETICULATION`. ``bic``: half the
    parameters times ``ln(data_size)``. ``elbow`` and None: 0, which is
    DIMPLE's rule (any strict improvement).

    Args:
        selection (str | None): The model-selection rule.
        data_size (int): ``n`` for BIC.

    Returns:
        float: The minimum gain.
    """
    return _core._dp_selection_min_gain(selection, data_size)


def subset_chains(search: dict) -> list[dict]:
    """Expand subset search settings into one complete ``infer`` kwargs dict per chain.

    ``search`` is what :func:`infer_dimple` forwards. Two forms are read:

    - flat: ``method`` (default ``"sa"``), ``num_iter`` (default
      :data:`SUBSET_ITERS`), ``n_runs`` (default :data:`SUBSET_RUNS`),
      ``seed`` (default 0) and any other ``infer`` keyword. The chains
      differ only in seed: ``seed``, ``seed + 1``, ...
    - ``chains``: a list of dicts, each with its own ``method``,
      ``num_iter`` and ``n_runs`` (and any ``infer`` keyword), run as a
      portfolio and the best score kept. The flat keys of ``search`` are
      the defaults every entry starts from. Seeds continue across entries,
      so no two chains of one subset share a seed.

    Annealing chains get :data:`_SUBSET_ANNEALING` for the keys they do
    not set; hill-climbing chains get :data:`SUBSET_PHASE_PATIENCE`. Each
    dict also carries ``chain`` (its index); callers pop it before
    calling ``infer``.

    Args:
        search (dict): Search settings as described.

    Returns:
        list[dict]: One kwargs dict per chain, in seed order.

    Raises:
        DimpleError: If ``chains`` is empty or an entry asks for no runs.
    """
    shared = dict(search)
    entries = shared.pop("chains", None)
    if entries is None:
        entries = [{}]
    elif not entries:
        raise DimpleError("chains must list at least one search configuration.")
    base_seed = int(shared.pop("seed", 0))
    shared_runs = shared.pop("n_runs", SUBSET_RUNS)
    out: list[dict] = []
    for entry in entries:
        settings = dict(shared)
        settings.update(entry)
        n_runs = int(settings.pop("n_runs", shared_runs))
        if n_runs < 1:
            raise DimpleError("each chain configuration needs n_runs >= 1.")
        settings.setdefault("method", "sa")
        settings.setdefault("num_iter", SUBSET_ITERS)
        if settings["method"] == "sa":
            for key, value in _SUBSET_ANNEALING.items():
                settings.setdefault(key, value)
        else:
            # In-loop gamma fitting is hill-climbing only. The kernel default
            # of 150 non-improvements before the reticulation phase was
            # chosen for a long annealing run.
            settings.setdefault("phase_patience", SUBSET_PHASE_PATIENCE)
        for _ in range(n_runs):
            chain = dict(settings)
            chain["seed"] = base_seed + len(out)
            chain["chain"] = len(out)
            out.append(chain)
    return out


def run_subset_chain(
    subset_trees: GeneTrees,
    subset_start: Network,
    max_reticulations: int,
    chain: dict,
):
    """Run one search chain on an already-restricted subset.

    Args:
        subset_trees (GeneTrees): Gene trees restricted to the subset.
        subset_start (Network): Start network on the subset's taxa.
        max_reticulations (int): Reticulation cap for this chain.
        chain (dict): One entry of :func:`subset_chains`.

    Returns:
        InferenceResult: The chain's best network and search score.
    """
    from .infer import Start, infer

    settings = dict(chain)
    settings.pop("chain", None)
    if settings.get("method") == "sa":
        settings.pop("phase_patience", None)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="optimize_params is only supported for method='hc'",
        )
        return infer(
            subset_trees,
            model=MSC(),
            criterion=PseudoLikelihood(),
            start=Start(subset_start),
            max_reticulations=max_reticulations,
            **settings,
        )


def finish_subset(
    subset_trees: GeneTrees,
    taxa: frozenset[str],
    network: Network,
    max_reticulations: int,
    *,
    k_selected: Optional[int] = None,
    selection: Optional[str] = None,
    sweep: Optional[list[dict]] = None,
    chains: Optional[list[dict]] = None,
) -> SubsetRecord:
    """Re-score a chosen subset network and file it as a :class:`SubsetRecord`.

    The score is a fresh :func:`phynetpy.infer.score` of ``network`` on
    ``subset_trees``, so the trace does not depend on a search statistic.
    ``blob_name`` is left empty for the caller to fill.

    Args:
        subset_trees (GeneTrees): Gene trees restricted to the subset.
        taxa (frozenset[str]): The subset's taxa.
        network (Network): The network chosen for the subset.
        max_reticulations (int): The cap the search ran under.
        k_selected (int | None): Reticulation count model selection chose;
            defaults to the cap.
        selection (str | None): The selection rule used, if any.
        sweep (list[dict] | None): Per-cap rows of the selection sweep.
        chains (list[dict] | None): Per-chain summaries from
            :func:`_chain_summary`.

    Returns:
        SubsetRecord: The subset's network, score, and search trace.
    """
    from .infer import score

    value = float(score(
        network, subset_trees, model=MSC(), criterion=PseudoLikelihood(),
    ))
    return SubsetRecord(
        "", taxa, network, value, bound=max_reticulations,
        k_selected=max_reticulations if k_selected is None else k_selected,
        selection=selection, sweep=sweep, chains=chains,
    )


def _chain_summary(chain: dict, result, seconds: float, chosen: bool) -> dict:
    """One trace row for a subset chain: its settings, score, time, and whether it won."""
    return {
        "chain": chain["chain"], "method": chain["method"],
        "num_iter": int(chain["num_iter"]), "seed": int(chain["seed"]),
        "score": float(result.score), "seconds": float(seconds), "chosen": chosen,
    }


def _infer_subset(
    gene_trees: GeneTrees,
    start_tree: Network,
    taxa: frozenset[str],
    max_reticulations: int,
    search: dict,
    *,
    model_selection: Optional[str] = None,
) -> SubsetRecord:
    """Infer one subset by a portfolio of search chains and keep the best.

    The chains come from :func:`subset_chains`: by default
    :data:`SUBSET_RUNS` annealing runs of :data:`SUBSET_ITERS` iterations
    that differ only in seed, or an explicit ``chains`` portfolio (for
    instance annealing runs plus hill-climbing runs). Without model
    selection every chain searches at the cap ``max_reticulations`` and
    the network with the higher log pseudo-likelihood is kept. With
    ``model_selection`` the same chains are run for every cap from 0 to
    ``max_reticulations`` (one more for the elbow, which needs two steps
    to see a bend) through :func:`phynetpy.ModelSelection.reticulation_sweep`,
    and the cap is chosen by the rule. Each cap above 0 starts its chains
    from the best network of the cap below (a free start, the search may
    still reshape it), so a step measures what one more reticulation adds
    rather than the luck of an independent search. ``gain`` keeps raising
    the cap while a step adds more than :data:`SELECTION_GAIN_THRESHOLD`.
    The elbow's extra cap is never selected: a choice above
    ``max_reticulations`` is clamped. The score returned is a fresh
    :func:`phynetpy.infer.score` of the chosen network, so the trace does
    not depend on a search statistic.

    Args:
        gene_trees (GeneTrees): Full-taxon gene trees. Each tree may omit taxa.
        start_tree (Network): Binary start tree on the full taxon set.
        taxa (frozenset[str]): Labels in this subset.
        max_reticulations (int): Reticulation cap (the true count when a
            ground-truth network drives the run).
        search (dict): Search settings, see :func:`subset_chains`.
        model_selection (str | None): ``gain``, ``aic``, ``bic``,
            ``elbow``, or None.

    Returns:
        SubsetRecord: The chosen network, its score, the sweep and the
        chains, with an empty ``blob_name`` for the caller to fill.

    Raises:
        DimpleError: If the subset gene trees or the restricted start tree
            are illegal, or the rule is unknown.
    """
    if model_selection is not None and model_selection not in SELECTION_RULES:
        raise DimpleError(
            f"model_selection must be one of {SELECTION_RULES} or None, "
            f"not {model_selection!r}."
        )
    subset_trees = restrict_gene_trees(gene_trees, sorted(taxa))
    subset_start = restrict_start(start_tree, sorted(taxa))
    chains = subset_chains(search)

    found: dict[tuple[int, int], object] = {}
    timings: dict[tuple[int, int], float] = {}

    def best_key(cap: int) -> tuple[int, int]:
        """``(cap, chain)`` of the highest-scoring finished chain at ``cap``."""
        keys = [(cap, chain["chain"]) for chain in chains if (cap, chain["chain"]) in found]
        return max(keys, key=lambda key: found[key].score)

    def run_one(cap: int, chain: dict, warm: bool):
        """Run ``chain`` at ``cap``, warm-started from the cap below when asked, and file it."""
        start_net = subset_start
        if warm and cap > 0 and any((cap - 1, c["chain"]) in found for c in chains):
            # Sweep step: continue from the best network one cap below. The
            # engine copies the start, so the stored result is untouched.
            start_net = found[best_key(cap - 1)].best
            start_net.set_branch_length_unit(BranchLengthUnit.COALESCENT_2N)
        started = time.perf_counter()
        result = run_subset_chain(subset_trees, start_net, cap, chain)
        found[(cap, chain["chain"])] = result
        timings[(cap, chain["chain"])] = time.perf_counter() - started
        return result

    if model_selection is None:
        for chain in chains:
            run_one(max_reticulations, chain, warm=False)
        chosen_k = max_reticulations
        sweep = None
    else:
        caps = list(range(0, max_reticulations + 1))
        if model_selection == "elbow":
            caps.append(max_reticulations + 1)
        by_index = {chain["chain"]: chain for chain in chains}

        def search_fn(cap: int, index: int) -> float:
            """Score of chain ``index`` at ``cap``, in the form ``reticulation_sweep`` calls."""
            return float(run_one(cap, by_index[index], warm=True).score)

        swept = reticulation_sweep(
            search_fn, caps, seeds=[chain["chain"] for chain in chains],
            data_size=_selection_data_size(subset_trees),
            params_per_reticulation=SELECTION_PARAMS_PER_RETICULATION,
            log_lik_label="log-pseudo-likelihood", progress=False,
        )
        if model_selection == "gain":
            chosen_k = swept.rows[0].k
            for row in swept.rows[1:]:
                if row.delta_log_lik is not None and row.delta_log_lik > SELECTION_GAIN_THRESHOLD:
                    chosen_k = row.k
                else:
                    break
        else:
            chosen_k = int(swept.best_by(model_selection))
        chosen_k = min(chosen_k, max_reticulations)
        sweep = [
            {
                "k": row.k,
                "best_log_lik": float(row.best_log_lik),
                "n_params": int(row.n_params),
                "aic": float(row.aic),
                "bic": float(row.bic),
                "delta_log_lik": None if row.delta_log_lik is None else float(row.delta_log_lik),
                "n_retics": count_reticulations(found[best_key(row.k)].best),
            }
            for row in swept.rows
        ]
    winner = best_key(chosen_k)
    summaries = [
        _chain_summary(chain, found[(chosen_k, chain["chain"])],
                       timings[(chosen_k, chain["chain"])], (chosen_k, chain["chain"]) == winner)
        for chain in chains if (chosen_k, chain["chain"]) in found
    ]
    return finish_subset(
        subset_trees, taxa, found[winner].best, max_reticulations,
        k_selected=chosen_k, selection=model_selection, sweep=sweep, chains=summaries,
    )


# ── Displayed trees ───────────────────────────────────────────────────


def _miss_fraction(
    tree: Network,
    keep: set[str],
    observed: dict[tuple[str, str, str], int],
) -> float:
    """Fraction of observed rooted triples on ``keep`` that ``tree`` does not display.

    Lower is better. Fewer than three shared taxa is a total miss. This is
    not an MSC pseudo-likelihood.
    """
    return _core._dp_miss_fraction(tree, keep, observed)



def _displayed_tree_choices(net: Network) -> list[dict[Node, Edge]]:
    """The in-edge each tree of :func:`get_all_subtrees` keeps per reticulation, in its order."""
    return _core._dp_displayed_tree_choices(net)


def select_displayed_trees_with_choices(
    networks: Sequence[Network],
    observed: Optional[dict[tuple[str, str, str], int]] = None,
) -> tuple[list[Network], list[Optional[dict[Node, Edge]]]]:
    """:func:`select_displayed_trees`, also returning which hybrid edge each tree kept.

    The second list has one entry per network: the kept in-edge per
    reticulation of the chosen tree, or None for a tree (nothing to choose).
    :func:`reticulation_records` reads a network's leaf sets on that tree,
    so the sides it records are clades of the tree the backbone was built
    from.

    Args:
        networks (Sequence[Network]): Inferred subnetworks, some of them trees.
        observed (dict | None): Counts of rooted triples ``(a, b, c)`` with
            ``a < b``. Required only when some network has two or more
            displayed trees.

    Returns:
        tuple[list[Network], list[dict[Node, Edge] | None]]: The chosen tree
        per input, and the hybrid in-edge it kept per reticulation.

    Raises:
        DimpleError: If a network has several displayed trees and
            ``observed`` is None.
    """
    chosen, choices = _core._dp_select_displayed_trees_with_choices(networks, observed)
    return chosen, choices


def select_displayed_trees(
    networks: Sequence[Network],
    observed: Optional[dict[tuple[str, str, str], int]] = None,
) -> list[Network]:
    """Replace each network with the displayed tree that misses the fewest triples.

    A tree is returned as itself. A network with one displayed tree is
    returned without consulting ``observed``. Ties keep the earlier tree
    from :func:`phynetpy.GraphUtils.get_all_subtrees`.

    Args:
        networks (Sequence[Network]): Inferred subnetworks, some of them trees.
        observed (dict | None): Counts of rooted triples ``(a, b, c)`` with
            ``a < b``. Required only when some network has two or more
            displayed trees.

    Returns:
        list[Network]: One tree per input, same order.
    """
    return select_displayed_trees_with_choices(networks, observed)[0]


def observed_triples(gene_trees: GeneTrees, taxa: Sequence[str]) -> dict[tuple[str, str, str], int]:
    """Count rooted triples of the gene trees on ``taxa``.

    Args:
        gene_trees (GeneTrees): Source trees.
        taxa (Sequence[str]): Labels that may enter a triple.

    Returns:
        dict: ``(a, b, c)`` with ``a < b`` mapped to how many gene trees
        display it.
    """
    return _core._dp_observed_triples(gene_trees, taxa)


# ── Compatibility and neighbor-joining ────────────────────────────────


def _clade_sets(net: Network) -> list[frozenset[str]]:
    """Leaf set of every node of ``net``, computed once per tree."""
    return _core._dp_clade_sets(net)


def _splits(clades: Sequence[frozenset[str]], keep: frozenset[str]) -> list[frozenset[str]]:
    """Nontrivial clades among ``clades`` after restriction to ``keep``."""
    return _core._dp_splits(clades, keep)


def trees_incompatible(
    left: Network,
    right: Network,
    left_clades: Optional[Sequence[frozenset[str]]] = None,
    right_clades: Optional[Sequence[frozenset[str]]] = None,
) -> bool:
    """Whether two trees conflict on their shared taxa.

    Fewer than four shared taxa cannot conflict. Otherwise some clade of
    one tree crosses a clade of the other (all four intersections nonempty).

    Args:
        left (Network): A tree.
        right (Network): A tree.
        left_clades, right_clades: Each tree's clade leaf sets from
            :func:`_clade_sets`, so a caller comparing many pairs computes
            them once per tree. Computed here when omitted.

    Returns:
        bool: True when the unrooted shapes disagree on the shared labels.
    """
    return _core._dp_trees_incompatible(left, right, left_clades, right_clades)


def greedy_compatible(trees: Sequence[Network]) -> list[Network]:
    """First-fit subset of pairwise compatible trees.

    Args:
        trees (Sequence[Network]): Candidate trees, in the order they should
            be considered.

    Returns:
        list[Network]: The trees that were kept.
    """
    return _core._dp_greedy_compatible(trees)


def max_clique_compatible(trees: Sequence[Network]) -> list[Network]:
    """Largest set of pairwise compatible trees, in input order.

    DIMPLE's default compatibility filter (``select_compatible_subset`` with
    strategy ``maxclique``). Two trees are joined in the compatibility graph
    when :func:`trees_incompatible` is false; the largest clique is kept.
    Among equally large cliques the first one networkx reports wins.
    :func:`greedy_compatible` is the first-fit alternative.

    Args:
        trees (Sequence[Network]): Candidate trees.

    Returns:
        list[Network]: The trees of the largest clique, in input order.
    """
    return _core._dp_max_clique_compatible(trees)


# The neighbor-joining constraint tree lives in the Rust core; the class
# keeps its name and interface (leaves, adj, broken, cherries, join, ...).
_Constraint = _core._Constraint


def _leaf_network(label: str) -> Network:
    """A one-node network whose only node is the leaf ``label``."""
    return _core._dp_leaf_network(label)


def _join_clusters(left: Network, right: Network) -> Network:
    """A new root with ``left`` and ``right`` as its two clades."""
    return _core._dp_join_clusters(left, right)


def _distance(dist: dict[tuple[str, str], float], left: str, right: str) -> float:
    """Pairwise distance. The matrix stores no diagonal, so a label against itself is 0."""
    if left == right:
        return 0.0
    return dist[(left, right)]


def _reverse_upward(net: Network, start: Node, ignore: Optional[Node] = None) -> None:
    """Reverse edges from ``start`` up to the root, skipping ``ignore``.

    After a new root is inserted, the path from the old root down to
    ``start`` has to point the other way, or the new root is not the only
    in-degree-0 node.

    Args:
        net (Network): Tree mutated in place.
        start (Node): Lowest node of the path that is reversed.
        ignore (Node | None): Parent to skip. The new root, when the edge
            into ``start`` was just replaced.
    """
    _core._dp_reverse_upward(net, start, ignore)


def _reroot_on_edge(net: Network, parent: Node, child: Node) -> None:
    """Subdivide ``parent`` to ``child`` with a new root and reverse the path above it.

    Args:
        net (Network): Tree to reroot in place.
        parent (Node): Source of the edge that becomes the root edge.
        child (Node): Destination of that edge.
    """
    _core._dp_reroot_on_edge(net, parent, child)


def _reroot_above(net: Network, label: str) -> None:
    """Make the parent of leaf ``label`` the root, by reversing the path upward.

    Used to delete the virtual outgroup after neighbor-joining: the outgroup
    leaf hangs off the new root, and removing it leaves that root in place.
    A missing label, or a leaf that is already the root, is left unchanged.

    Args:
        net (Network): Tree mutated in place.
        label (str): Leaf whose parent becomes the root.
    """
    _core._dp_reroot_above(net, label)


def _smaller_side(net: Network, child: Node) -> tuple[str, ...]:
    """Sorted labels of the smaller side of the edge into ``child``."""
    return _core._dp_smaller_side(net, child)


def _root_by_triples(
    tree: Network,
    counts: dict[tuple[str, str, str], int],
) -> Network:
    """Copy of ``tree`` rooted on the edge that displays the most observed triples.

    The two edges under the current root are the same rooting; one of them is
    kept as the current-root candidate and the other is skipped. Ties break
    by the sorted labels of the smaller side of the edge. An empty count, or
    a count that no rooting displays, returns ``tree`` unchanged.

    Args:
        tree (Network): Unrooted-in-practice neighbor-joining tree.
        counts (dict): Observed rooted triples ``(a, b, c)`` with ``a < b``.

    Returns:
        Network: The chosen rooting.
    """
    return _core._dp_root_by_triples(tree, counts)


def overlap_nj(
    trees: Sequence[Network],
    labels: Sequence[str],
    matrix: np.ndarray,
    *,
    outgroup_distances: Optional[dict[str, float]] = None,
) -> tuple[Network, int]:
    """Neighbor-joining constrained by sibling pairs of ``trees``.

    ``matrix`` is indexed in the same order as ``labels``. A virtual outgroup
    is added from ``outgroup_distances`` (mean distance to scaffold siblings,
    already scaled) and then deleted, so the tree roots toward the scaffold
    parent. The outgroup is not a leaf of any constraint tree. When every
    pair violates a constraint, that step is relaxed and counted.

    The pair chosen at each step minimizes Q(i, j) = (n - 2) d(i, j) - r_i - r_j
    (Saitou and Nei, 1987, *Molecular Biology and Evolution*). Which pairs
    are legal is the NJMerge constraint (Molloy and Warnow, 2018).

    Args:
        trees (Sequence[Network]): Compatible constraint trees.
        labels (Sequence[str]): Taxon order of ``matrix``.
        matrix (np.ndarray): Square pairwise distances.
        outgroup_distances (dict | None): Distance from each label to the
            virtual outgroup. Omit to skip the outgroup.

    Returns:
        tuple[Network, int]: The merged tree (no virtual outgroup) and how
        many steps were relaxed.

    Raises:
        DimpleError: If a distance the merge needs is undefined, or the
            outgroup distances do not cover ``labels``.
    """
    return _core._dp_overlap_nj(trees, labels, matrix, outgroup_distances)


def _leaf_path_lengths(tree: Network, wanted: set[str]) -> dict[tuple[str, str], float]:
    """Path length between every pair of ``wanted`` leaves of one tree.

    One post-order pass: each node carries the distance to every wanted
    leaf below it, and two leaves meet once, at their MRCA. An edge with no
    length counts as one unit, as the newick readers default it.

    Args:
        tree (Network): A gene tree.
        wanted (set[str]): Leaf labels to measure.

    Returns:
        dict[tuple[str, str], float]: ``(a, b)`` with ``a < b`` to the path length.
    """
    return _core._dp_leaf_path_lengths(tree, wanted)


def gene_tree_distance_matrix(
    gene_trees: GeneTrees,
    taxa: Sequence[str],
    *,
    strict: bool = True,
) -> tuple[list[str], np.ndarray]:
    """Mean path length over gene trees in which both taxa occur.

    This is not :func:`phynetpy.GraphUtils.pairwise_leaf_distance` (one tree)
    and not :func:`phynetpy.GraphUtils.average_path_distance` (a distance
    between two networks). Each tree contributes through
    :func:`_leaf_path_lengths`; the mean is taken only over the trees that
    contain the pair. A pair that never co-occurs is NaN. The trees are
    read on their full leaf sets; only ``taxa`` are measured.

    Args:
        gene_trees (GeneTrees): Source trees. Branch lengths are path weights.
        taxa (Sequence[str]): Labels to index the matrix, in sorted order
            in the return value.
        strict (bool): When True, a NaN pair raises. When False, NaN is left
            in the matrix for a caller that will not use that pair.

    Returns:
        tuple[list[str], np.ndarray]: Sorted labels and the square matrix.

    Raises:
        DimpleError: If ``strict`` and some pair never co-occurs.
    """
    return _core._dp_gene_tree_distance_matrix(gene_trees, taxa, strict)


def _outgroup_distances(
    labels: Sequence[str],
    matrix_labels: Sequence[str],
    matrix: np.ndarray,
    siblings: frozenset[str],
    *,
    scale: Optional[float] = None,
) -> dict[str, float]:
    """``scale`` times each blob taxon's mean distance to the scaffold siblings.

    If the sibling distances are a tree metric, adding a constant only
    lengthens the outgroup's pendant edge. Multiplying them does not stay
    additive, and neighbor-joining can then attach the outgroup to the
    wrong edge.

    Args:
        labels (Sequence[str]): Blob taxa to give a distance.
        matrix_labels (Sequence[str]): Row order of ``matrix``.
        matrix (np.ndarray): Pairwise distances.
        siblings (frozenset[str]): Scaffold leaves outside the blob.
        scale (float | None): Multiplier on the mean. Default
            :data:`OUTGROUP_SCALE`, DIMPLE's 2.0.

    Returns:
        dict[str, float]: Outgroup distance for each blob taxon that has
        a finite distance to every sibling.
    """
    return _core._dp_outgroup_distances(labels, matrix_labels, matrix, siblings, scale)


# ── Reticulation placement ────────────────────────────────────────────



def _major_edges(net: Network) -> dict[Node, Edge]:
    """The in-edge the major tree keeps at every node of in-degree two or more.

    Highest gamma first, then source label: the order
    :func:`phynetpy.GraphUtils.get_all_subtrees` enumerates, so its first
    tree keeps exactly these edges.
    """
    return _core._dp_major_edges(net)


def _view(net: Network, drop: Sequence[Edge]) -> tuple[Network, dict[Node, Node]]:
    """A copy of ``net`` without the edges in ``drop``.

    An internal node left childless is removed, and so on upward, because
    :meth:`Network.leaf_descendants` would report it as a leaf. Degree-2
    chains are kept, so a surviving node of ``net`` is still found through
    the returned map; a removed node is simply absent from the view.
    """
    view, node_map = _core._dp_view(net, list(drop))
    return view, node_map


def _attachment_leaves(
    net: Network,
    kept: dict[Node, Edge],
    drop: Sequence[Edge],
    start: Node,
    taxa: frozenset[str],
) -> tuple[frozenset[str], int]:
    """Clade under the point a removed hybrid edge attaches to, and its height.

    ``start`` is the hybrid edge's source. Once ``drop`` (that edge and the
    other reticulations' non-reference edges) is gone, the edge attaches
    where ``start`` stands in the remaining tree: under ``start`` itself if
    it still has a child, otherwise under its nearest surviving ancestor
    (a reticulation's ancestor is through its ``kept`` edge). The clade is
    empty when nothing above ``start`` survives.

    The height counts the one-child nodes between the attachment point and
    the clade's top node. Those are the positions of other hybrid edges
    that leave or enter the same clade; on a backbone where they have been
    inserted, the edges above the clade form a chain, and the height says
    which link this edge belongs on.
    """
    leaves, height = _core._dp_attachment_leaves(net, kept, list(drop), start, taxa)
    return leaves, height


def _chain_depth(net: Network, node: Node) -> int:
    """Reticulations stacked directly under ``node``, each the other's only child."""
    return _core._dp_chain_depth(net, node)


def reticulation_records(net: Network, kept: Optional[dict[Node, Edge]] = None) -> list[dict]:
    """One record per in-degree-2 node, with parent sides ordered by gamma.

    A node of any other in-degree is not a record. ``kept_sibling_leaves``
    is the higher-gamma side and ``minor_sibling_leaves`` is the lower-gamma
    side, which is the side added back when the backbone kept the major
    parent.

    Leaf sets are read on a displayed tree, not on the network. ``kept``
    names the in-edge that tree keeps at each other reticulation (default:
    the major tree, :func:`_major_edges`). ``retic_leaves`` is the clade
    under the reticulation in that tree; each side is the clade under the
    point the parent's hybrid edge attaches to once it is removed
    (:func:`_attachment_leaves`); the signature is read on the same tree.
    On a level-1 network these are the network's own leaf sets. With nested
    reticulations the network's leaf sets leak through other hybrid edges:
    the child is then not a clade of any tree the backbone displays, and a
    side can contain the child itself or be empty when the parent is a
    reticulation whose only child this is. ``retic_depth`` counts
    reticulations stacked directly below this one; the merger places the
    upper one of such a chain first. ``kept_height`` and ``minor_height``
    say how many other hybrid-edge positions lie between each side's
    attachment point and its clade (see :func:`_attachment_leaves`).

    Args:
        net (Network): A network, possibly a tree (then the list is empty).
        kept (dict[Node, Edge] | None): The displayed tree's kept in-edge
            per reticulation, as :func:`select_displayed_trees_with_choices`
            returns. None means the major tree.

    Returns:
        list[dict]: Records carrying ``signature``, leaf sets, gammas and
        ``retic_depth``, in label order of the reticulation node.
    """
    return _core._dp_reticulation_records(net, kept)


def collect_reticulation_candidates(
    networks: Sequence[Network],
    references: Optional[Sequence[Optional[dict[Node, Edge]]]] = None,
) -> list[dict]:
    """One candidate per hybridization event across ``networks``.

    Records from every network are grouped with
    :func:`phynetpy.GraphUtils.group_reticulations`: two records are the
    same event when their parent-side sibling clades overlap, in either
    parent order. Each group contributes the record with the largest child
    clade (ties keep the earliest), which is how DIMPLE's
    ``collect_unique_retics`` picks a representative. Literal repeats of a
    representative are then dropped. The same event seen by several
    overlapping subsets is therefore tried once, not once per subset.

    Args:
        networks (Sequence[Network]): Inferred subnetworks.
        references (Sequence[dict | None] | None): Per network, the in-edge
            its displayed tree kept at each reticulation (see
            :func:`reticulation_records`); None means the major tree.

    Returns:
        list[dict]: Records from :func:`reticulation_records`, one per event.
    """
    return _core._dp_collect_reticulation_candidates(networks, references)


def _mrca_or_none(net: Network, leaves: frozenset[str]) -> Optional[Node]:
    """MRCA of ``leaves``, or None when a label is absent."""
    return _core._dp_mrca_or_none(net, leaves)


def _clade_node(net: Network, leaves: frozenset[str]) -> Optional[Node]:
    """The node whose leaf set is exactly ``leaves``, or their MRCA.

    An exact clade is preferred. The MRCA is the fallback when a graft has
    left extra leaves inside that region, so the original set is no longer
    one node's descendant set.

    Args:
        net (Network): Tree or network to search.
        leaves (frozenset[str]): Target taxon set.

    Returns:
        Node | None: The clade node, the MRCA, or None when a label is
        absent or the MRCA cannot be computed.
    """
    return _core._dp_clade_node(net, leaves)


def _child_clade(net: Network, record: dict) -> Optional[Node]:
    """MRCA of the reticulation's leaves, or None if it is not their clade.

    Leaves from other subsets may sit inside the clade: the source network
    never saw them. A leaf of the source network that sits inside the clade
    but was not under the reticulation means this is the wrong node.

    Args:
        net (Network): Current backbone.
        record (dict): A record from :func:`reticulation_records`.

    Returns:
        Node | None: The clade node, or None when the leaves are not a clade
        on the source network's taxa.
    """
    return _core._dp_child_clade(net, record)


def _side_leaves(
    net: Network,
    child: Node,
    parent: Node,
    taxa: set[str] | frozenset[str],
) -> set[str]:
    """Leaves beside ``child`` under ``parent``, restricted to ``taxa``."""
    return _core._dp_side_leaves(net, child, parent, taxa)


def choose_inserted_side(net: Network, record: dict) -> frozenset[str]:
    """Sibling leaves of the parental side the backbone does not already hold.

    Returns one of the record's own sets, so identity distinguishes the
    major side from the minor side. A reticulation child has two parents.
    Each parent's sibling leaves are compared with the two recorded sides,
    and the parent label breaks ties, so the choice does not follow
    ``get_parents()`` order.

    Args:
        net (Network): Current backbone.
        record (dict): A record from :func:`reticulation_records`.

    Returns:
        frozenset[str]: The side that should be inserted.
    """
    return _core._dp_choose_inserted_side(net, record)


def _destination_edge(
    net: Network,
    child: Node,
    held: frozenset[str],
    taxa: set[str],
) -> Optional[Edge]:
    """Edge above ``child`` on the parental side the backbone already holds.

    One parent means one edge. Two parents means the incoming edge whose
    other children match ``held``. The source label breaks a tie.

    Args:
        net (Network): Current backbone.
        child (Node): MRCA of the reticulation's descendant leaves.
        held (frozenset[str]): Sibling leaves of the side not being inserted.
        taxa (set[str]): Taxa of the source network.

    Returns:
        Edge | None: The edge to split as the hybrid's destination.
    """
    return _core._dp_destination_edge(net, child, held, taxa)


def candidate_edges(net: Network, record: dict) -> list[Edge]:
    """Edges where the missing parental side can be attached.

    Exact leaf-set matches come first. With none, partial matches are
    ordered by how many of the target leaves they cover. The edge into the
    reticulation child itself is excluded, so the source and destination of
    the edit are never the same edge (that edit would be a bubble).

    Args:
        net (Network): Current backbone.
        record (dict): Reticulation to place.

    Returns:
        list[Edge]: Candidate source edges, best match first. Empty when
        nothing on the tree carries the target leaves.
    """
    return _core._dp_candidate_edges(net, record)


def _edge_distance(net: Network, start: Node, goal: Node, limit: int = 2) -> int:
    """Undirected distance between two nodes, or ``limit + 1`` when farther."""
    return _core._dp_edge_distance(net, start, goal, limit)


def _sort_records(records: list[dict], net: Network) -> list[dict]:
    """Place an event before one whose inserted side contains its child leaves.

    ``net`` is the backbone as the placement sees it (see
    :func:`_placement_view`). An upper reticulation of a chain also goes
    before the lower one.
    """
    return _core._dp_sort_records(records, net)


def _insert_hybrid(
    net: Network,
    source: Edge,
    destination: Edge,
    major_gamma: float,
    minor_gamma: float,
) -> tuple[str, str]:
    """Split ``source`` and ``destination`` and set the two inheritance probabilities.

    ``destination`` is the edge above the reticulation child (the side the
    backbone already holds). ``source`` is the side being added. Gamma on an
    edge that already entered a reticulation stays on the lower half of the
    split. ``add_hybrid`` does not take inheritance probabilities, so they
    are written onto the new edges after the split.

    Args:
        net (Network): Backbone mutated in place.
        source (Edge): Edge that donates the new parental side.
        destination (Edge): Edge above the reticulation child.
        major_gamma (float): Inheritance probability of the side the backbone
            already holds.
        minor_gamma (float): Inheritance probability of the side being added.

    Returns:
        tuple[str, str]: Labels of the new hybrid edge's source node and of
        the reticulation node, which name the inserted edge on any copy.
    """
    minor_label, retic_label = _core._dp_insert_hybrid(
        net, source, destination, major_gamma, minor_gamma,
    )
    return minor_label, retic_label





def _placement_view(net: Network, inserted: set[tuple[str, str]]) -> Network:
    """The backbone as its displayed tree: ``net`` without the hybrid edges placed so far.

    Clades are read on this view, so a reticulation already inserted does
    not leak its child's leaves into the clade of an edge above its source
    and spoil a later candidate's clade test or exact edge match. ``net``
    itself is returned while nothing has been inserted; a pre-existing
    reticulation of the input backbone is part of the view.
    """
    return _core._dp_placement_view(net, inserted)


def _edge_by_labels(net: Network, edge: Edge) -> Optional[Edge]:
    """The edge of ``net`` between the nodes named like ``edge``'s endpoints, or None."""
    return _core._dp_edge_by_labels(net, edge)


def place_reticulations(
    tree: Network,
    networks: Sequence[Network],
    score_fn: Callable[[Network], float],
    min_gain: float = 0.0,
    references: Optional[Sequence[Optional[dict[Node, Edge]]]] = None,
) -> tuple[Network, list[ReticTrial]]:
    """Greedily add reticulations from ``networks`` while the score rises.

    Each trial is a copy. The copy is kept only when it is the one that was
    scored and its log pseudo-likelihood is higher by more than
    ``min_gain`` (DIMPLE's rule is any strict gain, ``min_gain=0``; an
    AIC or BIC rule charges the reticulation's parameters, see
    :func:`selection_min_gain`). A non-finite score of the starting tree is
    an error, not a reason to add nothing. A candidate within two edges of
    an accepted reticulation with an overlapping child clade must clear
    :data:`REDUNDANCY_THRESHOLD`.

    Clades are read on the backbone's displayed tree
    (:func:`_placement_view`): the hybrid edges inserted here are not
    followed, so nested events can be placed in either order.

    Args:
        tree (Network): Backbone, usually the neighbor-joined tree.
        networks (Sequence[Network]): Inferred subnetworks the reticulations
            are taken from.
        score_fn (Callable): ``network -> log pseudo-likelihood``. Higher is
            better.
        min_gain (float): Log pseudo-likelihood a candidate must add on top
            of a strict improvement. Default 0.
        references (Sequence[dict | None] | None): Per network, the in-edge
            its displayed tree kept at each reticulation, from
            :func:`select_displayed_trees_with_choices`. None reads every
            network on its major tree.

    Returns:
        tuple[Network, list[ReticTrial]]: The tree with accepted reticulations
        and every trial.

    Raises:
        RuntimeError: If ``score_fn`` on the starting tree is not finite.
    """
    network, trials = _core._dp_place_reticulations(
        tree, networks, score_fn, min_gain, references,
    )
    return network, trials


def _near_accepted(
    net: Network,
    anchor: Node,
    child: frozenset[str],
    accepted: Sequence[frozenset[str]],
) -> bool:
    """Whether ``anchor`` is within two edges of a reticulation ``child`` overlaps.

    Accepted events are leaf sets. Each one is resolved on ``net``, which is
    the graph ``anchor`` belongs to.
    """
    return _core._dp_near_accepted(net, anchor, child, accepted)


# ── Prune reattachment ────────────────────────────────────────────────


def _attachment_rank(scaffold: Optional[Network], prune: PrunedSubtree) -> tuple:
    """Higher means closer to the anchor, and therefore attached later (inner)."""
    return _core._dp_attachment_rank(scaffold, prune)


def _place_stuck_prune(
    net: Network,
    waiting: list[PrunedSubtree],
    attach: Callable[[PrunedSubtree, Network, Node], Network],
) -> tuple[Network, list[PrunedSubtree]]:
    """DIMPLE's host search for a reattachment pass that placed nothing.

    Every waiting clade is matched against every host: the blob, then each
    other waiting clade. The pair sharing the most anchor leaves wins (ties
    keep the first found), and the clade is attached inside that host at the
    MRCA of the shared leaves. A clade attached inside another waiting clade
    rides along when that clade is placed. One placement per call. Mirrors
    the generalized host search in ``add_pruned_subtree.add_pruned``.

    Args:
        net (Network): Blob with the clades placed so far.
        waiting (list[PrunedSubtree]): Clades no pass could place.
        attach (Callable): ``attach(prune, host, anchor)`` from
            :func:`reattach_prunes`.

    Returns:
        tuple[Network, list[PrunedSubtree]]: The blob and the clades still
        waiting, with a host clade replaced by its enlarged copy.

    Raises:
        DimpleError: If no host shares a leaf with any waiting anchor.
    """
    net, waiting = _core._dp_place_stuck_prune(net, list(waiting), attach)
    return net, waiting


def reattach_prunes(
    blob: Network,
    prunes: Sequence[PrunedSubtree],
    scaffold: Optional[Network] = None,
) -> Network:
    """Hang each pruned clade back on ``blob``.

    A clade is a sibling of the MRCA of its cut-time siblings. If that node
    is a reticulation, the clade goes under its child, inside the hybrid.
    The anchor being the root creates a new root rather than dropping the
    clade. A blob-item anchor that shares no leaf with the blob, or an empty
    anchor, raises. Two clades on the same anchor attach outer-first, so the
    cut closer to the anchor on ``scaffold`` ends up the inner sibling. When
    a pass attaches nothing, DIMPLE's host search (:func:`_place_stuck_prune`)
    places one clade wherever most of its anchor is found, on the blob or
    inside another waiting clade.

    Args:
        blob (Network): Merged blob network.
        prunes (Sequence[PrunedSubtree]): Clades to put back.
        scaffold (Network | None): Tree of blobs, used only to order clades
            that claim the same anchor.

    Returns:
        Network: ``blob`` plus every pruned clade.

    Raises:
        DimpleError: If a clade cannot be attached, or its anchor is empty
            or disjoint from the blob when it was supposed to land there.
    """
    return _core._dp_reattach_prunes(blob, prunes, scaffold)


# ── Scaffold graft ────────────────────────────────────────────────────


def _remove_clade(net: Network, leaves: frozenset[str]) -> None:
    """Delete the clade whose leaf set is ``leaves``, then contract degree-2 nodes."""
    _core._dp_remove_clade(net, leaves)


def _rehang_inner(result: Network, graft_root: Node, saved: list[GraftPiece]) -> None:
    """Put detached inner blobs back inside a freshly grafted outer blob.

    Mirrors ``merge_to_tob.merge``: at each step the inner blob with the most
    anchor leaves currently on ``result`` goes first, since another inner
    blob may hold the leaves this one anchors on. It hangs as a sibling of
    the MRCA of the anchor leaves that are present (not necessarily all of
    them). With none present it hangs beside the first child of the graft
    root, or under the root when it has no children, so the leaves are kept.

    Args:
        result (Network): Network being assembled, mutated.
        graft_root (Node): Image of the outer blob's root on ``result``.
        saved (list[GraftPiece]): Inner blobs that were inside the replaced
            clade.
    """
    _core._dp_rehang_inner(result, graft_root, saved)


def graft_blobs(scaffold: Network, pieces: Sequence[GraftPiece]) -> Network:
    """Replace each blob clade on ``scaffold`` with its combined network.

    The deepest blob is replaced first. An inner piece already grafted inside
    an outer clade is detached, the outer clade is replaced, and the inner
    piece is hung back on its anchor inside the new clade by
    :func:`_rehang_inner`, most-available anchor first, as DIMPLE's
    ``merge_to_tob`` does. A leaf of a piece that is still on the scaffold
    after the clade is removed is an error.

    Args:
        scaffold (Network): Tree of blobs, outgroup already removed.
        pieces (Sequence[GraftPiece]): Combined blobs.

    Returns:
        Network: The full-taxon network.

    Raises:
        DimpleError: If a piece repeats a leaf that is still on the scaffold,
            or its clade cannot be found.
    """
    return _core._dp_graft_blobs(scaffold, pieces)


# ── Per-blob merge ────────────────────────────────────────────────────


def _msc_scorer(gene_trees: GeneTrees, taxa: frozenset[str]) -> Callable[[Network], float]:
    """Log pseudo-likelihood on ``taxa``. Higher is better.

    A non-finite value is returned as ``-inf``. Any other failure from the
    triplet scorer propagates, so a broken reticulation is not reported as
    a topology that simply scored badly.

    Args:
        gene_trees (GeneTrees): Full gene trees. Trees that omit taxa are
            restricted to ``taxa``.
        taxa (frozenset[str]): Leaves of the network being scored.

    Returns:
        Callable[[Network], float]: The scorer.
    """
    from .infer import compute_gene_tree_triplets, score_species_network_triplets
    from ._mpl import _HAS_CYTHON_MPL, _TripleDPEngine, _score_with_cython

    subset = restrict_gene_trees(gene_trees, sorted(taxa))
    table = compute_gene_tree_triplets(
        subset, subset.resolved_mapping(), species_labels=sorted(taxa),
    )

    def score(net: Network) -> float:
        """Log pseudo-likelihood of ``net`` on the subset's triplet table."""
        _populate_default_branch_lengths(net)
        value = None
        if _HAS_CYTHON_MPL:
            # The compiled scorer the subset searches use. It refuses
            # networks past its compiled limits (nodes, reticulations);
            # the Python scorer takes those.
            try:
                value = float(_score_with_cython(
                    _TripleDPEngine(net), table.triplets, table.rho_by_triplet,
                ))
            except ValueError as exc:
                if "compiled limit" not in str(exc):
                    raise
        if value is None:
            value = float(score_species_network_triplets(net, table).log_pseudo_likelihood)
        if not math.isfinite(value):
            return float("-inf")
        return value

    return score


def merge_blob_networks(
    networks: Sequence[Network],
    gene_trees: GeneTrees,
    external_siblings: frozenset[str],
    required_taxa: frozenset[str] = frozenset(),
    model_selection: Optional[str] = None,
    trace: Optional[dict] = None,
    tree_sources: Optional[Sequence[Network]] = None,
    backbone: Optional[Network] = None,
) -> tuple[Network, int, list[ReticTrial]]:
    """Displayed trees, constrained neighbor-joining, then reticulations.

    A single network that already carries every required taxon is returned
    as itself: there is nothing to reconcile. The returned relaxation count
    is zero in that case.

    Args:
        networks (Sequence[Network]): Inferred subnetworks of one blob.
        gene_trees (GeneTrees): Full gene trees.
        external_siblings (frozenset[str]): Scaffold leaves outside the blob,
            used only to place the virtual outgroup.
        required_taxa (frozenset[str]): Blob taxa that must be in the result
            even if no subnetwork carries them (a subset too small to
            infer). They join neighbor-joining as unconstrained leaves.
        model_selection (str | None): ``gain``, ``aic`` or ``bic`` make
            the merger require that gain of each reticulation
            (:func:`selection_min_gain`); ``elbow`` and None keep DIMPLE's
            any-improvement rule.
        trace (dict | None): When given, filled with the intermediate
            results for diagnostics: ``displayed`` (one tree per input),
            ``compatible`` (the trees that reached neighbor-joining),
            ``backbone`` (the merged tree before reticulations), and
            ``candidates`` (the reticulation records tried).
        tree_sources (Sequence[Network] | None): Diagnostic override. Trees
            to use as the displayed trees instead of picking one per
            network by triple support; reticulations are still taken from
            ``networks``.
        backbone (Network | None): Diagnostic override. A tree on the blob
            taxa to place reticulations on, skipping displayed-tree
            selection, the compatibility filter and neighbor-joining.

    Returns:
        tuple[Network, int, list[ReticTrial]]: Merged tree-plus-reticulations,
        neighbor-joining relaxations, and reticulation trials.
    """
    merged, relaxations, trials = _core._dp_merge_blob_networks(
        networks, gene_trees, external_siblings, required_taxa,
        model_selection, trace, tree_sources, backbone,
    )
    return merged, relaxations, trials


# ── Entry point ───────────────────────────────────────────────────────


def infer_dimple(
    gene_trees: GeneTrees,
    tree_of_blobs: Network,
    start_tree: Network,
    *,
    max_subset_size: int = 12,
    n_divisions: int = 15,
    max_reticulations: int = 1,
    seed: int = 0,
    outgroup: Optional[str] = None,
    model_selection: Optional[str] = None,
    true_network: Optional[Network] = None,
    score_final: bool = True,
    **search,
):
    """Infer a network by dividing a tree of blobs and stitching the pieces.

    The scaffold is partitioned at its polytomies. Each subset is inferred
    with MSC triplet pseudo-likelihood (:func:`phynetpy.infer.infer`), and
    the subset networks are merged back onto the scaffold. This is not a
    criterion and it is not dispatched through the inference registry.

    Each subset is inferred by simulated annealing: :data:`SUBSET_RUNS`
    runs of :data:`SUBSET_ITERS` iterations, and the highest log
    pseudo-likelihood is kept. Extra keyword arguments are forwarded to
    each subset ``infer`` call and override those defaults (``num_iter``,
    ``n_runs``, ``method``, ``t_start``). ``seed`` seeds only the
    partition shuffle. The search seeds are ``0, 1, 2`` unless a
    ``seed`` keyword is passed in the forwarded arguments.

    Args:
        gene_trees (GeneTrees): Rooted gene trees on the full taxon set.
            Individual trees may omit taxa. They are not rerooted, and no
            outgroup leaf is added to them.
        tree_of_blobs (Network): Rooted scaffold. A node with more than two
            children is a blob.
        start_tree (Network): Binary tree on the same taxa, used as the
            pseudo-likelihood start on each subset.
        max_subset_size (int): Largest subset. Default 12, minimum 4.
        n_divisions (int): Alternative partitions of each blob. Default 15.
        max_reticulations (int): Cap on each subset search. The merge can
            still keep every reticulation the subsets returned, so the
            stitched network can contain more than this many.
        seed (int): Seed for the partition packings and the enumeration
            shuffle. The first division of every blob is deterministic.
        outgroup (str | None): Leaf to drop from the scaffold before
            partitioning. Not inserted into the gene trees. ``None`` leaves
            the scaffold unchanged.
        model_selection (str | None): Choose each subset's reticulation
            count by ``gain`` (recommended; a step must add more than
            :data:`SELECTION_GAIN_THRESHOLD`), ``aic``, ``bic``, or
            ``elbow`` over caps 0 to ``max_reticulations``, each cap
            warm-started from the one below, and, for ``gain``, ``aic`` and
            ``bic``, make the merger require the same gain of each
            reticulation. None keeps the fixed cap and DIMPLE's
            any-improvement rule.
        true_network (Network | None): Ground-truth network. When given,
            each subset's cap is the number of detectable reticulations in
            the true network restricted to the subset
            (:func:`true_reticulation_bound`), the protocol of the DIMPLE
            paper, and ``max_reticulations`` and the subset part of
            ``model_selection`` are not used. For benchmarks only.
        score_final (bool): Score the stitched network (a triplet table on
            every taxon; seconds to minutes on large data). False leaves
            ``score`` None.
        **search: Subset search settings, see :func:`subset_chains`.
            Defaults are simulated annealing, :data:`SUBSET_ITERS` iterations,
            and :data:`SUBSET_RUNS` runs. A passed value replaces the default.
            ``chains=[{"method": "sa", "num_iter": 20000, "n_runs": 5},
            {"method": "hc", "num_iter": 20000, "n_runs": 3}]`` runs a
            portfolio per subset and keeps the best-scoring network.

    Returns:
        InferenceResult: ``best`` is the full-taxon network, ``score`` is its
        log pseudo-likelihood, ``method`` is ``"DIMPLE"``, and ``raw`` is a
        :class:`DimpleTrace`.

    Raises:
        DimpleError: If the inputs are illegal or a stitch step cannot place
            a clade.
        RuntimeError: If a blob's base tree has a non-finite pseudo-likelihood
            while reticulations are being placed.
    """
    from .infer import InferenceResult, score

    validate_inputs(
        gene_trees, tree_of_blobs, start_tree,
        max_subset_size=max_subset_size,
        n_divisions=n_divisions,
        outgroup=outgroup,
    )
    if model_selection is not None and model_selection not in SELECTION_RULES:
        raise DimpleError(
            f"model_selection must be one of {SELECTION_RULES} or None, "
            f"not {model_selection!r}."
        )
    scaffold = strip_outgroup(tree_of_blobs, outgroup)
    # Every later step reads the gene trees on scaffold taxa. Drop the
    # outgroup (and any taxon not on the scaffold) once here instead of
    # inside every subset, blob, and the final score.
    gene_trees = restrict_gene_trees(gene_trees, sorted(_labels(scaffold)))
    if true_network is not None:
        absent = sorted(_labels(scaffold) - _labels(true_network))
        if absent:
            raise DimpleError(
                "The true network lacks scaffold taxa: " + ", ".join(absent[:5]) + "."
            )
    jobs = partition_scaffold(
        scaffold,
        max_subset_size=max_subset_size,
        n_divisions=n_divisions,
        seed=seed,
    )
    blob_results: list[BlobResult] = []
    pieces: list[GraftPiece] = []
    for job in jobs:
        subsets: list[SubsetRecord] = []
        seen_subsets: set[frozenset[str]] = set()
        for division in job.divisions:
            for taxa in division:
                if taxa in seen_subsets or len(taxa) < 3:
                    continue
                seen_subsets.add(taxa)
                if true_network is not None:
                    bound = true_reticulation_bound(true_network, sorted(taxa))
                    found = _infer_subset(gene_trees, start_tree, taxa, bound, search)
                else:
                    found = _infer_subset(
                        gene_trees, start_tree, taxa, max_reticulations, search,
                        model_selection=model_selection,
                    )
                subsets.append(replace(found, blob_name=job.name))
        if not subsets:
            raise DimpleError(f"Blob {job.name} produced no subset to infer.")
        # Every taxon the division gave this blob, including those in a
        # subset too small to infer. They must not fall off the blob.
        required = frozenset().union(
            *(subset for division in job.divisions for subset in division)
        )
        if len(subsets) == 1 and required <= _labels(subsets[0].network):
            merged: Network = subsets[0].network
            relaxations = 0
            trials: list[ReticTrial] = []
        else:
            merged, relaxations, trials = merge_blob_networks(
                [record.network for record in subsets],
                gene_trees,
                job.external_siblings,
                required_taxa=required,
                model_selection=model_selection,
            )
        combined = reattach_prunes(merged, job.pruned, scaffold)
        blob_results.append(BlobResult(job, subsets, combined, relaxations, trials))
        pieces.append(GraftPiece(job.tob_leaves, combined, job.external_siblings))

    final = graft_blobs(scaffold, pieces)
    final.set_branch_length_unit(BranchLengthUnit.COALESCENT_2N)
    # Score on the leaves the network actually has, so an outgroup that was
    # stripped from the scaffold and is still sitting in a gene tree does not
    # make the score undefined.
    final_score: Optional[float] = None
    if score_final:
        scoring_trees = restrict_gene_trees(gene_trees, sorted(_labels(final)))
        final_score = float(score(
            final, scoring_trees, model=MSC(), criterion=PseudoLikelihood(),
        ))
    trace = DimpleTrace(blob_results, final, final_score)
    return InferenceResult(
        best=final,
        score=final_score,
        method="DIMPLE",
        raw=trace,
    )
