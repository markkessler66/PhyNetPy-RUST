#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared MSNC density: ancestral-configurations DP (Yu, Degnan and Nakhleh 2012).

Used by :mod:`phynetpy._mcmc_gt` and :mod:`phynetpy._seq_likelihood` / MCMC_SEQ.

The dynamic programmes run in the compiled core (``phynetpy._core``): the
topology-only frontier DP (``_msnc_log_prob_network_int``), the timed
frontier DP (``_msnc_log_density_timed``) and the species-tree DP on a
displayed tree (``_msc_log_prob_tree_int``), all over native lineage
bitmasks of whatever width the gene tree needs.  ``MSCBranchKernel``,
``_GeneTreeIndex`` and ``_NetworkIndex`` are the compiled index classes with
their original attribute surface.  The pure-Python branch kernels are kept
as the reference implementation the parity tests compare against.
"""
from __future__ import annotations

import math
from collections import deque
from typing import TYPE_CHECKING, Any, Optional, Sequence

import numpy as np

from .Network import Network, Node
from .GraphUtils import _node_height
from ._units import (
    _positive_theta,
    BranchLengthUnit,
    BranchThetaKey,
    require_branch_length_unit,
    resolve_branch_theta,
    validate_branch_thetas,
)
from ._core import (
    MSCBranchKernel,
    _GeneTreeIndex,
    _NetworkIndex,
    _DisplayedTreeIndex,
    _msnc_log_prob_network_int,
    _msc_log_prob_tree_int,
    _apply_branch_coalescent_int,
    _combine_configs_int,
    _apply_branch_timed,
    _popcount,
    _bits,
    _logsumexp,
)
from ._core import _msnc_log_density_timed as _msnc_log_density_timed_core

if TYPE_CHECKING:
    from . import _mcmc_gt

_LOG_FLOOR: float = math.log(1e-200)

# Compatibility flags from the Cython era: the compiled kernels are always
# present now.  Tests and callers that toggled these can still do so; the
# compiled DPs do not consult them.
_CYTHON_AVAILABLE = True
_CYTHON_NETWORK_DP = True
_CYTHON_TIMED_DP = True

# Kept for callers that consulted the old Cython limit; the compiled DP
# picks a mask width per gene tree and has no such ceiling below 2048 nodes.
_CY_MAX_BITS = 2048


def _gene_coalescence_events(
    gene_tree: Network,
    gti: "_GeneTreeIndex",
    heights: Optional[dict[Node, float]] = None,
) -> list[tuple[float, int, int, int]]:
    """Sorted gene-tree coalescences as (time, parent_bit, child0_bit, child1_bit).

    When ``heights`` (a Node -> height-above-present map covering every node)
    is supplied it is used directly, skipping the recursive ``_node_height``
    walk -- the caller (e.g. the MCMC_SEQ candidate enumerator) already holds
    the exact ultrametric heights it just wrote onto the tree.
    """
    height_cache: dict[Node, float] = {} if heights is None else heights
    events: list[tuple[float, int, int, int]] = []
    for node in gene_tree.V():
        kids = gene_tree.get_children(node)
        if len(kids) == 2:
            if heights is not None:
                h = heights[node]
            else:
                h = _node_height(gene_tree, node, height_cache)
            events.append(
                (
                    h,
                    gti.bit_of[node],
                    gti.bit_of[kids[0]],
                    gti.bit_of[kids[1]],
                )
            )
    events.sort(key=lambda e: e[0])
    return events



def _msnc_log_density_timed(
    net_idx: "_NetworkIndex",
    gti: "_GeneTreeIndex",
    events: list[tuple[float, int, int, int]],
    sp_heights: list[float],
    theta: float,
    branch_thetas: Optional[dict[BranchThetaKey, float]] = None,
) -> float:
    """Joint-edge AC DP with timed (event-based) branch coalescent factors."""
    if gti.root_bit < 0 or not gti.leaves:
        return 0.0
    if net_idx.n_nodes == 0 or net_idx.root < 0:
        return _LOG_FLOOR

    edge_thetas = None
    root_theta = None
    if branch_thetas:
        node_objs = net_idx._node_objs
        edge_src = net_idx.edge_src
        edge_dst = net_idx.edge_dst
        edge_thetas = [
            resolve_branch_theta(
                theta,
                branch_thetas,
                node_objs[edge_dst[e]],
                parent=node_objs[edge_src[e]],
            )
            for e in range(net_idx.n_edges)
        ]
        root_theta = resolve_branch_theta(theta, branch_thetas, node_objs[net_idx.root])
    return _msnc_log_density_timed_core(
        net_idx,
        gti,
        [tuple(e) for e in events],
        list(sp_heights),
        float(theta),
        edge_thetas,
        root_theta,
    )


def build_network_msnc_index(
    species_net: Network,
) -> tuple["_NetworkIndex", Optional[list[float]]]:
    """Build the reusable network view + node-height vector for the timed DP.

    Split out of :func:`gene_tree_msnc_log_density` so a caller (e.g. the
    MCMC_SEQ likelihood engine) can build this *once per species network* and
    reuse it across every locus, instead of rebuilding the identical index and
    recomputing identical node heights once per (locus, network) pair.

    Returns:
        ``(net_idx, sp_heights)`` where ``sp_heights[i]`` is the ultrametric
        height of node ``i`` above the present.  ``sp_heights`` is ``None`` for
        an empty / rootless network (the caller should score such states as
        ``-inf``).
    """
    net_idx = _NetworkIndex(species_net)
    if net_idx.n_nodes == 0 or net_idx.root < 0:
        return net_idx, None
    height_cache: dict[Node, float] = {}
    sp_heights = [
        _node_height(species_net, net_idx._node_objs[i], height_cache)
        for i in range(net_idx.n_nodes)
    ]
    return net_idx, sp_heights


def build_gene_tree_msnc_index(
    gene_tree: Network,
    species_of: dict[str, str],
) -> tuple["_GeneTreeIndex", list[tuple[float, int, int, int]]]:
    """Build the reusable gene-tree view + coalescence-event list.

    Reusable for the lifetime of the (immutable) ``gene_tree`` object; the
    MCMC_SEQ engine caches this per locus keyed by gene-tree identity and only
    rebuilds it when that locus's gene tree is actually mutated.
    """
    gti = _GeneTreeIndex(gene_tree, species_of)
    events = _gene_coalescence_events(gene_tree, gti)
    return gti, events


def build_gene_tree_topology_index(
    gene_tree: Network,
    species_of: dict[str, str],
) -> "_GeneTreeIndex":
    """Build only the *topology* view of ``gene_tree`` (no timed events).

    The index (bit assignments, parent/children, coarsening + linear-extension
    caches) depends solely on topology.  A caller scoring one topology under
    several branch-length rescalings -- e.g. the MCMC_SEQ candidate enumerator
    over :data:`_HEIGHT_SCALE_GRID` -- can build this once and pair it with a
    fresh :func:`gene_tree_events` list per rescaling, so the (expensive)
    coarsening cache is populated once and reused across scales.
    """
    return _GeneTreeIndex(gene_tree, species_of)


def gene_tree_events(
    gene_tree: Network,
    gti: "_GeneTreeIndex",
    heights: Optional[dict[Node, float]] = None,
) -> list[tuple[float, int, int, int]]:
    """Timed coalescence events for ``gene_tree`` against a prebuilt ``gti``.

    Only this list changes when a gene tree's branch lengths (node heights)
    are rescaled; the topology :class:`_GeneTreeIndex` is invariant.  Pass
    ``heights`` (Node -> height-above-present) to reuse heights the caller
    already computed instead of re-deriving them from edge lengths.
    """
    return _gene_coalescence_events(gene_tree, gti, heights)


def msnc_log_density_prebuilt(
    net_idx: "_NetworkIndex",
    sp_heights: Optional[list[float]],
    gti: "_GeneTreeIndex",
    events: list[tuple[float, int, int, int]],
    theta: float,
    branch_thetas: Optional[dict[BranchThetaKey, float]] = None,
) -> float:
    """Timed MSNC density from pre-built indices (hot MCMC_SEQ entry point).

    Equivalent to :func:`gene_tree_msnc_log_density` but takes the network /
    gene-tree views and node-height vector the caller has already built and
    cached.  Applies the same ``-inf`` floor for incompatible embeddings.
    """
    if sp_heights is None or net_idx.n_nodes == 0 or net_idx.root < 0:
        return float("-inf")
    score = _msnc_log_density_timed(
        net_idx,
        gti,
        events,
        sp_heights,
        theta,
        branch_thetas,
    )
    if score <= _LOG_FLOOR + 1:
        return float("-inf")
    return float(score)


def gene_tree_msnc_log_density(
    gene_tree: Network,
    species_net: Network,
    species_of: dict[str, str],
    *,
    theta: float = 0.02,
    branch_thetas: Optional[dict[BranchThetaKey, float]] = None,
) -> float:
    """Log timed MSNC density log P(g | Psi) via joint-edge AC DP."""
    _positive_theta(theta, "default theta")
    validate_branch_thetas(branch_thetas)
    require_branch_length_unit(
        species_net,
        BranchLengthUnit.SUBSTITUTIONS_PER_SITE,
        context="timed MSNC density",
    )
    require_branch_length_unit(
        gene_tree,
        BranchLengthUnit.SUBSTITUTIONS_PER_SITE,
        context="timed MSNC density",
    )
    net_idx, sp_heights = build_network_msnc_index(species_net)
    if sp_heights is None:
        return float("-inf")
    gti, events = build_gene_tree_msnc_index(gene_tree, species_of)
    return msnc_log_density_prebuilt(
        net_idx,
        sp_heights,
        gti,
        events,
        theta,
        branch_thetas,
    )


__all__ = [
    "MSCBranchKernel",
    "gene_tree_msnc_log_density",
    "_LOG_FLOOR",
    "_GeneTreeIndex",
    "_NetworkIndex",
    "_msnc_log_prob_network_int",
    "_msc_log_prob_tree_int",
    "_apply_branch_coalescent_int",
    "_combine_configs_int",
    "_logsumexp",
    "_popcount",
]


def _safe_add(a: float | None, b: float | None) -> float | None:
    """Add two possibly-``None`` branch lengths.

    Treats ``None`` as "no finite length" (the root edge / infinite
    branch); summing ``None`` with anything yields ``None``.  For
    finite inputs the usual sum applies.
    """
    if a is None or b is None:
        return None
    return float(a) + float(b)


def _apply_branch_coalescent_int_py(
    config_in: dict[int, float],
    length: "float | None",
    gti: "_GeneTreeIndex",
    engine: "MSCBranchKernel",
) -> dict[int, float]:
    """Pure-Python bitmask version of :func:`_apply_branch_coalescent`.

    Hot inner loop of the MSC DP.  All length-independent quantities
    -- popcounts, ``log(linear_extensions)``, the merge-mask itself
    -- come pre-baked from :meth:`_GeneTreeIndex.coarsenings`; only
    ``log(g_ij)`` and ``log_denom`` (both functions of branch length)
    are looked up each call, against engine-level caches.

    The Cython extension :mod:`phynetpy.cython.gt_msc_cy` provides a
    drop-in replacement bound to :func:`_apply_branch_coalescent_int`
    when available; this Python copy is the fallback (and the
    ground truth used by the unit tests when the extension is
    rebuilt).
    """
    out: dict[int, list[float]] = {}
    coarsen = gti.coarsenings
    log_gij = engine._log_gij
    log_denom = engine._log_denom
    log_floor = _LOG_FLOOR
    for cfg_in, lp_in in config_in.items():
        n = _popcount(cfg_in)
        for cfg_out, merge_mask, m, k, log_le in coarsen(cfg_in):
            log_branch = log_gij(length, n, m)
            if log_branch <= log_floor:
                continue
            if k > 0:
                log_branch += log_le - log_denom(n, k)
            log_total = lp_in + log_branch
            existing = out.get(cfg_out)
            if existing is None:
                out[cfg_out] = [log_total]
            else:
                existing.append(log_total)
    result: dict[int, float] = {}
    for cfg, terms in out.items():
        if len(terms) == 1:
            result[cfg] = terms[0]
        else:
            result[cfg] = _logsumexp(terms)
    return result


def _combine_configs_int_py(
    left: dict[int, float],
    right: dict[int, float],
) -> dict[int, float]:
    """Pure-Python outer-product disjoint-union combine.

    Disjoint-union of bitmasks; entries with overlapping bits are
    skipped (mapping ill-formed) and same-union duplicates merged
    via log-sum-exp.  See :func:`_apply_branch_coalescent_int_py`
    for the Cython-acceleration story.
    """
    out: dict[int, list[float]] = {}
    for cfg_l, lp_l in left.items():
        for cfg_r, lp_r in right.items():
            if cfg_l & cfg_r:
                continue
            union = cfg_l | cfg_r
            existing = out.get(union)
            if existing is None:
                out[union] = [lp_l + lp_r]
            else:
                existing.append(lp_l + lp_r)
    result: dict[int, float] = {}
    for cfg, terms in out.items():
        if len(terms) == 1:
            result[cfg] = terms[0]
        else:
            result[cfg] = _logsumexp(terms)
    return result



def _frontier_insert(
    tup: tuple[tuple[int, int], ...],
    item: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    """Insert ``item`` into a sorted-tuple frontier key, preserving order."""
    if not tup:
        return (item,)
    eid = item[0]
    out = list(tup)
    pos = 0
    while pos < len(out) and out[pos][0] < eid:
        pos += 1
    out.insert(pos, item)
    return tuple(out)


def _frontier_acc(
    frontier: dict[tuple[tuple[int, int], ...], float],
    key: tuple[tuple[int, int], ...],
    lp: float,
) -> None:
    """Accumulate ``lp`` into ``frontier[key]`` via log-sum-exp."""
    existing = frontier.get(key)
    if existing is None:
        frontier[key] = lp
    elif existing > lp:
        frontier[key] = existing + math.log1p(math.exp(lp - existing))
    else:
        frontier[key] = lp + math.log1p(math.exp(existing - lp))