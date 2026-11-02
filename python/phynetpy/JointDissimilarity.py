#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author : Mark Kessler
Last Edit : 10/29/26
First Included in Version : 0.7.0

JointDissimilarity -- a joint reticulation-and-tree dissimilarity for rooted
phylogenetic networks.

For two networks N1, N2 on the same taxon set, with r1 and r2 reticulations
and ``m = max(r1, r2)``:

* **Tripartitions.** Every reticulation h gives ``(A, B, C)``: B is the block
  below h, A and C the blocks below h's two parents once h is deleted, pruned
  to the vertices on a path from that parent to a taxon.  Two tripartitions
  cost ``delta(t, t') = d(B, B') + min(d(A, A') + d(C, C'), d(A, C') + d(C, A'))``
  in ``[0, 3]``.
* **Reticulation term.** The smaller collection is padded with dummies to m
  (real--dummy cost 3) and ``D_ret`` is the minimum over bijections of the
  summed cost, by one Hungarian solve.
* **Remainder.** ``T(N)`` deletes the union of the blocks below all
  reticulations and hangs one marker leaf per surviving parent edge, labelled
  by the reticulation's descendant taxa.  ``D_tree = d(T(N1), T(N2))``.
* **Joint.** ``D_joint = D_ret + 3 D_tree``; normalised forms ``D_ret / 3m``
  (0 when m = 0) and ``D_joint / 3(m + 1)``, both in ``[0, 1]``.

The structure dissimilarity ``d(S, S')`` in ``[0, 1]`` is one of three
instantiations, each comparing the COMPLETE structures (no restriction to
shared leaves, no weighting; both empty -> 0, exactly one empty -> 1):

===========  ==========================================================
``leaf``     Jaccard distance of the leaf-label sets
``cluster``  multiset hardwired-cluster dissimilarity, one cluster per
             vertex
``nakhleh``  ``L / U`` over Nakhleh's vertex-equivalence classes, with
             ``L = sum_q |c_S(q) - c_S'(q)|`` and
             ``U = sum_q max(c_S(q), c_S'(q))``
===========  ==========================================================

Nothing is ever suppressed (a parent left unary by a deletion stays), and two
marker leaves may share a label, which is what makes the structural
instantiations well defined on the multi-labelled remainder.  Everything after
validation runs on a small rooted DAG detached from
:class:`~.Network.Network`, so no network is mutated or copied.

The computation runs in the compiled core (``phynetpy._core``):
``_jd_joint_dissimilarity`` validates the two networks, builds their
tripartitions and remainders, evaluates the chosen structure dissimilarity and
solves the padded matching (a port of SciPy's ``linear_sum_assignment``) in
Rust.  This module is its public face; :class:`JointDissimilarityResult` stays
a Python dataclass.

Reference: Nakhleh, *Reticulation-Aware Dissimilarity for Phylogenetic
Networks* (2026 manuscript, joint reticulation-and-tree form); Nakhleh,
*A metric on the space of reduced phylogenetic networks*, IEEE/ACM TCBB 7
(2010) 218--222 for the vertex-equivalence classes.

Docs   - [x]
Tests  - [x]
Design - [x]
"""

from __future__ import annotations

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

from dataclasses import dataclass

from .Network import Network, Node, NetworkError
from . import _core

__all__ = ["JOINT_KINDS", "JointDissimilarityResult", "joint_dissimilarity"]

#: The three instantiations of the structure dissimilarity ``d(S, S')``.
JOINT_KINDS = ("leaf", "cluster", "nakhleh")

#: Cost of pairing a real tripartition with a padding dummy: the maximum of
#: ``delta``, three block comparisons each in ``[0, 1]``.
_DUMMY_COST = 3.0


# ── Public surface ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class JointDissimilarityResult:
    """
    The result of :func:`joint_dissimilarity` under one instantiation.

    Args:
        kind (str): The instantiation used, one of :data:`JOINT_KINDS`.
        r1 (int): Reticulation number of the first network.
        r2 (int): Reticulation number of the second network.
        D_ret (float): Reticulation term, raw, in ``[0, 3m]``.
        D_tree (float): Remainder term ``d(T(N1), T(N2))`` in ``[0, 1]``.
    """

    kind: str
    r1: int
    r2: int
    D_ret: float
    D_tree: float

    @property
    def m(self) -> int:
        """
        Returns:
            int: ``max(r1, r2)``, the size of the padded matching.
        """
        return max(self.r1, self.r2)

    @property
    def D_joint(self) -> float:
        """
        Returns:
            float: ``D_ret + 3 D_tree``, raw, in ``[0, 3(m + 1)]``.
        """
        return self.D_ret + 3.0 * self.D_tree

    @property
    def D_ret_normalized(self) -> float:
        """
        Returns:
            float: ``D_ret / 3m`` in ``[0, 1]``; 0 when both are trees.
        """
        return self.D_ret / (3.0 * self.m) if self.m else 0.0

    @property
    def D_joint_normalized(self) -> float:
        """
        Returns:
            float: ``D_joint / 3(m + 1)`` in ``[0, 1]``.
        """
        return self.D_joint / (3.0 * (self.m + 1))


def joint_dissimilarity(net1: Network,
                        net2: Network,
                        kind: str = "nakhleh") -> JointDissimilarityResult:
    """
    The joint reticulation-and-tree dissimilarity of two rooted phylogenetic
    networks on the same taxon set (see the module docstring for the
    definition).  Symmetric, 0 on identical networks, and a metric on the
    tripartition and remainder representations under each instantiation.

    Both networks must have one root, be acyclic, have labelled leaves with no
    label used twice, and give every reticulation exactly two distinct parents
    and one child.  Internal-node labels and branch lengths are ignored.

    Args:
        net1 (Network): First network.
        net2 (Network): Second network.
        kind (str): Structure dissimilarity: ``"leaf"`` (Jaccard of leaf
                    sets), ``"cluster"`` (multiset hardwired clusters) or
                    ``"nakhleh"`` (vertex-equivalence classes; the default).

    Raises:
        ValueError: If ``kind`` is not one of :data:`JOINT_KINDS`.
        NetworkError: If either network violates the requirements above, or
                      the two taxon sets differ.

    Returns:
        JointDissimilarityResult: ``r1``, ``r2``, ``D_ret`` and ``D_tree``, with
                            ``m``, ``D_joint`` and the normalised forms as
                            properties.
    """
    r1, r2, d_ret, d_tree = _core._jd_joint_dissimilarity(net1, net2, kind)
    return JointDissimilarityResult(kind, r1, r2, d_ret, d_tree)
