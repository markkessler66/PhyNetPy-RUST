#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author : Mark Kessler
Last Edit : 5/12/26
First Included in Version : 1.0.0

GraphUtils -- topology analysis, manipulation, and rendering utilities
for phylogenetic networks.

This module is the catch-all for graph-theoretic operations on
:class:`~.Network.Network` objects, including:

* topology summaries (network level, blob/biconnected-component
  decomposition, tree-child / tree-based predicates);
* enumeration of displayed trees and subnetworks
  (``get_displayed_trees``, ``get_all_subtrees``, ``subnet_given_leaves``,
  ``induced_subnetwork_by_taxa``);
* distance metrics on networks
  (mu-distance, hardwired / softwired / Robinson--Foulds /
  tripartition / displayed-tree / average-path / weighted-APD, and the
  joint reticulation-and-tree dissimilarity of ``JointDissimilarity``);
* convenience helpers for branch lengths, clades, and ASCII rendering.

The graph algorithms run in the compiled core (``phynetpy._core``); this
module is their public face.  Every function keeps its signature and its
documentation, and delegates to the matching ``_gu_*`` kernel.  The ASCII
renderer stays in Python.

Docs   - [x]
Tests  - [ ]
Design - [ ]
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

from .Network import Network, Node, Edge, NetworkError
from .JointDissimilarity import JOINT_KINDS, JointDissimilarityResult, joint_dissimilarity
from collections import deque
from typing import Dict, Optional, Set, Union
from . import _core

#: Public surface of this module.  Declared explicitly so that
#: ``from .GraphUtils import *`` exports only these helpers and not the
#: ``Network`` names re-imported above.  The reticulation-comparison metrics
#: appended at the bottom of this file are listed here too.
__all__ = [
    # Structural edits
    "add_hybrid",
    # Topology extraction / sub-structures
    "subnet_given_leaves",
    "induced_subnetwork_by_taxa",
    "get_displayed_trees",
    "get_all_subtrees",
    "get_all_clusters",
    "network_clusters",
    "dominant_tree",
    "count_displayed_trees",
    # Decomposition
    "blobs",
    "tree_of_blobs",
    "bridges_and_articulations",
    # Descriptive properties
    "level",
    "is_tree",
    "count_reticulations",
    "reticulation_cycle_size",
    "is_isomorphic",
    # Distances / dissimilarities
    "hardwired_cluster_distance",
    "softwired_cluster_distance",
    "displayed_tree_distance",
    "robinson_foulds_distance",
    "tripartition_distance",
    "mu_distance",
    "joint_dissimilarity",
    "JointDissimilarityResult",
    "JOINT_KINDS",
    "hybrid_ancestry",
    "hybrid_ancestry_error",
    "pairwise_leaf_distance",
    "average_path_distance",
    "weighted_average_path_distance",
    # Reticulation-aware comparison (re-exported from ReticulationComparison)
    "JACCARD",
    "SYMMETRIC",
    "ReticulationTripartition",
    "NetworkComparison",
    "reticulation_tripartitions",
    "set_distance",
    "tripartition_dissimilarity",
    "nakhleh_metric",
    "reticulation_dissimilarity",
    "reticulation_precision_recall",
    "combined_dissimilarity",
    "compare_networks",
    # Rendering
    "ascii",
    # DIMPLE-style reticulation identity and rooted triples
    "reticulation_signature",
    "same_reticulation",
    "group_reticulations",
    "rooted_triples",
]

def get_all_clusters(net : Network,
                     include_trivial : bool = False)\
                     -> set[frozenset[Node]]:
    """
    Compile the set of clusters that make up this network.
    Ie: for a graph ((A, B)C, D); , set of all clusters is {(A,B), (A,B,C)}.
    
    A cluster is the set of leaves below some non-root node. Clusters of
    size 1 can optionally be included.

    Note that clusters are sets of :class:`~phynetpy.Network.Node` objects
    belonging to *this* network, so they cannot be compared across networks;
    use :func:`network_clusters` for that.

    Args:
        net (Network): the network to operate on
        include_trivial (bool): If set to True, includes clusters of size 1. 
                                Defaults to False.

    Returns:
        set: A set of all clusters in this graph, each a frozenset of Nodes.
    """
    return _core._gu_get_all_clusters(net, include_trivial)

def _induced_subnetwork(net : Network, targets : set[Node]) -> Network:
    """
    Copy the part of 'net' at or below MRCA(targets) that leads to a target.

    Shared core of :func:`subnet_given_leaves` and
    :func:`induced_subnetwork_by_taxa`; the caller decides which
    :meth:`Network.clean` passes to run on the result.

    Each retained node is enqueued exactly once. A reticulation reachable
    from two retained parents would otherwise have its whole subtree walked
    once per parent, and since Edge equality is by identity, every edge below
    it would be copied twice into a spurious bubble.

    Args:
        net (Network): A network object.
        targets (set[Node]): The leaves the subnetwork must retain.
    Returns:
        Network: An uncleaned copy of the induced subnetwork.
    """
    return _core._gu_induced_subnetwork(net, targets)

def subnet_given_leaves(net : Network, leaf_set : list[Node]) -> Network:
    """
    Compute the minimally sized subnetwork of a network such that the leaf set 
    of the subnetwork is a subset of the original network's leaf set.

    Args:
        net (Network): A network
        leaf_set (list[Node]): A set of leaf nodes of the given network

    Returns:
        Network: A new Network object with node and edge copies of the original.
    """
    return _core._gu_subnet_given_leaves(net, leaf_set)

def _suppress_childless_nodes(tree : Network, taxa : set[str]) -> None:
    """
    Delete out-degree-0 nodes that are not leaves of the original network.

    Dropping the unused in-edges of a reticulation can leave a tree node with
    no children at all. In
    ``(t2,((t4,t6),(((t3)#H1,(t1)#H2),((t5,#H1),#H2))));`` the node whose only
    children are ``#H1`` and ``#H2`` keeps neither of them when both
    reticulations inherit from their *other* parent. Such a node has
    out-degree 0 but stands for no taxon, so a displayed tree has to suppress
    it -- and suppressing it can orphan its own parent in turn, hence the
    worklist rather than a single pass.

    Args:
        tree (Network): A network being reduced to a displayed tree. Mutated
                        in place.
        taxa (set[str]): Labels of the leaves of the network the tree is
                         displayed by. Nodes carrying these labels are kept
                         however few children they have.
    Returns:
        N/A
    """
    _core._gu_suppress_childless_nodes(tree, taxa)

def _displayed_tree(net : Network,
                    dropped : set[Edge],
                    taxa : set[str]) -> Network:
    """
    Build the tree that remains once 'dropped' hybrid edges are discarded.

    Copies the whole node set and every edge outside 'dropped', then reduces
    the result to a tree on 'taxa': childless internal nodes are suppressed,
    degree-2 chains are contracted, and the reticulation flag is cleared on
    every node, since nothing in a tree has in-degree 2 any more.

    The copy is built edge by edge instead of by pruning
    :meth:`Network.copy`, because looking an edge up by its (src, dest) pair
    cannot tell the two edges of a bubble apart.

    Args:
        net (Network): The network the tree is displayed by.
        dropped (set[Edge]): Edges of 'net' to leave out, identified by
                             object identity.
        taxa (set[str]): Labels of the leaves of 'net'.
    Returns:
        Network: The displayed tree, with fresh Node and Edge objects.
    """
    return _core._gu_displayed_tree(net, dropped, taxa)

def _displayed_trees_with_probs(net : Network) -> list[tuple[Network, float]]:
    """
    Enumerate one displayed tree per hybrid-edge choice, with its probability.

    A displayed tree is obtained by keeping exactly one in-edge at each
    reticulation; its probability is the product of the gammas of the kept
    edges. When gamma is unset for an edge, a uniform 1/k weight is assumed
    (k = in-degree of the reticulation node).

    There is one entry per element of the product of the reticulations'
    in-edge sets, so the same topology can appear more than once: two choices
    that differ only at a reticulation whose subtree is discarded anyway
    display the same tree. :func:`get_displayed_trees` collapses those;
    callers that weight trees by probability need them kept apart.

    Reticulations are sorted by label and each one's in-edges by gamma,
    highest first (then by source label), so the first tree enumerated is
    the major tree and the order is deterministic across runs. A caller
    that breaks ties by taking the first candidate therefore resolves them
    toward the major tree, the same way for every network of a family.

    A displayed tree has exactly the network's leaves: an internal node left
    without children by a dropped edge (as happens in a network restricted
    to a subset of its taxa) is removed, and degree-2 chains are contracted
    (:func:`_displayed_tree`).

    Args:
        net (Network): A phylogenetic network.
    Returns:
        list[tuple[Network, float]]: (tree, probability) pairs.
    """
    return _core._gu_displayed_trees_with_probs(net)

def _tree_topology_key(tree : Network) -> frozenset[frozenset[str]]:
    """
    Canonical key identifying the topology of a rooted tree.

    A rooted tree with no degree-2 nodes is determined by the leaf sets below
    its nodes, so two trees share a key exactly when they have the same
    topology. Internal node labels and branch lengths are deliberately
    ignored: displayed trees inherit internal names from whichever hybrid
    edges survived, so the same topology reached two ways is named two ways.

    Args:
        tree (Network): A tree.
    Returns:
        frozenset[frozenset[str]]: The tree's clusters, as leaf label sets.
    """
    return _core._gu_tree_topology_key(tree)

def get_displayed_trees(net : Network,
                        unique : bool = True) -> list[Network]:
    """
    Enumerate the trees displayed by a phylogenetic network.

    A tree is displayed by a network when it can be obtained by keeping
    exactly one in-edge at every reticulation, deleting the rest, and then
    reducing what is left to a tree over the network's full leaf set:

    * nodes left without children are suppressed, repeatedly, since
      suppressing one can leave its parent childless;
    * chains of degree-2 nodes are contracted into single branches;
    * the reticulation flag is cleared, so :func:`is_tree` holds of every
      result.

    Every displayed tree has exactly the leaf set of 'net'. Removing in-edges
    of reticulations never disconnects a leaf, because each reticulation keeps
    one parent and no other node loses its only one.

    A network with k reticulations admits at most 2^k displayed trees, but
    often fewer distinct ones: when the choices at two reticulations discard
    each other's subtrees, different choices display the same tree. In
    ``(t2,((t4,t6),(((t3)#H1,(t1)#H2),((t5,#H1),#H2))));`` two of the four
    choices display ``(t2,((t4,t6),(t1,(t5,t3))));``, so the network displays
    three trees, not four.

    Args:
        net (Network): A phylogenetic network.
        unique (bool): If True (the default), return one Network per distinct
                       topology. If False, return one per hybrid-edge choice,
                       so that the length of the result is the product of the
                       reticulations' in-degrees.
    Returns:
        list[Network]: The displayed trees, as new Network objects. Order is
                       deterministic: reticulations and their in-edges are
                       enumerated by label, and duplicates are dropped in
                       favour of the first occurrence.
    """
    return _core._gu_get_displayed_trees(net, unique)

def get_all_subtrees(net : Network) -> list[Network]:
    """
    Generate one tree per choice of hybrid edges to keep at the reticulations.

    Prefer :func:`get_displayed_trees`, which returns the *distinct* displayed
    trees. This function returns one tree per element of the product of the
    reticulations' in-edge sets, so the same topology can appear twice; see
    :func:`get_displayed_trees` for when that happens.

    Args:
        net (Network): A network object
    Returns:
        list[Network]: A list of network objects, each representing a tree that
                       is derived from the original network.
    """
    return _core._gu_get_displayed_trees(net, False)

def dominant_tree(net : Network) -> Network:
    """
    Generate the dominant tree from a given network by retaining only the 
    reticulation edges with the highest inheritance probability and removing 
    all other reticulation edges.

    Args:
        net (Network): A network object
    Returns:
        Network: A new network object representing the dominant tree derived 
                 from the original network.
    """
    return _core._gu_dominant_tree(net)

def _ancestor_depths(net: Network, start: Node) -> dict[Node, int]:
    """Map every ancestor of ``start`` (inclusive) to its min edge-distance."""
    dist: dict[Node, int] = {start: 0}
    frontier = [start]
    while frontier:
        nxt: list[Node] = []
        for u in frontier:
            for p in net.get_parents(u):
                if p not in dist:
                    dist[p] = dist[u] + 1
                    nxt.append(p)
        frontier = nxt
    return dist


def reticulation_cycle_size(net: Network, retic: Node) -> int:
    """Number of nodes on a reticulation's cycle, in the undirected graph.

    The cycle is ``retic`` plus the two ancestral paths from its parents up
    to their lowest common ancestor. A cycle of 3 means the two parents are
    parent and grandparent of each other, which no rooted triple can
    distinguish from a tree; such a reticulation is invisible to triplet
    pseudo-likelihood. ``2`` means parallel edges (both parents identical)
    and ``0`` means the node is not a proper two-parent reticulation.

    Args:
        net (Network): Network containing ``retic``.
        retic (Node): The reticulation node.

    Returns:
        int: Cycle size, or 0 when ``retic`` has fewer than two parents or
        its parents share no ancestor.
    """
    return _core._gu_reticulation_cycle_size(net, retic)


def count_reticulations(net: Network, min_cycle_size: Optional[int] = None) -> int:
    """
    Count the number of reticulation nodes in a network.

    Args:
        net (Network): A network object.
        min_cycle_size (int | None): When given, count only reticulations
            whose cycle (see :func:`reticulation_cycle_size`) has at least
            this many nodes. ``4`` counts the reticulations a rooted-triple
            criterion can detect; a cycle of 3 or less is invisible to it.

    Returns:
        int: Number of nodes flagged as reticulations, filtered by cycle size
        when ``min_cycle_size`` is given.
    """
    return _core._gu_count_reticulations(net, min_cycle_size)

def add_hybrid(net : Network,
               source : Edge,
               destination : Edge,
               t_src : float = None,
               t_dest : float = None) -> None:
    """
    Add a hybrid (reticulation) edge between two existing edges, raising the
    network level by one.

    This is a structural edit, not a Metropolis-Hastings proposal: it mutates
    ``net`` in place and reports no Hastings ratio. Samplers should use the
    :class:`~.ModelMove.AddReticulation` move instead, which carries the
    reversible-jump proposal math.

    How it works (normal case, ``source != destination``)::

        BEFORE                        AFTER
        a           x                a           x
        |           |                |           |
        |           |                n1 -------> n2   (new hybrid edge)
        |           |                |           |
        b           y                b           y

    Two fresh internal nodes (n1, n2) are inserted by splitting the ``source``
    and ``destination`` edges respectively. n2 is flagged as a reticulation
    node, and a directed edge n1 -> n2 is added.

    When ``source`` and ``destination`` are the *same* edge, two parallel edges
    with equal inheritance probabilities (gamma = 0.5) are created between two
    new internal nodes, modeling a whole-genome duplication ("bubble") event.

    Args:
        net (Network): The network to modify in place.
        source (Edge): The edge that will donate the new outgoing lineage.
        destination (Edge): The edge that will receive the incoming hybrid
                            lineage.
        t_src (float, optional): Divergence time for the new source-side node.
                                 Defaults to None.
        t_dest (float, optional): Divergence time for the new reticulation
                                  node. Defaults to None.
    Returns:
        N/A
    """
    _core._gu_add_hybrid(net, source, destination, t_src, t_dest)

def _valid_network_degrees(net : Network) -> bool:
    """
    Check the degree invariants a rooted phylogenetic network must satisfy.

    Unlike :func:`validate_binary`, this tolerates polytomies (internal
    out-degree >= 2 rather than exactly 2), insists on a *single* root, and
    reports one boolean instead of a violation map.  Acyclicity is not
    checked here -- callers pair this with :meth:`Network.is_acyclic`.

    Args:
        net (Network): A network object.

    Returns:
        bool: True when there is exactly one root (in-degree 0, out-degree
              >= 2), every leaf has in-degree 1, every reticulation has
              in-degree 2 and out-degree 1, and every other internal node
              has in-degree 1 and out-degree >= 2.
    """
    return _core._gu_valid_network_degrees(net)

def pairwise_leaf_distance(net: Network,
                           use_branch_lengths: bool = True
                           ) -> dict[tuple[str, str], float]:
    """
    Compute pairwise distances between leaves on the underlying undirected graph.

    If use_branch_lengths is True, sum edge lengths; otherwise, use unit weights.

    Args:
        net (Network): A network object.
        use_branch_lengths (bool): Whether to sum edge lengths.

    Returns:
        dict[tuple[str, str], float]: Map from sorted (leaf_i, leaf_j) to distance.
    """
    return _core._gu_pairwise_leaf_distance(net, use_branch_lengths)

def _undirected_adjacency(net : Network) -> Dict[Node, list[Node]]:
    """
    Build neighbour lists for the undirected view of a directed network.

    Parallel edges collapse to a single neighbour entry, so that a bubble does
    not read as a cycle to the Tarjan traversals that consume this.

    Args:
        net (Network): A network object.
    Returns:
        Dict[Node, list[Node]]: Each node mapped to its distinct neighbours,
                                in first-encountered order.
    """
    return _core._gu_undirected_adjacency(net)

def bridges_and_articulations(net: Network) -> tuple[list[tuple[str, str]], list[str]]:
    """
    Compute bridges and articulation points on the underlying undirected graph
    using Tarjan's algorithm.

    Args:
        net (Network): A network object.

    Returns:
        tuple[list[tuple[str, str]], list[str]]:
            A pair (bridges, articulations) where bridges are (u, v) tuples
            with u < v by node label, and articulations are node labels.
    """
    return _core._gu_bridges_and_articulations(net)

def blobs(net: Network) -> list[set[Node]]:
    """
    Return the biconnected components ("blobs") of the underlying undirected graph.

    Args:
        net (Network): A network object.

    Returns:
        list[set[Node]]: A list where each element is the node set of one
                         biconnected component in the undirected view of the
                         network.
    """
    return _core._gu_blobs(net)

def tree_of_blobs(net: Network) -> list[Network]:
    """
    Decompose a phylogenetic network into its biconnected components (blobs).

    Each blob is a maximal biconnected subgraph of the underlying undirected
    graph. Returns a list of Network objects, one per biconnected component,
    containing copies of the original nodes and directed edges. Node names
    are preserved but all objects are distinct from the original network.

    Args:
        net (Network): A phylogenetic network.

    Returns:
        list[Network]: One Network per biconnected component (blob).
    """
    return _core._gu_tree_of_blobs(net)

def level(net: Network) -> int:
    """
    Compute the level of a phylogenetic network.

    The level is the maximum number of reticulation nodes contained in any
    biconnected component ("blob") of the underlying undirected graph.

    Returns 0 for trees.

    Args:
        net (Network): Input network.

    Returns:
        int: The network level.
    """
    return _core._gu_level(net)

def count_displayed_trees(net: Network, exact: bool = False) -> int:
    """
    Count the trees displayed by a network.

    By default this is the product, over reticulation nodes, of their inbound
    edge counts (typically 2), which counts hybrid-edge *choices* rather than
    trees. It is an upper bound: two choices display the same tree whenever
    each discards the other's subtree, so the number of distinct displayed
    trees can be strictly smaller.

    Pass ``exact=True`` to enumerate the displayed trees with
    :func:`get_displayed_trees` and count the distinct ones instead. That is
    exponentially more expensive, since it builds every tree.

    Args:
        net (Network): A network object.
        exact (bool): Count distinct displayed trees rather than hybrid-edge
                      choices. Defaults to False.

    Returns:
        int: The number of displayed trees, or of hybrid-edge choices.
    """
    return _core._gu_count_displayed_trees(net, exact)

def is_tree(net: Network) -> bool:
    """
    Check whether a network is a tree (i.e., has no reticulation nodes).

    Args:
        net (Network): A network object.

    Returns:
        bool: True if there are no reticulation nodes, False otherwise.
    """
    return _core._gu_is_tree(net)

def _edge_between(net : Network, parent : Node, child : Node) -> Edge:
    """
    The single directed edge ``parent -> child``.

    ``Network.get_edge`` returns either an ``Edge`` or a list of them
    (bubbles put two edges on the same node pair); this normalises both
    shapes to one edge.

    Args:
        net (Network): A network object.
        parent (Node): Source of the edge.
        child (Node): Destination of the edge.

    Returns:
        Edge: The edge running from parent to child.
    """
    return _core._gu_edge_between(net, parent, child)

def _node_height(net : Network,
                 node : Node,
                 cache : dict[Node, float]) -> float:
    """
    Ultrametric height above the present, from child branch lengths.

    Derived from child branch lengths by a memoised post-order recursion
    rather than read from ``Node.get_time()``, so it is correct even when
    explicit node times were never populated. Leaves are height 0, and a
    missing branch length counts as 0.

    Args:
        net (Network): The tree or network 'node' belongs to.
        node (Node): Node whose height is requested.
        cache (dict[Node, float]): Memo of already-computed heights, mutated
                                   in place.

    Returns:
        float: The node's height, in the branch lengths' own units.
    """
    return _core._gu_node_height(net, node, cache)

def _node_heights(net : Network) -> dict[Node, float]:
    """
    Map every node of a network to its height above the present.

    Args:
        net (Network): A network object.

    Returns:
        dict[Node, float]: Height of every node, in branch-length units.
    """
    return _core._gu_node_heights(net)

def _clone_net(net : Network) -> Network:
    """
    Fast structural clone of a network, for cheap whole-network rollback.

    Topology-changing moves and samplers snapshot a network before mutating
    it in place so the mutation can be undone wholesale.
    :func:`copy.deepcopy` walks the entire object graph reflectively --
    nested attribute values, sequence data, and back-references -- which
    profiling showed to dominate per-iteration cost.
    :meth:`Network.copy` instead builds fresh ``Node`` / ``Edge`` objects
    directly in O(V+E), preserving topology, branch lengths, gammas, node
    times, and reticulation flags: everything the coalescent / Felsenstein
    scorers read.

    Node attribute dictionaries are shared with the source by reference,
    which is safe because no move or scorer mutates them in place -- the
    per-node state moves change (time, reticulation flag) and per-edge state
    (length, gamma) are copied by value, and ultrametric heights live in
    external ``dict[Node, float]`` maps.

    Args:
        net (Network): A network object.

    Returns:
        Network: A structural copy of 'net'.
    """
    return _core._gu_clone_net(net)

def induced_subnetwork_by_taxa(net: Network, taxa: list[str]) -> Network:
    """
    Construct the subnetwork induced by a set of leaf labels.

    The induced subnetwork is formed by taking the MRCA of the target leaves
    and retaining only those descendant branches that lead to at least one
    target leaf. Node/edge attributes (gamma, length) are copied where
    applicable.

    Args:
        net (Network): A network object.
        taxa (list[str]): List of leaf labels to induce on.

    Raises:
        NetworkError: If any leaf name is not found in the network.

    Returns:
        Network: The induced subnetwork as a new Network object.
    """
    return _core._gu_induced_subnetwork_by_taxa(net, taxa)

def is_isomorphic(net1 : Network, net2 : Network) -> bool:
    """
    Returns True, if net1 and net2 are topologically identical networks. Even if 
    branch lengths are different, or node labels are different, networks can
    be isomorphic.
    
    ie-- all trees with 3 taxa are isomorphic. quartet trees have only a few 
    different variations. Networks of any size, have infinitely many possible 
    topologies.
    
    (A,B)C; is isomorphic with (D,E)F; 
    (A,B),(C,D); is not isomorphic with ((A,B)C, D);

    Args:
        net1 (Network): A Network object
        net2 (Network): Another Network object
    
    Returns:
        bool: True if the networks are topologically isomorphic, False otherwise
    """
    return _core._gu_is_isomorphic(net1, net2)


# ── Network Distance / Comparison Helpers ──────────────────────────────

def _leaf_label_set(net: Network) -> frozenset[str]:
    """Return the frozenset of leaf labels in the network."""
    return _core._gu_leaf_label_set(net)


def _require_same_leaves(net1: Network, net2: Network, caller: str) -> frozenset[str]:
    """
    Check that two networks have identical leaf label sets.

    Args:
        net1 (Network): First network.
        net2 (Network): Second network.
        caller (str): Name of the metric, for the error message.

    Returns:
        frozenset[str]: The shared leaf label set.

    Raises:
        NetworkError: If the leaf label sets differ.
    """
    return _core._gu_require_same_leaves(net1, net2, caller)


def _hardwired_clusters_by_label(
    net: Network,
    include_trivial: bool = False
) -> set[frozenset[str]]:
    """
    Compute hardwired clusters expressed as frozensets of leaf labels.

    A hardwired cluster of node v is the full set of leaf descendants
    reachable from v by following *all* directed edges (ignoring the
    reticulation / switching semantics).

    Args:
        net: A phylogenetic network.
        include_trivial: If True, include singleton leaf clusters.

    Returns:
        The set of hardwired clusters (each a frozenset of leaf labels).
    """
    return _core._gu_hardwired_clusters_by_label(net, include_trivial)

def _leaf_labels_below(net : Network) -> dict[Node, frozenset[str]]:
    """
    Map every node of a network to the leaf *labels* below it.

    The label-valued form of :meth:`Network.leaf_descendants_all`, which
    returns :class:`~phynetpy.Network.Node` objects belonging to one
    particular graph and so cannot be compared across networks.

    Args:
        net (Network): A network object.

    Returns:
        dict[Node, frozenset[str]]: Hardwired cluster of every node
                                    reachable from the root.
    """
    return _core._gu_leaf_labels_below(net)

def network_clusters(net: Network) -> set[frozenset[str]]:
    """
    The set of non-trivial leaf-label clusters induced by a network's nodes.

    Label-based rather than node-based, so clusters are comparable *across*
    networks -- which :func:`get_all_clusters` is not, since it returns sets
    of :class:`~phynetpy.Network.Node` objects belonging to one graph.

    Args:
        net (Network): The network to summarise.

    Returns:
        set[frozenset[str]]: One frozenset of leaf labels per non-trivial
                             cluster: size 2 or more, excluding the full
                             leaf set.
    """
    return _core._gu_network_clusters(net)


def _softwired_clusters_by_label(
    net: Network,
    include_trivial: bool = False
) -> set[frozenset[str]]:
    """
    Compute softwired clusters expressed as frozensets of leaf labels.

    The softwired cluster set is the union of cluster sets taken over
    every displayed tree of the network.

    Args:
        net: A phylogenetic network.
        include_trivial: If True, include singleton leaf clusters.

    Returns:
        The set of softwired clusters (each a frozenset of leaf labels).
    """
    return _core._gu_softwired_clusters_by_label(net, include_trivial)


def _mu_vectors(
    net: Network,
    leaf_order: list[str] | None = None
) -> list[tuple[int, ...]]:
    """
    Compute the mu-representation of a network.

    For every node v in the network whose leaf set is
    X = {l_1, ..., l_n}, the mu-vector is

        mu(v) = (m(v, l_1), ..., m(v, l_n))

    where m(v, l_i) counts the number of distinct directed paths from v
    to leaf l_i.

    Args:
        net: A phylogenetic network (must be acyclic).
        leaf_order: Fixed ordering of leaf labels.  When comparing two
            networks the same ordering must be used.  If *None*, leaves
            are sorted alphabetically.

    Returns:
        A list of mu-vectors (one per node in V).
    """
    return _core._gu_mu_vectors(net, leaf_order)


def _tripartitions(
    net: Network
) -> set[tuple[frozenset[str], frozenset[str], frozenset[str]]]:
    """
    Compute tripartitions induced by tree nodes with exactly two children.

    For each such node u with children c_1, c_2 the tripartition is

        (desc(c_1),  desc(c_2),  X \\ (desc(c_1) | desc(c_2)))

    where X is the full leaf label set and desc(c_i) are the hardwired
    leaf descendants of child c_i.

    Args:
        net: A phylogenetic network.

    Returns:
        A set of tripartitions, each in canonical (sorted) form.
    """
    return _core._gu_tripartitions(net)


def _displayed_tree_topology_set(
    net: Network
) -> set[frozenset[frozenset[str]]]:
    """
    Enumerate the set of unique displayed-tree topologies.

    Each topology is identified by its cluster set (frozenset of
    frozensets of leaf labels), so two trees that share the same
    cluster set are considered identical.

    Args:
        net: A phylogenetic network.

    Returns:
        A set of tree topologies, each encoded as a frozenset of clusters.
    """
    return _core._gu_displayed_tree_topology_set(net)


# ── Network Distance Metrics ──────────────────────────────────────────

def _symmetric_difference_distance(
    s1: set,
    s2: set,
    normalize: bool
) -> Union[int, float]:
    """
    Size of the symmetric difference of two feature sets.

    Shared tail of the cluster, tripartition, and displayed-tree metrics:
    each builds a set of structural features per network and then differs
    only in how that set is built.

    Args:
        s1: Feature set of the first network.
        s2: Feature set of the second network.
        normalize: Divide by |s1 | s2| to get a value in [0, 1].

    Returns:
        The distance (int, or float when normalized).
    """
    return _core._gu_symmetric_difference_distance(s1, s2, normalize)


def hardwired_cluster_distance(
    net1: Network,
    net2: Network,
    normalize: bool = False
) -> Union[int, float]:
    """
    Hardwired cluster distance between two phylogenetic networks.

    The hardwired cluster of a node v is the set of all leaf labels
    reachable from v by following every directed edge.  The distance is
    the cardinality of the symmetric difference of the two hardwired
    cluster sets:

        d_HC(N1, N2) = |HC(N1)  triangle  HC(N2)|

    When *normalize* is True the result is divided by |HC(N1) | HC(N2)|,
    yielding a value in [0, 1].

    Reference
    ---------
    Huson, Rupp & Scornavacca.  *Phylogenetic Networks: Concepts,
    Algorithms and Applications* (2010), Chapter 7.

    Args:
        net1: First network.
        net2: Second network.
        normalize: Return a value in [0, 1] instead of a raw count.

    Returns:
        The hardwired cluster distance (int, or float when normalized).
    """
    return _core._gu_hardwired_cluster_distance(net1, net2, normalize)


def softwired_cluster_distance(
    net1: Network,
    net2: Network,
    normalize: bool = False
) -> Union[int, float]:
    """
    Softwired cluster distance between two phylogenetic networks.

    The softwired cluster set of a network is the union of cluster sets
    across all of its displayed trees.  The distance is the cardinality
    of the symmetric difference:

        d_SC(N1, N2) = |SC(N1)  triangle  SC(N2)|

    .. note::
       This enumerates all 2^k displayed trees where k is the number of
       reticulations.  For networks with many reticulations the cost is
       exponential.

    When *normalize* is True the result is divided by |SC(N1) | SC(N2)|,
    yielding a value in [0, 1].

    Reference
    ---------
    Huson, Rupp & Scornavacca.  *Phylogenetic Networks: Concepts,
    Algorithms and Applications* (2010), Chapter 7.

    Args:
        net1: First network.
        net2: Second network.
        normalize: Return a value in [0, 1] instead of a raw count.

    Returns:
        The softwired cluster distance (int, or float when normalized).
    """
    return _core._gu_softwired_cluster_distance(net1, net2, normalize)


def mu_distance(net1: Network, net2: Network) -> int:
    """
    Mu-representation distance between two phylogenetic networks.

    For every node v in a network with leaf set X = {l_1, ..., l_n} the
    *mu-vector* is

        mu(v) = (m(v, l_1), ..., m(v, l_n))

    where m(v, l_i) is the number of directed paths from v to leaf l_i.
    The *mu-representation* of a network is the multiset of mu-vectors
    over all nodes.  The distance is the size of the multiset symmetric
    difference of the two mu-representations.

    This is a metric on the space of *reduced* phylogenetic networks
    (networks with no degree-2 tree nodes and no parallel edges).  It
    is also referred to as the *topological distance for reduced
    networks*.  For trees it reduces to the Robinson--Foulds distance.

    Both networks must share the same leaf label set.

    Reference
    ---------
    Cardona, Llabres, Rossello & Valiente.  *Metrics for Phylogenetic
    Networks I: Generalizations of the Robinson-Foulds Metric*.
    IEEE/ACM Trans. Comput. Biol. Bioinformatics **6** (2009), 46--61.

    Args:
        net1: First network.
        net2: Second network.

    Raises:
        NetworkError: If the two networks have different leaf label sets.

    Returns:
        The mu-distance (non-negative integer).
    """
    return _core._gu_mu_distance(net1, net2)


def hybrid_ancestry(net: Network) -> frozenset[str]:
    """
    The leaf labels that descend from at least one reticulation node.

    A taxon is of *hybrid ancestry* if some reticulation lies on a directed
    path from the root to it, i.e. it belongs to the hardwired cluster of
    some reticulation node. Since every leaf below a reticulation is reached
    along every edge, the set is independent of inheritance probabilities and
    of the root placement.

    Args:
        net (Network): A phylogenetic network.

    Returns:
        frozenset[str]: Labels of the leaves below any reticulation; empty
                        for a tree.
    """
    return _core._gu_hybrid_ancestry(net)


def hybrid_ancestry_error(estimated: Network, true_net: Network) -> float:
    """
    Hybrid-ancestry balanced error (HAE) of an estimated network against the
    true network.

    Let B be the hybrid-ancestry set (see :func:`hybrid_ancestry`) of the
    estimate and A that of the truth. The false-positive rate is
    |B - A| / |B| and the false-negative rate |A - B| / |A|; HAE is their
    mean, in [0, 1]. Each rate is defined as 0 when its denominator is empty,
    so a tree estimated for a reticulate truth scores 0.5 and a tree estimated
    for a tree scores 0. Unlike :func:`mu_distance`, this measure is
    *asymmetric*: the argument order matters.

    The convention follows the HAE column of the NetCSEval evaluation
    scripts (``hae_datafinal.jl``), so values are comparable to that work.

    Args:
        estimated (Network): The inferred network.
        true_net (Network): The reference (true) network.

    Returns:
        float: Balanced hybrid-ancestry error in [0, 1].

    Raises:
        NetworkError: If the two networks have different leaf label sets.
    """
    return _core._gu_hybrid_ancestry_error(estimated, true_net)


def tripartition_distance(
    net1: Network,
    net2: Network,
    normalize: bool = False
) -> Union[int, float]:
    """
    Tripartition-based distance between two phylogenetic networks.

    For each tree node u with exactly two children c_1, c_2 the
    tripartition is

        (desc(c_1),  desc(c_2),  X \\ (desc(c_1) | desc(c_2)))

    where X is the full leaf set.  The distance is the cardinality of
    the symmetric difference of the tripartition sets:

        d_T(N1, N2) = |T(N1)  triangle  T(N2)|

    Reticulation nodes (in-degree >= 2) are excluded from tripartition
    generation.

    When *normalize* is True the result is divided by |T(N1) | T(N2)|,
    yielding a value in [0, 1].

    Reference
    ---------
    Moret *et al.*; Huson, Rupp & Scornavacca.  *Phylogenetic Networks:
    Concepts, Algorithms and Applications* (2010), Chapter 7.

    Args:
        net1: First network.
        net2: Second network.
        normalize: Return a value in [0, 1] instead of a raw count.

    Returns:
        The tripartition distance (int, or float when normalized).
    """
    return _core._gu_tripartition_distance(net1, net2, normalize)


def displayed_tree_distance(
    net1: Network,
    net2: Network,
    normalize: bool = False
) -> Union[int, float]:
    """
    Tree-based distance between two phylogenetic networks.

    Each network's set of unique displayed-tree topologies (identified
    by cluster representation) is computed.  The distance is the
    cardinality of the symmetric difference:

        d_DT(N1, N2) = |DT(N1)  triangle  DT(N2)|

    .. note::
       This enumerates all 2^k displayed trees per network.  For
       networks with many reticulations the cost is exponential.

    When *normalize* is True the result is divided by |DT(N1) | DT(N2)|,
    yielding a value in [0, 1].

    Args:
        net1: First network.
        net2: Second network.
        normalize: Return a value in [0, 1] instead of a raw count.

    Returns:
        The displayed-tree distance (int, or float when normalized).
    """
    return _core._gu_displayed_tree_distance(net1, net2, normalize)


def robinson_foulds_distance(
    net1: Network,
    net2: Network,
    normalize: bool = False
) -> Union[int, float]:
    """
    Robinson--Foulds distance between two phylogenetic trees or networks.

    For rooted trees the RF distance equals the cardinality of the
    symmetric difference of the non-trivial cluster sets (clusters of
    size >= 2 and < n, where n is the number of leaves).

    For networks the function uses hardwired clusters, which is the
    natural generalization of RF to DAGs.

    Both inputs must share the same leaf label set.

    When *normalize* is True the result is divided by
    |C(N1)| + |C(N2)| (the maximum possible RF for the given cluster
    counts), yielding a value in [0, 1].

    Reference
    ---------
    Robinson & Foulds.  *Comparison of Phylogenetic Trees*.
    Mathematical Biosciences **53** (1981), 131--147.

    Cardona, Llabres, Rossello & Valiente.  *Metrics for Phylogenetic
    Networks I: Generalizations of the Robinson-Foulds Metric*.
    IEEE/ACM Trans. Comput. Biol. Bioinformatics **6** (2009), 46--61.

    Args:
        net1: First tree / network.
        net2: Second tree / network.
        normalize: Return a value in [0, 1] instead of a raw count.

    Raises:
        NetworkError: If the two inputs have different leaf label sets.

    Returns:
        The RF distance (int, or float when normalized).
    """
    return _core._gu_robinson_foulds_distance(net1, net2, normalize)


# ── Branch-Length-Aware Distance Helpers ───────────────────────────────

def _average_pairwise_matrix(
    net: Network,
    weighted: bool = False
) -> dict[tuple[str, str], float]:
    """
    Compute the (weighted) average pairwise leaf distance matrix.

    For each pair of *original* leaves, the distance is averaged across
    all displayed trees.  If *weighted* is True the average uses each
    displayed tree's probability (product of kept gammas); otherwise a
    uniform average is used.

    Branch lengths on the network edges are required.  Edges whose
    length is *None* are treated as having length 0.

    Args:
        net: A phylogenetic network.
        weighted: Use probability-weighted averaging.

    Returns:
        A dictionary mapping sorted (label_i, label_j) pairs to their
        averaged distance.
    """
    return _core._gu_average_pairwise_matrix(net, weighted)


def average_path_distance(
    net1: Network,
    net2: Network,
    normalize: bool = False
) -> float:
    """
    Average Path Distance (APD) between two phylogenetic networks.

    For each network a pairwise leaf distance matrix is computed by
    averaging the path lengths across all displayed trees (uniform
    weighting).  The APD is the L1 norm of the difference between the
    two matrices:

        APD(N1, N2) = sum_{i<j} |D_N1(l_i,l_j) - D_N2(l_i,l_j)|

    When *normalize* is True the sum is divided by the number of leaf
    pairs C(n, 2).

    Branch lengths on every edge are required.

    .. note::
       This enumerates all 2^k displayed trees per network.  For
       networks with many reticulations the cost is exponential.

    Reference
    ---------
    Yakici, Ogilvie & Nakhleh.  *Phylogenetic Network Dissimilarity
    Measures that Take Branch Lengths into Account*.  RECOMB-CG 2022,
    LNCS 13234, pp. 86--102.

    Args:
        net1: First network.
        net2: Second network.
        normalize: Divide by C(n, 2) to get a per-pair average.

    Raises:
        NetworkError: If the two networks have different leaf label sets.

    Returns:
        The APD (float).
    """
    return _core._gu_average_path_distance(net1, net2, normalize)


def weighted_average_path_distance(
    net1: Network,
    net2: Network,
    normalize: bool = False
) -> float:
    """
    Weighted Average Path Distance (WAPD) between two networks.

    Identical to :func:`average_path_distance` except that the average
    over displayed trees is weighted by each tree's probability (the
    product of the gamma values of the kept reticulation edges).

        WAPD(N1, N2) = sum_{i<j} |WD_N1(l_i,l_j) - WD_N2(l_i,l_j)|

    When *normalize* is True the sum is divided by the number of leaf
    pairs C(n, 2).

    Branch lengths and gamma values on every reticulation edge are
    required.

    .. note::
       This enumerates all 2^k displayed trees per network.  For
       networks with many reticulations the cost is exponential.

    Reference
    ---------
    Yakici, Ogilvie & Nakhleh.  *Phylogenetic Network Dissimilarity
    Measures that Take Branch Lengths into Account*.  RECOMB-CG 2022,
    LNCS 13234, pp. 86--102.

    Args:
        net1: First network.
        net2: Second network.
        normalize: Divide by C(n, 2) to get a per-pair average.

    Raises:
        NetworkError: If the two networks have different leaf label sets.

    Returns:
        The WAPD (float).
    """
    return _core._gu_weighted_average_path_distance(net1, net2, normalize)


# ── Shared helpers for ascii() / ascii_extended() ──────────────────────

def _ascii_compute_depths(
    net: Network, root: Node
) -> tuple[dict[Node, int], int]:
    """BFS depth assignment. Reticulation nodes keep the maximum depth."""
    node_depth: dict[Node, int] = {}
    queue: deque[tuple[Node, int]] = deque([(root, 0)])
    max_depth = 0
    while queue:
        node, depth = queue.popleft()
        if node in node_depth:
            node_depth[node] = max(node_depth[node], depth)
        else:
            node_depth[node] = depth
        max_depth = max(max_depth, depth)
        for child in net.get_children(node):
            queue.append((child, depth + 1))
    return node_depth, max_depth


def _ascii_ordered_leaves(net: Network, node: Node) -> list[Node]:
    """Left-to-right DFS collection of leaf nodes."""
    children = net.get_children(node)
    if len(children) == 0:
        return [node]
    result: list[Node] = []
    for child in children:
        result.extend(_ascii_ordered_leaves(net, child))
    return result


def _ascii_compute_x(
    net: Network, node: Node, node_x: dict[Node, float]
) -> float:
    """Recursive x-position: leaves keep their preset value; internal
    nodes are centred on the mean of their children."""
    if node in node_x:
        return node_x[node]
    children = net.get_children(node)
    if len(children) == 0:
        return node_x.get(node, 0.0)
    child_xs = [_ascii_compute_x(net, child, node_x) for child in children]
    node_x[node] = sum(child_xs) / len(child_xs)
    return node_x[node]


def _ascii_group_by_depth(
    node_depth: dict[Node, int], node_x: dict[Node, float]
) -> dict[int, list[Node]]:
    """Group nodes by depth and sort each group by x-position."""
    nodes_by_depth: dict[int, list[Node]] = {}
    for node, depth in node_depth.items():
        nodes_by_depth.setdefault(depth, []).append(node)
    for depth in nodes_by_depth:
        nodes_by_depth[depth].sort(key=lambda n: node_x[n])
    return nodes_by_depth


def _ascii_prepare_layout(
    net: Network,
) -> tuple[Node, dict[Node, int], int, list[Node], dict[Node, float], None] | str:
    """Shared preamble for both ascii renderers.

    Returns either a descriptive error string or a tuple of
    ``(root, node_depth, max_depth, ordered_leaves, node_x, None)``.
    """
    if len(net.V()) == 0:
        return "(empty network)"
    try:
        root = net.root()
    except Exception:
        return "(network has no valid root)"

    node_depth, max_depth = _ascii_compute_depths(net, root)
    ordered_leaves = _ascii_ordered_leaves(net, root)
    if len(ordered_leaves) == 0:
        return f"  {root.label if root.label else '?'}"

    node_x: dict[Node, float] = {leaf: float(i) for i, leaf in enumerate(ordered_leaves)}
    _ascii_compute_x(net, root, node_x)
    return root, node_depth, max_depth, ordered_leaves, node_x, None


# ── Public ASCII renderers ─────────────────────────────────────────────

def ascii(net: Network, show_edge_lengths: bool = False) -> str:
    """
    Prints out an ascii art depiction of this Network object as a vertical
    tree/network with the root at the top and leaves at the bottom.

    Args:
        net (Network): A Network
        show_edge_lengths (bool): If True, display edge lengths. Defaults to False.

    Returns:
        str: The ASCII art representation of the network

    Example:
        For newick string "((C,D)A, E)Root;", outputs::

                  Root
                 /    \\
                A      E
               / \\
              C   D
    """
    layout = _ascii_prepare_layout(net)
    if isinstance(layout, str):
        return layout
    root, node_depth, max_depth, ordered_leaves, node_x, _ = layout

    char_width = 6
    max_label_len = max(len(n.label) if n.label else 1 for n in net.V())
    char_width = max(char_width, max_label_len + 2)
    total_width = int(len(ordered_leaves) * char_width + max_label_len)
    nodes_by_depth = _ascii_group_by_depth(node_depth, node_x)

    def _center(node: Node) -> int:
        """Column of the node's label centre in the rendered rows."""
        return int(node_x[node] * char_width) + char_width // 2

    lines: list[str] = []
    for depth in range(max_depth + 1):
        if depth not in nodes_by_depth:
            continue
        nodes_at_depth = nodes_by_depth[depth]

        label_line = [' '] * total_width
        for node in nodes_at_depth:
            x_pos = int(node_x[node] * char_width)
            label = node.label if node.label else "?"
            start = max(0, x_pos - len(label) // 2 + char_width // 2)
            for i, ch in enumerate(label):
                if start + i < total_width:
                    label_line[start + i] = ch
        lines.append(''.join(label_line).rstrip())

        if depth < max_depth:
            branch_line = [' '] * total_width
            for node in nodes_at_depth:
                children = net.get_children(node)
                if len(children) == 0:
                    continue
                parent_x = _center(node)
                child_positions = sorted(
                    [(_center(c), c) for c in children], key=lambda t: t[0]
                )
                if len(child_positions) == 1:
                    cx = child_positions[0][0]
                    if cx < parent_x:
                        if 0 <= parent_x - 1 < total_width:
                            branch_line[parent_x - 1] = '/'
                    elif cx > parent_x:
                        if 0 <= parent_x + 1 < total_width:
                            branch_line[parent_x + 1] = '\\'
                    else:
                        if 0 <= parent_x < total_width:
                            branch_line[parent_x] = '|'
                elif len(child_positions) == 2:
                    if child_positions[0][0] < parent_x:
                        if 0 <= parent_x - 1 < total_width:
                            branch_line[parent_x - 1] = '/'
                    if child_positions[1][0] > parent_x:
                        if 0 <= parent_x + 1 < total_width:
                            branch_line[parent_x + 1] = '\\'
                else:
                    leftmost = child_positions[0][0]
                    rightmost = child_positions[-1][0]
                    if leftmost < parent_x and 0 <= parent_x - 1 < total_width:
                        branch_line[parent_x - 1] = '/'
                    if rightmost > parent_x and 0 <= parent_x + 1 < total_width:
                        branch_line[parent_x + 1] = '\\'
                    for cx, _ in child_positions:
                        if cx == parent_x and 0 <= parent_x < total_width:
                            branch_line[parent_x] = '|'
            lines.append(''.join(branch_line).rstrip())

    return '\n'.join(lines)


# ── Reticulation-Aware Comparison (re-export) ──────────────────────────
#
def _direct_branch_leaf_sets(net: Network, node: Node) -> list[frozenset[str]]:
    """Leaf sets of the clades directly below ``node``.

    A leaf is its own one-element branch. Used to describe a reticulation
    by the clades sitting beside it, rather than by node labels, which are
    not stable across inferred subnetworks.

    Args:
        net (Network): Network containing ``node``.
        node (Node): The node whose child clades are read.

    Returns:
        list[frozenset[str]]: One frozenset of leaf labels per child.
    """
    return _core._gu_direct_branch_leaf_sets(net, node)


def reticulation_signature(net: Network, node: Node) -> tuple:
    """Leaf-set signature of one reticulation.

    The signature is ``(child_branches, parent_branches)``. ``child_branches``
    is the list of leaf sets directly below the reticulation's child.
    ``parent_branches`` has one entry per parent: the leaf sets of that
    parent's other children (the sibling clades on that side). Parent order
    is the network's parent order and is not significant for equality; see
    :func:`same_reticulation`.

    This is the identity used to decide that two inferred subnetworks found
    the same hybridization. It is not a phylogenetic distance.

    Args:
        net (Network): Network containing ``node``.
        node (Node): A reticulation node (in-degree 2).

    Returns:
        tuple: ``(child_branches, parent_branches)``, each a tuple of
        frozensets (parent side: a tuple of tuples of frozensets).

    Raises:
        NetworkError: If ``node`` is not in ``net``.
    """
    return _core._gu_reticulation_signature(net, node)


def _signature_parents(sig) -> tuple:
    """Parent-side branch lists of a signature or of a record carrying one."""
    return _core._gu_signature_parents(sig)


def _branches_overlap(side_a, side_b) -> bool:
    """True when some branch on ``side_a`` shares a leaf with some branch on ``side_b``."""
    return _core._gu_branches_overlap(side_a, side_b)


def same_reticulation(sig_a, sig_b) -> bool:
    """Whether two reticulation signatures are the same event.

    Parents match up to order. A side matches when one of its sibling
    branches shares at least one leaf with a sibling branch on the other
    signature's corresponding side (overlap, not set equality). The child
    clade is ignored here; callers that need it read it off the signature.

    Args:
        sig_a: A signature from :func:`reticulation_signature`, or a record
            with a ``signature`` attribute or key.
        sig_b: The signature to compare against.

    Returns:
        bool: True when the two parent pairs overlap in either order.
    """
    return _core._gu_same_reticulation(sig_a, sig_b)


def group_reticulations(records: list) -> list[list]:
    """Group reticulation records that are the same event.

    The first record of a group is its representative. Later records join
    the group when :func:`same_reticulation` says they match that
    representative. Records are not reordered inside a group.

    Args:
        records (list): Records that carry a ``signature`` (dict key or
            attribute), as returned alongside :func:`reticulation_signature`.

    Returns:
        list[list]: One list per distinct event.
    """
    return _core._gu_group_reticulations(records)


def rooted_triples(net: Network) -> set[tuple[str, str, str]]:
    """Rooted triples displayed by a tree or network.

    Each triple is ``(a, b, c)`` with ``a < b``, meaning the rooted triple
    ``(a, b | c)``: ``a`` and ``b`` are together in a clade that excludes
    ``c``. Every such triple is emitted once, at the node that is the MRCA
    of ``a`` and ``b``. A root polytomy displays no triple, because no pair
    is resolved.

    This is the set the displayed-tree picker counts. It is not
    :meth:`Network.rooted_triplet_distance`, which is a distance between
    two networks, and it is not an MSC pseudo-likelihood.

    Args:
        net (Network): A rooted tree or network.

    Returns:
        set[tuple[str, str, str]]: The displayed triples.
    """
    return _core._gu_rooted_triples(net)


# The tripartition-matching reticulation dissimilarity lives in its own
# module (:mod:`~.ReticulationComparison`) but is surfaced here alongside the
# other network distance metrics for discoverability.  The import is placed at
# the bottom of this file, and ``ReticulationComparison`` imports the induced
# sub-network helper lazily, so there is no circular import at load time.
from .ReticulationComparison import (  # noqa: E402
    JACCARD,
    SYMMETRIC,
    ReticulationTripartition,
    NetworkComparison,
    reticulation_tripartitions,
    set_distance,
    tripartition_dissimilarity,
    nakhleh_metric,
    reticulation_dissimilarity,
    reticulation_precision_recall,
    combined_dissimilarity,
    compare_networks,
)
