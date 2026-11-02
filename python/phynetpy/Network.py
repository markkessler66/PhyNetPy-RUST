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
Last Edit : 3/11/25
First Included in Version : 1.0.0
Docs   - [ ]
Tests  - [ ] 
Design - [ ]
"""

from __future__ import annotations

import copy
import re
from typing import Any, Union

import numpy as np

from . import _core
from ._core import Node, Edge, Branch, NodeSet, EdgeSet, stable_hash
from ._core import _complete_hybrid_gammas

#############################
####  EXCEPTION CLASSES  ####
#############################

class NetworkError(Exception):
    """
    This exception is raised when a network is malformed, 
    or if a network operation fails.
    """
    def __init__(self, message : str = "Error operating on a Network") -> None:
        """
        Initialize with an error message that will print upon the exception 
        being raised.

        Args:
            message (str, optional): Error message. Defaults to 
                                    "Error operating on a Network".
        Returns:
            N/A
        """
        self.message = message
        super().__init__(self.message)

class NodeError(Exception):
    """
    This exception is raised when a Node operation fails.
    """
    def __init__(self, message : str = "Error in Node Class") -> None:
        """
        Initialize with an error message that will print upon the exception 
        being raised.

        Args:
            message (str, optional): Error message. Defaults to 
                                     "Error in Node Class".
        Returns:
            N/A
        """
        super().__init__(message)
        
class EdgeError(Exception):
    """
    This exception is raised when an Edge operation fails.
    """
    def __init__(self, message : str = "Error in Edge Class") -> None:
        """
        Initialize with an error message that will print upon the exception 
        being raised.

        Args:
            message (str, optional): Error message. Defaults to 
                                     "Error in Edge Class".
        Returns:
            N/A
        """
        super().__init__(message)


# The Rust core raises these classes, so ``except NetworkError`` catches
# errors from either side.
_core._register_exceptions(NetworkError, NodeError, EdgeError)


#########################
#### NETWORK CLASSES ####
#########################

class Network(_core.Network):
    """
    A directed (and potentially acyclic) graph of Nodes joined by Edges.
    
    An 'Edge' object is a wrapper class for a tuple of two nodes, (a, b),
    where a and b are Node objects, and the direction of the edge is from 
    a to b (a is b's parent) -- thus (a, b) is NOT the same as (b, a).

    The graph structure, its bookkeeping and every algorithm live in the
    compiled core (``phynetpy._core.Network``); this class is the public
    face and adds the few conveniences that call back into Python modules.

    Notes and Allowances:
    
    1) You may create cycles -- however we have provided a method to check if 
       this graph object is acyclic. This method is internally called on 
       methods that assume that a network has no cycles, so be mindful of the 
       state of networks that are passed as arguments.
    
    2) You may have multiple roots. Be mindful of whether this graph is 
       connected and what root you wish to operate on.
    
    3) You may end up with floater nodes/edges, ie this may be an unconnected 
       network with multiple connected components. We will provide a method to 
       check for whether your object is one single connected component. 
       We have also provided methods to remove such artifacts.      
    """

    def __init__(self, 
                 edges : Union[EdgeSet, set[Edge], None] = None, 
                 nodes : Union[NodeSet, set[Node], None] = None,
                 branch_length_unit : Any = None) -> None:
        """
        Initialize a Network object.
        You may initialize with any combination of edges/nodes,
        or provide none at all.
        
        If you provide edges and no nodes, each node present in those edges
        *WILL* be added to the network.

        Raises:
            TypeError: If any provided edge is not an Edge object.
        Args:
            edges (EdgeSet | set[Edge], optional): The edges of the network.
                                                   Defaults to no edges.
            nodes (NodeSet | set[Node], optional): The nodes of the network.
                                                   Defaults to no nodes.
            branch_length_unit: Explicit unit carried by edge lengths. Graph
                operations accept ``UNSPECIFIED``; biological calculations
                require a concrete unit.
        Returns:
            N/A
        """
        self._init(edges, nodes, branch_length_unit)

    @classmethod
    def from_newick(
        cls,
        newick : str,
        branch_length_unit : Any = None,
    ) -> "Network":
        """
        Construct a Network object by passing in a newick string and parsing out
        the topology from there.

        Handles standard Newick, branch lengths, internal labels, reticulation
        nodes marked with ``#``, and ``[&gamma=g]`` inheritance probabilities.

        Args:
            newick (str): An extended newick string.
            branch_length_unit: Unit represented by parsed branch lengths.
        Returns:
            Network: An initialized network object
        Raises:
            NewickParserError: If the Newick string is malformed.
        """
        return _core.parse_newick(cls, newick, branch_length_unit)

    def is_isomorphic(self, other: "Network") -> bool:
        """
        Check if this network is topologically isomorphic to another network.
        
        Two networks are isomorphic if they have the same topology, regardless of
        node labels or branch lengths. This method calls the GraphUtils.is_isomorphic
        function.
        
        Args:
            other (Network): The other network to compare against.
        
        Returns:
            bool: True if the networks are topologically isomorphic, False otherwise.
        """
        from .GraphUtils import is_isomorphic
        return is_isomorphic(self, other)

    def joint_dissimilarity(self,
                            other: "Network",
                            kind: str = "nakhleh") -> "JointDissimilarityResult":
        """
        The joint reticulation-and-tree dissimilarity between this network
        and another on the same taxon set.  See
        :mod:`~phynetpy.JointDissimilarity` for the definition; this method
        calls ``JointDissimilarity.joint_dissimilarity``.

        Args:
            other (Network): The other network to compare against.
            kind (str): Structure dissimilarity used on blocks and
                        remainders: ``"leaf"``, ``"cluster"`` or
                        ``"nakhleh"`` (default).

        Returns:
            JointDissimilarityResult: The dissimilarity record.
        """
        from .JointDissimilarity import joint_dissimilarity
        return joint_dissimilarity(self, other, kind)


# Rust-made networks (copies, subnets, parsed strings) are instances of
# this class, not of the bare core type.
_core._register_network_class(Network)


class MUL(Network):
    """
    A subclass of a Network, that is a binary tree that results from the 
    transformation of a standard network into a Multilabeled Species Tree.
    """
    def __new__(cls, gene_map, rng):
        return super().__new__(cls)

    def __init__(self, 
                 gene_map : dict[str, list[str]],
                 rng : np.random.Generator) -> None:
        """
        Initializes a MUL object.

        Args:
            gene_map (dict[str, list[str]]): A mapping from gene names to 
                                             a set of gene copy names.
            rng (np.random.Generator): A numpy random number generator.
        Returns:    
            N/A
        """
        super().__init__()
        self.net : Network | None = None
        self.mul : Network | None = None
        self.gene_map : dict[str, list[str]] = gene_map
        self.rng : np.random.Generator = rng
   
    def to_mul(self, net : Network) -> Network:
        """
        Creates a (MU)lti-(L)abeled Species Tree from a network

        Raises:
            NetworkError: If the network is malformed with regards to ploidy
        Args:
            net (Network): A Network
        Returns:
            Network: a MUL tree (as a Network obj)
        """
       
        # Number of network leaves must match the number of gene map keys
        if len(net.get_leaves()) != len(self.gene_map.keys()):
            raise NetworkError(f"Input network has incorrect amount of \
                leaves. Given : {len(net.get_leaves())} \
                Expected : { len(self.gene_map.keys())}")
       
        copy_gene_map = copy.deepcopy(self.gene_map)

        # Work on an independent copy of the network so the input is untouched.
        mul_tree, _ = net.copy()

        # Expand reticulations until none remain. The MUL tree of a network is
        # unique, so the construction MUST be deterministic and independent of
        # set-iteration order. The previous bottom-up, leaf-driven traversal
        # chose which parent kept the original subnetwork (vs. a duplicate) by
        # the iteration order of the parent set, and pushed work back onto a
        # queue using the iteration order of child sets. For reticulations that
        # sit directly above a single leaf this is harmless, but for a
        # reticulation above a non-trivial (and possibly nested) clade -- e.g.
        # the (t,u) and (w,x,y,z) hybrid clades in DEFJ scenario J -- different
        # orderings produced structurally different (and incorrect) MUL trees,
        # making the parsimony score random rather than a function of the
        # network. We instead expand reticulations deterministically.
        #
        # Key invariant: always expand a *lowest* reticulation -- one with no
        # reticulation among its descendants -- so the subnetwork being copied
        # is already a pure tree. This is what makes a single ``subnet`` copy a
        # faithful duplicate and keeps the result order-independent.
        def _descendants(node : Node) -> set[Node]:
            """Strict descendants of 'node' in the working MUL tree."""
            return mul_tree.get_subtree_at(node) - {node}

        while True:
            retics = [n for n in mul_tree.V() if mul_tree.in_degree(n) >= 2]
            if not retics:
                break

            # Pick a reticulation with no reticulation descendants (a "lowest"
            # one). Tie-break by label for a fully deterministic order.
            retic_set = set(retics)
            lowest = sorted(
                (r for r in retics if not (_descendants(r) & retic_set)),
                key=lambda r: r.label,
            )
            target = lowest[0]

            # Keep the subnetwork under the first parent (sorted by label) and
            # graft an independent copy of it under every additional parent.
            parents = sorted(mul_tree.get_parents(target), key=lambda p: p.label)
            for parent in parents[1:]:
                subtree = mul_tree.subnet(target)
                mul_tree.remove_edge([parent, target])
                mul_tree.add_nodes(subtree.V())
                for edge in subtree.E():
                    mul_tree.add_edges(edge)
                mul_tree.add_edges(Edge(parent, subtree.root()))

        # Get rid of excess degree-2 connection nodes left by the expansion.
        mul_tree.clean([True, True, True])

        # Rename tips based on the gene mapping. Iterate leaves in sorted label
        # order so the (arbitrary) assignment of homeolog labels to the
        # duplicated copies of a species is reproducible across runs.
        #
        # Reticulation expansion duplicates hybrid subtrees, so a species that
        # sits under (or is) a reticulation needs as many allele/subgenome
        # labels in ``gene_map`` as MUL tips of that species. A 1:1 identity
        # map is not enough when the network has hybrids.
        #
        # Network leaf labels are species names; ``subnet`` appends
        # ``_copyN`` when duplicating. Allele labels in ``gene_map`` values
        # may be any strings and are applied only in this rename step.
        for leaf in sorted(mul_tree.get_leaves(), key=lambda l: l.label):
            species = _mul_tip_species(leaf.label, self.gene_map)
            if species is None:
                raise NetworkError(
                    f"Species tip '{leaf.label}' (after reticulation "
                    f"expansion) does not match any key in the "
                    f"allele/subgenome map. Map keys: "
                    f"{sorted(self.gene_map.keys())}."
                )
            remaining = copy_gene_map[species]
            if not remaining:
                provided = len(self.gene_map.get(species, []))
                raise NetworkError(
                    f"Allele/subgenome map has too few copies for species "
                    f"'{species}' (provided {provided}). Reticulation "
                    f"expansion created more MUL tips for this taxon than "
                    f"the map supplies. Provide one distinct allele label "
                    f"per parental lineage of each hybrid (e.g. "
                    f"'{species}': ['{species}_a', '{species}_b'] for a "
                    f"single reticulation involving '{species}')."
                )
            new_name: str = remaining.pop()
            mul_tree.update_node_name(leaf, new_name)

        leftovers = {
            sp: copies for sp, copies in copy_gene_map.items() if copies
        }
        if leftovers:
            detail = ", ".join(
                f"'{sp}' has {len(copies)} unused "
                f"({'/'.join(copies)})" for sp, copies in sorted(leftovers.items())
            )
            raise NetworkError(
                f"Allele/subgenome map has more copies than MUL tips after "
                f"reticulation expansion: {detail}. Reduce the map to one "
                f"allele label per MUL tip for each species."
            )

        self.mul = mul_tree

        return mul_tree


def _mul_tip_species(leaf_label: str,
                     gene_map: dict[str, list[str]]) -> str | None:
    """
    Recover the species key for a MUL tip label produced by reticulation
    expansion.

    Tips are either the original network leaf name (a ``gene_map`` key) or
    that name plus a ``_copyN`` suffix from :meth:`Network.subnet`. Allele
    labels in ``gene_map`` values are unrelated and are not consulted here.
    """
    if leaf_label in gene_map:
        return leaf_label

    # Strip trailing ``_copy<digits>`` suffixes (may nest for nested retics).
    label = leaf_label
    while True:
        match = re.fullmatch(r"(.*)_copy\d+", label)
        if match is None:
            break
        label = match.group(1)
        if label in gene_map:
            return label

    return None
