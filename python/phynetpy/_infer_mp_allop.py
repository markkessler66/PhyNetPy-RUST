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
Author: Mark Kessler

Description: This file contains the method Infer_MP_Allop_2.0, which
is a maximum parsimony approach to inferring phylogenetic networks that contain
allopolyploid, polyploid, and autopolyploid species, given a set of gene trees.

The numerical core -- MUL-tree construction, allele-map enumeration and the
extra-lineage (deep coalescence) score of gene trees against a MUL tree --
runs in the compiled core (``phynetpy._core``); this module keeps the public
classes and the search glue.

Last Edit: 3/4/26
Included in version : 1.0.0

Docs   - [ ]
Tests  - [x]
Design - [x]

"""
from __future__ import annotations
from typing import Any

import numpy as np

from . import _core
from .Network import Network, Node, Edge, MUL, NetworkError
from .BirthDeath import Yule
from .ModelGraph import Model
from .MetropolisHastings import HillClimbing, Infer_MP_Allop_Kernel
from .State import State
from .GraphUtils import add_hybrid

from . import GraphUtils as utils

"""
Sources:

1)
https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000501

2)
https://doi.org/10.1371/journal.pgen.1002660

3)
https://academic.oup.com/sysbio/article/71/3/706/6380964
"""

"""
Clusters are represented a few different ways in this file.

1) frozen set of strings
2) frozen set of Nodes
3) set of Nodes
4) set of strings

(where the strings are node names)


The ILP (integer linear programming) Algorithm used to compute an MDC tree can
be found here:

https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000501

The rest of this file is based on the algorithms described here:

https://academic.oup.com/sysbio/article/71/3/706/6380964

"""
seed = np.random.randint(0, 1000)
RNG = np.random.default_rng(seed)

###########################
#### EXCEPTION HANDLER ####
###########################

class InferAllopError(Exception):
    """
    Exception class for Infer_MP_Allop related operation errors.
    """
    def __init__(self, message: str = "Error during the execution of "
                                       "Infer_MP_Allop_2.0") -> None:
        """Create an ``InferAllopError`` with a custom message.

        Args:
            message (str, optional): Error message. Defaults to a generic
                Infer_MP_Allop_2.0 execution error.
        """
        self.message = message
        super().__init__(self.message)

#########################
#### HELPER FUNCTIONS ###
#########################

def _ploidy_dif(goal: dict[Node, int], cur: dict[Node, int]) -> bool:
    """
    Checks if cur has achieved goal ploidy.

    Args:
        goal (dict[Node, int]): map of leaf nodes to their desired ploidy
        cur (dict[Node, int]): map of leaf nodes to their current ploidy

    Returns:
        bool: True if goal and cur are equivalent mappings.
    """
    for leaf in goal.keys():
        if cur[leaf] != goal[leaf]:
            return False
    return True

def _attach(net: Network,
            nti: dict[Node, int],
            goal_ploidy: dict[Node, int],
            rng: np.random.Generator = None) -> None:
    """
    Given a set of nodes that need to be edited as to increase ploidy, add a
    hybrid edge in such a location as to increase the ploidy of a maximally
    sized set of nodes that need to be improved.

    Args:
        net (Network): The Network.
        nti (dict[Node, int]): A map of nodes in net to the amount of ploidy
                               that the node needs to gain.
        goal_ploidy (dict[Node, int]): Target ploidy per leaf.
    """
    clusters: set[tuple[Node]] = utils.get_all_clusters(net,
                                                        include_trivial=True)
    # Deterministic ordering: largest clusters first, ties broken by the
    # sorted tuple of member labels. Without the secondary key the order of
    # equal-size clusters depends on set iteration order (and thus on
    # per-process string-hash randomization), making start networks
    # irreproducible across runs even with a fixed seed.
    sorted_clusters = sorted(
        clusters,
        key=lambda c: (-len(c), tuple(sorted(node.label for node in c))),
    )

    def amt_allowed(cluster: tuple[Node], lti: dict) -> int:
        """Max ploidy gain a hybrid edge to ``cluster`` can give, bounded by its neediest member."""
        return min([lti[node] for node in cluster])

    chosen_cluster = None
    ploidy_amt = 0

    for candidate in sorted_clusters:
        max_ploidy = amt_allowed(candidate, nti)
        if max_ploidy > 0:
            chosen_cluster = candidate
            ploidy_amt = max_ploidy
            break

    if chosen_cluster is None:
        raise InferAllopError("No clusters whose taxa needed more ploidy")

    top_of_cluster: Node = net.mrca(set(chosen_cluster))
    valid_edges = net.subgenome_ct_edges(downstream_node=top_of_cluster,
                                         delta=ploidy_amt)

    valid_edge_list = list(valid_edges.keys())
    if not valid_edge_list:
        raise InferAllopError("No valid edges for ploidy attachment")

    src: Edge = random_object(valid_edge_list, rng if rng is not None else RNG)
    if top_of_cluster != net.root():
        in_edges = list(net.in_edges(top_of_cluster))
        if in_edges:
            add_hybrid(net, src, in_edges[0])
        else:
            raise InferAllopError("top_of_cluster has no parent edges")
    else:
        raise NetworkError("Cannot attach hybrid at root")


def _resolve_ploidy(net: Network,
                    subgenomes: dict[str, list[str]],
                    rng: np.random.Generator = None) -> Network:
    """
    Given a tree and a subgenome mapping of network leaves to genes, add
    reticulation edges such that each leaf has the desired ploidy.

    Args:
        net (Network): A standard binary tree. Each leaf will have ploidy 1.
        subgenomes (dict[str, list[str]]): A subgenome mapping. Maps names of
                                           subgenomes to the gene names.
    Returns:
        Network: A reconciled network with correct ploidy.
    """
    cur_ploidy = {node: net.subgenome_count(node)
                  for node in net.get_leaves()}
    goal_ploidy = {node: len(subgenomes[node.label])
                   for node in net.get_leaves()}

    max_iterations = 25
    iteration = 0

    while not _ploidy_dif(goal_ploidy, cur_ploidy) and iteration < max_iterations:
        iteration += 1

        lti: dict[Node, int] = {}
        for node in net.get_leaves():
            lti[node] = goal_ploidy[node] - cur_ploidy[node]

        _attach(net, lti, goal_ploidy, rng=rng)

        cur_ploidy = {node: net.subgenome_count(node)
                      for node in net.get_leaves()}

    return net

def random_object(mylist: list[Any], rng: np.random.Generator) -> object:
    """
    Select a random object from a list using a numpy rng.

    Args:
        mylist (list[Any]): any list of objects
        rng (np.random.Generator): numpy default rng object

    Returns:
        object: randomly selected element from mylist
    """
    if not mylist:
        return None
    rand_index: int = rng.integers(0, len(mylist))
    return mylist[rand_index]

def cluster_as_name_set(cluster: set[Node]) -> frozenset[str]:
    """
    Convert cluster from a set of nodes to a set of strings (names).

    Args:
        cluster (set[Node]): One form of a "cluster"

    Returns:
        frozenset[str]: The set of node names in the cluster.
    """
    return frozenset([node.label for node in cluster])

def clusters_contains(cluster: set[Node],
                      set_of_clusters: set[set[Node]]) -> bool:
    """
    Check if a cluster is in a set of clusters by checking names.

    Args:
        cluster (set[Node]): a cluster
        set_of_clusters (set[set[Node]]): a set of clusters

    Returns:
        bool: True if cluster is an element of set_of_clusters.
    """
    names = cluster_as_name_set(cluster)
    for item in set_of_clusters:
        if cluster_as_name_set(item) == names:
            return True
    return False

def cluster_partition(cluster: frozenset[Node],
                      processed: dict[set[Node], Node]) -> frozenset:
    """
    Given a cluster such as ('A', 'B', 'C'), if a cluster such as ('A', 'B')
    has already been processed, split the original cluster into subsets --
    {('A', 'B'), ('C')}.

    Args:
        cluster (frozenset[Node]): A cluster.
        processed (dict[set[Node], Node]): A mapping from clusters to the Node
                                           obj that is the root of that cluster.

    Returns:
        frozenset: the partitioned cluster
    """
    editable_cluster = set(cluster)
    new_cluster = set()

    for subcluster in processed.keys():
        if subcluster.issubset(editable_cluster):
            new_cluster.add(subcluster)
            for item in subcluster:
                editable_cluster.remove(item)

    for item in editable_cluster:
        new_cluster.add(item)

    return frozenset(new_cluster)

def generate_tree_from_clusters(tree_clusters: set[str]) -> Network:
    """
    Given a set of clusters (given by the taxa labels), reconstruct the tree
    it represents.

    Args:
        tree_clusters (set[str]): A set of tree clusters.
    Returns:
        Network: the MDC network.
    """
    net: Network = Network()
    root: Node = Node(name="ROOT")
    net.add_nodes(root)

    i = 2
    j = 1
    is_root = False
    processed_clusters: dict[set, Node] = {}
    root_children: list[Node] = []

    while len(tree_clusters) != 0:
        clusters_len_i = [c for c in tree_clusters if len(c) == i]

        if len(tree_clusters) == len(clusters_len_i):
            is_root = True

        for cluster in clusters_len_i:
            cluster_parent = Node(name=f"Internal_{j}")
            net.add_nodes(cluster_parent)
            j += 1

            if is_root:
                root_children.append(cluster_parent)

            partitioned_cluster = cluster_partition(cluster, processed_clusters)

            for subtree in partitioned_cluster:
                if type(subtree) == frozenset:
                    new_edge = Edge(cluster_parent, processed_clusters[subtree])
                    net.add_edges(new_edge)
                else:
                    taxa_node = Node(name=subtree)
                    net.add_nodes(taxa_node)
                    net.add_edges(Edge(cluster_parent, taxa_node))

            processed_clusters[cluster] = cluster_parent
            tree_clusters.remove(cluster)

        i += 1

    for root_child in root_children:
        net.add_edges(Edge(root, root_child))

    return net

def partition_gene_trees(gene_map: dict[str, list[str]],
                         rng: np.random.Generator = None) -> Network:
    """
    Generate a starting network given a subgenome mapping.

    Args:
        gene_map (dict[str, list[str]]): Map from gene names to a list of
                                         copy names.
        rng (np.random.Generator): A random generator instance.

    Returns:
        Network: A bootstrapped starting network with correct ploidy values.
    """
    if rng is None:
        rng = RNG
    yule_generator: Yule = Yule(.1, len(gene_map.keys()), rng=rng)
    simple_network: Network = yule_generator.generate_network()

    name_idx = 0
    names = list(gene_map.keys())
    for leaf in simple_network.get_leaves():
        simple_network.update_node_name(leaf, names[name_idx])
        name_idx += 1

    return _resolve_ploidy(simple_network, gene_map, rng=rng)

def get_other_copies(gene_tree_leaf: Node,
                     gene_map: dict[str, list[str]]) -> list[str]:
    """
    Return the MUL/subgenome allele labels that a gene-tree leaf may map to.

    Gene-tree tip names are matched in this order:

    1. As an allele label in any ``gene_map`` value list (typical when tips
       are already named like ``C_a`` / ``01xA``).
    2. As a species key in ``gene_map`` (typical when tips reuse the network
       leaf names, e.g. tip ``C`` with ``{"C": ["C_a", "C_b"]}``).

    Allele labels themselves may be any strings; they need not resemble the
    species name.

    Raises:
        InferAllopError: if the tip matches neither a value nor a species key.

    Args:
        gene_tree_leaf (Node): Leaf node of a gene tree
        gene_map (dict[str, list[str]]): species -> allele/subgenome labels

    Returns:
        list[str]: Allele labels available for this tip's species
    """
    label = gene_tree_leaf.label

    for copy_names in gene_map.values():
        if label in copy_names:
            return copy_names

    if label in gene_map:
        return gene_map[label]

    raise InferAllopError(
        f"Gene-tree leaf '{label}' is not in the allele/subgenome map. "
        f"Each tip must either equal a species key "
        f"({sorted(gene_map.keys())}) or appear in that species' allele "
        f"list (map values)."
    )

def _leaf_choices(g: Network,
                  gene_map: dict[str, list[str]]) -> list[tuple[str, list[str]]]:
    """The allele labels each leaf of ``g`` may map to, in leaf order."""
    return [(gleaf.label, list(get_other_copies(gleaf, gene_map)))
            for gleaf in g.get_leaves()]

def allele_map_set(g: Network,
                   gene_map: dict[str, list[str]]) -> list[AlleleMap]:
    """
    Let a MUL tree, T', have taxa labels drawn from the set X
    (keys of gene_map input dict). Calculate all possible mappings from taxa
    labels of g (values of gene_map input dict) into X.

    Args:
        g (Network): A gene tree
        gene_map (dict[str, list]): A taxon map.

    Returns:
        list[AlleleMap]: a list of functions that map labels of g to
                         labels of a MUL tree.
    """
    return _core._enumerate_allele_maps(_leaf_choices(g, gene_map))

def allele_map_set_ilp(g: Network,
                       gene_map: dict[str, list[str]],
                       max_solutions: int = 10) -> list[AlleleMap]:
    """
    Compute up to ``max_solutions`` valid allele maps for a gene tree.

    The maps are enumerated in the compiled core.  When no more than
    ``max_solutions`` exist, all of them are returned; otherwise a subset of
    ``max_solutions`` distinct maps spread evenly across the enumeration is
    returned, so the sample varies the assignment of every polyploid taxon
    rather than only the last few.  (Earlier versions solved an integer
    program with PuLP / CBC for the oversize case; it returned an arbitrary
    solver-dependent subset of the same size and dominated the running time
    of MP-allop inference.  The name is kept for compatibility.)

    Args:
        g (Network): A gene tree
        gene_map (dict[str, list[str]]): A taxon map.
        max_solutions (int): Maximum number of maps to return.

    Raises:
        InferAllopError: If no valid allele map exists.

    Returns:
        list[AlleleMap]: a list of functions that map labels of g to
                         labels of a MUL tree.
    """
    choices = _leaf_choices(g, gene_map)
    maps = _core._enumerate_allele_maps(choices, max_solutions)
    if maps is None:
        maps = _core._select_allele_maps(choices, max_solutions)
    if not maps:
        raise InferAllopError("No valid allele map found")
    return maps

##############################
### MUL TREE & Allele Maps ###
##############################

class Allop_MUL(MUL):
    """
    A Standard MUL tree with methods for calculating the maximum parsimony
    score for Infer_MP_Allop_2.0.

    The MUL-tree construction and the parsimony traversal run in the
    compiled core; ``self.mul`` is the current MUL tree (a Network).
    """
    # Per-score caches of the pure-Python traversal; kept for compatibility,
    # the compiled traversal indexes the MUL tree itself.
    _mul_leaf_set: set = set()
    _mul_root = None

    def to_mul(self, net: Network) -> Network:
        """
        Creates a (MU)lti-(L)abeled Species Tree from a network

        Raises:
            NetworkError: If the network is malformed with regards to ploidy
        Args:
            net (Network): A Network
        Returns:
            Network: a MUL tree (as a Network obj)
        """
        self.mul = _core._allop_to_mul(net, self.gene_map)
        return self.mul

    def extra_lineages(self,
                       coal_event_map: dict[tuple, str],
                       f: AlleleMap) -> int:
        """
        Computes the number of extra lineages in a mapping from a gene tree T,
        into a MUL tree, T'.

        Args:
            coal_event_map (dict[tuple[str, str], str]): A mapping from edges in
                                                         T' to a list of nodes
                                                         of T that have been
                                                         mapped into that edge.
            f (AlleleMap): An allele map.

        Returns:
            int: number of extra lineages
        """
        return _core._allop_extra_lineages(self.mul, coal_event_map, f)

    def xl_helper(self,
                  start_node: Node,
                  edge_xl_map: dict,
                  coal_map: dict[tuple[str, str], str],
                  f: AlleleMap) -> None:
        """
        Recursive helper that computes lineage counts for extra lineage
        calculation.

        Args:
            start_node (Node): Node to process
            edge_xl_map (dict): Mapping from edges to lineage counts [bottom, top]
            coal_map (dict): Mapping from edges to gene tree internal node names
            f (AlleleMap): allele mapping
        """
        _core._allop_xl_helper(self.mul, start_node, edge_xl_map, coal_map, f)

    def gene_tree_map(self,
                      g: Network,
                      leaf_map: AlleleMap,
                      mrca_cache: dict[frozenset[Node], Node]
                      ) -> dict[tuple, list[str]]:
        """
        Maps a gene tree (T) into a MUL tree (T').

        Args:
            g (Network): The gene tree
            leaf_map (AlleleMap): Allele map from gene tree to MUL tree.
            mrca_cache (dict): A cache for MRCA computations. Unused; the
                               compiled core resolves MRCAs directly.
        Returns:
            dict[tuple, list[str]]: Mapping from edges in T' to gene tree nodes.
        """
        return _core._allop_gene_tree_map(self.mul, g, leaf_map)

    def XL(self, g: Network, mrca_cache: dict[frozenset[Node], Node]) -> int:
        """
        Computes the minimum number of extra lineages over all possible
        allele maps for gene tree g mapped into this MUL tree.

        Args:
            g (Network): A gene tree
            mrca_cache (dict): A cache for MRCA computations. Unused; the
                               compiled core resolves MRCAs directly.
        Returns:
            int: the minimum number of extra lineages (>= 0)
        """
        return _core._allop_xl(self.mul, g)

    def XL_Allele(self,
                  g: Network,
                  f: AlleleMap,
                  mrca_cache: dict[frozenset[Node], Node]) -> int:
        """
        Compute the extra lineages for a specific allele mapping.

        Args:
            g (Network): A gene tree
            f (AlleleMap): An allele map from leaves of g to leaves of T
            mrca_cache (dict): A cache for MRCA computations. Unused; the
                               compiled core resolves MRCAs directly.
        Returns:
            int: number of extra lineages
        """
        return _core._allop_xl_allele(self.mul, g, f)

    def score(self, gt_list: list[Network]) -> int:
        """
        Compute the total parsimony score over a list of gene trees.

        Args:
            gt_list (list[Network]): a list of gene trees

        Returns:
            int: The MUL tree score
        """
        return _core._allop_score(self.mul, gt_list)


# Data structure that holds a mapping from gene tree leaf names to MUL tree
# leaf names (``map``), and the MUL leaves already taken (``disallowed``), so
# gene copies are never mapped to the same subgenome.  Implemented in the
# compiled core; ``AlleleMap()`` creates an empty map and ``put(g_leaf,
# mul_leaf)`` returns 1 on success, 0 when ``mul_leaf`` is already taken.
AlleleMap = _core.AlleleMap

######################################
#### LIKELIHOOD SCORER FOR MODEL  ####
######################################

class MPAllopScorer:
    """
    Callable scorer that integrates with Model.set_likelihood_calculator().
    Computes the MP Allop parsimony score (negated for maximization).
    """

    def __init__(self,
                 gene_map: dict[str, list[str]],
                 gene_trees: list[Network],
                 rng: np.random.Generator) -> None:
        """Create a new ``MPAllopScorer``.

        Args:
            gene_map: Subgenome mapping from species to their gene-copy
                labels, forwarded to the internal :class:`Allop_MUL`.
            gene_trees: Gene trees to reconcile against each candidate
                network's multiple-labelled tree.
            rng: Random generator used when resolving allele-map ties.
        """
        self.mul = Allop_MUL(gene_map, rng)
        self.gene_trees = gene_trees

    def __call__(self, model: Model) -> float:
        """
        Compute the parsimony score. Returns negated score since hill climbing
        maximizes but parsimony should be minimized.

        If the network is malformed (e.g. wrong leaf count after a move),
        returns ``-inf`` so the hill climber will reject the proposal.
        """
        net = model.network
        if net is None:
            return float("-inf")

        try:
            self.mul.to_mul(net)
        except Exception:
            return float("-inf")

        if self.mul.gene_map is None:
            return float("-inf")

        try:
            return -1 * self.mul.score(self.gene_trees)
        except Exception:
            return float("-inf")

#####################
#### MODEL BUILD ####
#####################

def build_mp_allop_model(network: Network,
                         gene_map: dict[str, list[str]],
                         gene_trees: list[Network],
                         rng: np.random.Generator) -> Model:
    """
    Build the model for Infer MP Allop 2.0.

    Attaches the network and registers the parsimony scorer that the search
    driver calls through :meth:`~phynetpy.ModelGraph.Model.likelihood`.

    Args:
        network (Network): A starting network.
        gene_map (dict[str, list[str]]): A subgenome mapping.
        gene_trees (list[Network]): A set of gene trees.
        rng (np.random.Generator): random number generator.
    Returns:
        Model: The model, ready to search.
    """
    # ``rng`` seeds the scorer only.  Move sampling draws from the model's own
    # RNG, which is left OS-seeded here, so a search is not reproducible from
    # ``rng`` alone.
    model = Model()
    model.network = network
    model.set_likelihood_calculator(MPAllopScorer(gene_map, gene_trees, rng))
    return model


class InferMPAllop:
    """
    Class that sets up and runs the Infer_MP_Allop_2.0 inference method.
    """

    def __init__(self,
                 network: Network,
                 gene_map: dict[str, list[str]],
                 gene_trees: list[Network],
                 iter_ct: int,
                 rng: np.random.Generator) -> None:
        """
        Initialize the parameters for this method call.

        Args:
            network (Network): A starting network.
            gene_map (dict[str, list[str]]): A subgenome mapping.
            gene_trees (list[Network]): A set of gene trees.
            iter_ct (int): Number of iterations to run the chain.
            rng (np.random.Generator): random number generator.
        """
        for gene_tree in gene_trees:
            allele_funcs = allele_map_set_ilp(gene_tree, gene_map)
            gene_tree.put_item("allele maps", allele_funcs)

            leaf_descendants = gene_tree.leaf_descendants_all()
            gene_tree.put_item("leaf descendants", leaf_descendants)

        # Fail fast if the allele map cannot label the MUL tips of the
        # starting network (e.g. identity map on a hybrid taxon). Without
        # this, MPAllopScorer would swallow the error as -inf and search
        # would look broken rather than misconfigured.
        try:
            Allop_MUL(gene_map, rng).to_mul(network)
        except NetworkError as exc:
            raise InferAllopError(str(exc)) from exc

        self.mp_allop_model: Model = build_mp_allop_model(
            network, gene_map, gene_trees, rng
        )
        self.iter_ct = iter_ct
        self.results: dict = {}

    def run(self) -> float:
        """
        Computes the network with the lowest parsimony score over the set
        of given gene trees.

        Returns:
            float: parsimony score of the most likely Network
        """
        hc = HillClimbing(Infer_MP_Allop_Kernel(),
                          num_iter=self.iter_ct,
                          model=self.mp_allop_model,
                          enhanced_stop=True)

        end_state: State = hc.run()

        self.results = hc.nets_2_scores
        return end_state.likelihood()


#################
#### METHODS ####
#################

def allop_parsimony_score(network: Network,
                          gene_trees: list[Network],
                          subgenome_map: dict[str, list[str]],
                          rng: np.random.Generator = None) -> int:
    """
    Compute the allopolyploid parsimony score of a network.

    The numerical core of the ``(GeneTrees, Allopolyploid, MDC)`` cell:
    expands *network* into its multiple-labelled (MUL) tree and counts the
    extra gene lineages needed to reconcile the gene trees with it.  Reached
    from the public API as ``score(net, gene_trees,
    model=Allopolyploid(...), criterion=MDC())``.

    Args:
        network (Network): The species network to score.
        gene_trees (list[Network]): The gene trees to reconcile.
        subgenome_map (dict[str, list[str]]): Subgenome-to-genes mapping.
        rng (np.random.Generator, optional): RNG used when resolving allele
                                             maps. Defaults to a fresh
                                             generator.

    Returns:
        int: The parsimony score (extra lineage count; lower is better).

    Raises:
        InferAllopError: If the subgenome map cannot label the MUL tips of
                         *network*.
    """
    rng = rng if rng is not None else np.random.default_rng()

    T = Allop_MUL(subgenome_map, rng)
    try:
        T.to_mul(network)
    except NetworkError as exc:
        raise InferAllopError(str(exc)) from exc

    for gene_tree in gene_trees:
        allele_funcs = allele_map_set_ilp(gene_tree, subgenome_map)
        gene_tree.put_item("allele maps", allele_funcs)
        gene_tree.put_item("leaf descendants", gene_tree.leaf_descendants_all())

    return T.score(gene_trees)
