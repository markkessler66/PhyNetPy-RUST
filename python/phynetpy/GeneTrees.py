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
Last Edit : 9/16/25
First Included in Version : 2.0.0

Docs   - [ ]
Tests  - [ ]
Design - [x]

The analysis routines (mapping resolution, cluster and split support,
consensus trees, Robinson-Foulds distances, the modal gene tree, gene
concordance factors, duplication-loss reconciliation and the Newick ->
Network conversion) run in the compiled core (``phynetpy._core``); the
:class:`GeneTrees` container itself stays a Python class so that its
``trees``, ``taxa_names``, mapping and naming-rule attributes are plain
Python state.
"""

import os
import tempfile
import subprocess
from typing import Any, Callable, Dict, FrozenSet, List, Literal, Optional, Set, Tuple

from .Network import Network, Node, Edge
from . import _core

#########################
#### EXCEPTION CLASS ####
#########################

class GeneTreeError(Exception):
    """
    Error class for all errors relating to gene trees.
    """
    def __init__(self, message : str = "Gene Tree Module Error") -> None:
        """
        Initialize a GeneTreeError with a message.

        Args:
            message (str, optional): Custom error message. Defaults to
                                     "Gene Tree Module Error".
        Returns:
            N/A
        """
        super().__init__(message)
        self.message = message


# The core raises GeneTreeError itself (e.g. when no gene tree can seed a
# search), so ``except GeneTreeError`` keeps working for errors from Rust.
_core._gt_register_gene_tree_error(GeneTreeError)

##########################
#### HELPER FUNCTIONS ####
##########################

def phynetpy_naming(taxa_name: str) -> str:
    """
    The default method for sorting taxa labels into groups

    Args:
        taxa_name (str): a taxa label from a nexus file
    Raises:
        GeneTreeError: if there is a problem applying the naming rule

    Returns:
        str: a string that is the key for this label
    """
    if not taxa_name[0:2].isnumeric():
        raise GeneTreeError("Error Applying PhyNetPy Naming Rule: \
                             first 2 digits is not numerical")

    if taxa_name[2].isalpha():
        return taxa_name[2].upper()
    else:
        raise GeneTreeError("Error Applying PhyNetPy Naming Rule: \
                             3rd position is not an a-z character")


####################
#### GENE TREES ####
####################

class GeneTrees:
    """
    A container for a set of networks that are binary and represent a
    gene tree.

    Supports two strategies for grouping gene labels into species:
      1. An explicit ``species_gene_mapping`` dict (species -> gene labels).
      2. A ``naming_rule`` callable that derives the species key from a
         gene label string.

    When neither is provided every gene label is treated as its own species
    (identity mapping).
    """

    def __init__(
        self,
        gene_tree_list: Optional[List[Network]] = None,
        naming_rule: Optional[Callable[..., Any]] = None,
        species_gene_mapping: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """
        Wrapper class for a set of networks that represent gene trees.

        Args:
            gene_tree_list (list[Network], optional): A list of networks,
                of the binary tree variety. Defaults to None.
            naming_rule (Callable[..., Any], optional): A function
                f : str -> str that maps a gene label to its species key.
                Ignored when *species_gene_mapping* is provided.
                Defaults to None (identity mapping).
            species_gene_mapping (dict[str, list[str]], optional): Explicit
                mapping from species name to a list of gene/allele labels.
                Takes priority over *naming_rule* when both are given.
        """

        self.trees: Set[Network] = set[Network]()
        self.taxa_names: Set[str] = set[str]()
        self._species_gene_mapping: Optional[Dict[str, List[str]]] = (
            species_gene_mapping
        )
        self.naming_rule: Optional[Callable[..., Any]] = naming_rule

        if gene_tree_list is not None:
            for tree in gene_tree_list:
                self.add(tree)

    def add(self, tree: Network) -> None:
        """
        Add a gene tree to the collection. Any new gene labels that belong to
        this tree will also be added to the collection of all
        gene tree leaf labels.

        Args:
            tree (Network): A network that is a tree, must be binary.
        """

        self.trees.add(tree)

        for leaf in tree.get_leaves():
            self.taxa_names.add(leaf.label)

    @property
    def species_gene_mapping(self) -> Optional[Dict[str, List[str]]]:
        """Return the explicit species-to-gene mapping, if set."""
        return self._species_gene_mapping

    @species_gene_mapping.setter
    def species_gene_mapping(self, value: Optional[Dict[str, List[str]]]) -> None:
        """Set the explicit species-to-gene mapping (``None`` to clear it)."""
        self._species_gene_mapping = value

    def _resolve_mapping(self) -> Dict[str, List[str]]:
        """
        Build the species -> gene-labels mapping using the best available
        strategy.

        Priority:
          1. Explicit ``species_gene_mapping`` (filtered to present taxa).
          2. ``naming_rule`` callable applied to each taxon.
          3. Identity mapping (each label is its own species).

        Returns:
            dict[str, list[str]]: species -> gene label mapping.
        """
        return _core._gt_resolve_mapping(
            self.taxa_names, self._species_gene_mapping, self.naming_rule
        )

    def mp_allop_map(self) -> Dict[str, List[str]]:
        """
        Create a subgenome mapping from the stored set of gene trees.

        Uses the explicit species_gene_mapping if available, otherwise
        falls back to the naming_rule, and finally to an identity mapping.

        Returns:
            dict[str, list[str]]: subgenome mapping
        """
        return self._resolve_mapping()

    def validate_mapping(self) -> List[str]:
        """
        Check the explicit species_gene_mapping against the actual taxa
        present in the gene trees.

        Returns:
            list[str]: A list of warning/error messages. Empty means valid.
        """
        return _core._gt_validate_mapping(
            self.taxa_names, self._species_gene_mapping
        )

    def cluster_support(self,
                        include_trivial: bool = False,
                        normalize: bool = True) -> Dict[FrozenSet[str], float]:
        """
        Aggregate support for all rooted clusters across the gene tree set.

        Args:
            include_trivial (bool): include size-1 clusters. Defaults to False.
            normalize (bool): return frequencies in [0,1] instead of counts.

        Returns:
            dict[FrozenSet[str], float]: map cluster (as frozenset of leaf labels)
                                         to count or frequency.
        """
        return _core._gt_cluster_support(self.trees, include_trivial, normalize)

    def split_support(self, normalize: bool = True) -> Dict[FrozenSet[str], float]:
        """
        Aggregate support for unrooted splits (bipartitions), canonicalized
        to the smaller side of the split.

        Note: Only non-trivial splits (both parts size >= 2) are counted.

        Args:
            normalize (bool): return frequencies in [0,1] instead of counts.

        Returns:
            dict[FrozenSet[str], float]: map smaller-side cluster to count/freq.
        """
        return _core._gt_split_support(self.trees, normalize)

    def support_on_reference(self,
                             ref_tree: Network,
                             include_trivial: bool = False,
                             normalize: bool = True) -> Dict[FrozenSet[str], float]:
        """
        Compute support of each rooted cluster present in a reference tree.

        Args:
            ref_tree (Network): reference binary tree.
            include_trivial (bool): include size-1 clusters. Defaults to False.
            normalize (bool): return frequencies in [0,1].

        Returns:
            dict[FrozenSet[str], float]: map of ref clusters to support.
        """
        return _core._gt_support_on_reference(
            self.trees, ref_tree, include_trivial, normalize
        )

    def annotate_reference_support(self,
                                   ref_tree: Network,
                                   include_trivial: bool = False,
                                   normalize: bool = True) -> None:
        """
        Annotate the reference tree's internal edges with support values stored
        in the edge weight field.

        For edge (u->v), the cluster is the set of leaf descendants of v.

        Args:
            ref_tree (Network): reference binary tree to annotate in place.
            include_trivial (bool): include size-1 clusters. Defaults to False.
            normalize (bool): frequencies vs counts.
        """
        _core._gt_annotate_reference_support(
            self.trees, ref_tree, include_trivial, normalize
        )

    def consensus_clusters(self,
                           threshold: float = 0.5,
                           include_trivial: bool = False) -> List[Set[str]]:
        """
        Return the set of rooted clusters whose support >= threshold.

        Note: This does not resolve incompatibilities; it is a simple filter.

        Args:
            threshold (float): minimum frequency in [0,1].
            include_trivial (bool): include size-1 clusters.

        Returns:
            list[Set[str]]: clusters passing threshold.
        """
        return _core._gt_consensus_clusters(self.trees, threshold, include_trivial)

    def rf_distance(self, ref_tree: Network, normalize: bool = False) -> float:
        """
        Compute the average Robinson-Foulds distance (rooted, using clusters)
        between each gene tree and the reference.

        Args:
            ref_tree (Network): reference tree.
            normalize (bool): if True, divide by the maximum possible RF for the
                              shared taxon set of each comparison.

        Returns:
            float: average RF distance across the gene trees.
        """
        return _core._gt_rf_distance(self.trees, ref_tree, normalize)

    def most_frequent_gene_tree(
        self,
        *,
        include_trivial: bool = False,
        restrict_to_taxa: Optional[Set[str]] = None,
        tiebreak: Literal["first", "random"] = "first",
        seed: Optional[int] = None,
    ) -> Network:
        """Return a deep copy of the most frequent gene tree topology.

        Gene trees are grouped by the frozenset of their rooted clusters
        (topology signature, ignoring branch lengths and internal-node
        labels). The largest group wins; a representative of that group is
        returned. Guaranteed binary iff the input gene trees are binary.

        This is generally a better MPL search seed than a majority-rule
        consensus because consensus trees are often non-binary (unresolved
        polytomies) and hence give pathologically low starting MPL scores.

        Args:
            include_trivial: Include size-1 clusters (leaves) in the
                topology signature. Defaults to False (only informative
                clusters matter for the topology class).
            restrict_to_taxa: If provided, only gene trees whose leaf set
                equals this taxa set participate. Useful when gene trees
                have ragged taxon coverage and you want a seed that covers
                the full species set.
            tiebreak: On ties between multiple topologies with the same
                maximum frequency, either pick the first one encountered
                (``"first"``) or sample uniformly (``"random"``).
            seed: RNG seed when ``tiebreak="random"``.

        Returns:
            Network: Deep copy of a representative gene tree from the modal
            topology class.

        Raises:
            GeneTreeError: If no gene trees satisfy ``restrict_to_taxa``.
        """
        return _core._gt_most_frequent_gene_tree(
            self.trees,
            include_trivial,
            restrict_to_taxa,
            tiebreak,
            seed,
        )

    def build_majority_rule_consensus_tree(self,
                                           threshold: float = 0.5) -> Network:
        """
        Construct a majority-rule (or threshold) consensus tree from the gene trees.

        Greedy compatibility: sort clusters by support descending; add if
        compatible with current set; then realize the set into a (possibly
        multifurcating) tree over the union of taxa.
        """
        return _core._gt_build_majority_rule_consensus_tree(
            self.taxa_names, self.trees, threshold
        )

    def _clusters_to_tree(self, taxa: Set[str], clusters: Set[FrozenSet[str]]) -> Network:
        """
        Realize a compatible set of clusters as a (possibly multifurcating) tree.

        Strategy: recursively partition taxa by maximal proper clusters within
        the current subset. Singleton taxa become leaves.
        """
        return _core._gt_clusters_to_tree(taxa, clusters)

    def gene_concordance_factors(self, ref_tree: Network) -> Dict[Tuple[str, str], float]:
        """
        Compute a split-based concordance factor per internal edge of the reference.

        For each edge (u->v) with split (A | B), count across gene trees the
        fraction of informative trees (with at least one taxon from A and B)
        that contain the induced split on their leaf set. Returns a map keyed by
        (min(child_label), max(child_label)) of the edge child cluster's name
        representative to gCF. If the child is an internal node, the key uses a
        canonical name derived from its cluster (string-joined).
        """
        return _core._gt_gene_concordance_factors(self.trees, ref_tree)

    def astral(self,
                astral_jar_path: str,
                mapping_rule: Optional[Callable[[str], str]] = None,
                extra_args: Optional[List[str]] = None) -> Network:
        """
        Infer a species tree using ASTRAL from the stored gene trees.

        Requires a path to the ASTRAL .jar. We write trees to a temp file and a
        multi-allele mapping, then call ASTRAL and parse the result.

        The mapping is resolved in this order:
          1. The instance-level ``species_gene_mapping`` (if set).
          2. The *mapping_rule* argument passed here.
          3. The instance-level ``naming_rule``.
          4. Identity mapping (each gene = its own species).

        Args:
            astral_jar_path (str): Path to the ASTRAL jar file.
            mapping_rule (Callable, optional): Override callable for
                deriving species from gene labels.  Ignored when an
                explicit species_gene_mapping is available.
            extra_args (list[str], optional): Additional CLI args for ASTRAL.
        """
        if extra_args is None:
            extra_args = []

        if mapping_rule is not None and self._species_gene_mapping is None:
            old_rule = self.naming_rule
            self.naming_rule = mapping_rule
            species_map = self._resolve_mapping()
            self.naming_rule = old_rule
        else:
            species_map = self._resolve_mapping()

        with tempfile.TemporaryDirectory() as tmpdir:
            trees_path = os.path.join(tmpdir, "genes.tre")
            map_path = os.path.join(tmpdir, "mapping.txt")
            out_path = os.path.join(tmpdir, "astral_out.tre")

            with open(trees_path, "w") as f:
                for t in self.trees:
                    f.write(t.newick())
                    if not t.newick().strip().endswith(";"):
                        f.write(";")
                    f.write("\n")

            with open(map_path, "w") as f:
                for sp, genes in species_map.items():
                    f.write(f"{sp}: {' '.join(sorted(genes))}\n")

            cmd = [
                "java", "-jar", astral_jar_path,
                "-a", map_path, "-i", trees_path, "-o", out_path,
            ]
            cmd.extend(extra_args)
            subprocess.run(
                cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            with open(out_path, "r") as f:
                newick = f.read().strip()

        return self._biopy_tree_to_network(newick)

    def _biopy_tree_to_network(self, tree: Any) -> Network:
        """
        Minimal Newick tree -> PhyNetPy Network conversion for plain trees.

        Accepts a Newick string, or any tree object exposing
        ``format("newick")`` (such as a Biopython ``Tree``); the string is
        parsed by the core with the same rules Biopython applied.
        """
        if not isinstance(tree, str):
            tree = tree.format("newick")
        return _core._gt_newick_to_network(tree)

    def duplication_loss_summary(
        self,
        species_tree: Network,
        naming_rule: Optional[Callable[[str], str]] = None,
    ) -> Dict[str, Any]:
        """
        Reconcile each gene tree against a species tree using LCA mapping and
        report total duplications and losses (parsimony-based estimate).

        The gene-to-species mapping is resolved in the same priority order
        as :meth:`_resolve_mapping`:
          1. Explicit ``species_gene_mapping``.
          2. *naming_rule* argument (if provided and no explicit mapping).
          3. Instance ``naming_rule``.
          4. Identity mapping.

        Returns:
            dict with ``"totals"`` and ``"per_tree"`` breakdowns.
        """
        if naming_rule is not None and self._species_gene_mapping is None:
            old_rule = self.naming_rule
            self.naming_rule = naming_rule
            species_map = self._resolve_mapping()
            self.naming_rule = old_rule
        else:
            species_map = self._resolve_mapping()

        return _core._gt_duplication_loss_summary(
            self.trees, species_tree, species_map
        )
