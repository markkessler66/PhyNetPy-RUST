"""Input checks for infer_dimple. Failures here are raised before any cut."""

import pytest

from phynetpy.Network import Edge, Network, Node
from phynetpy.data import GeneTrees
from phynetpy.data._base import DataError

from phynetpy._dimple import DimpleError, partition_scaffold, validate_inputs


def _binary() -> Network:
    return Network.from_newick("((a,b),(c,d));")


def _scaffold() -> Network:
    return Network.from_newick("(a,b,c,d);")


def _genes() -> GeneTrees:
    return GeneTrees(trees=[_binary()])


def test_subset_size_and_division_count_raise_before_a_cut():
    scaffold = _scaffold()
    with pytest.raises(DimpleError):
        partition_scaffold(scaffold, max_subset_size=3)
    with pytest.raises(DimpleError):
        partition_scaffold(scaffold, n_divisions=0)
    with pytest.raises(DimpleError):
        validate_inputs(_genes(), scaffold, _binary(), max_subset_size=3)


def test_reticulation_on_the_scaffold_raises():
    hybrid = Node("#H0", is_reticulation=True)
    root, left, right = Node("Root"), Node("L"), Node("R")
    a, b, c = Node("a"), Node("b"), Node("c")
    net = Network()
    net.add_nodes([hybrid, root, left, right, a, b, c])
    net.add_edges([
        Edge(root, left), Edge(root, right),
        Edge(left, a), Edge(left, hybrid), Edge(hybrid, b),
        Edge(right, c), Edge(right, hybrid),
    ])
    with pytest.raises(DimpleError, match="reticulation"):
        partition_scaffold(net)


def test_start_tree_must_be_binary_and_cover_the_scaffold():
    genes = _genes()
    scaffold = _scaffold()
    with pytest.raises(DimpleError, match="not binary"):
        validate_inputs(genes, scaffold, scaffold)
    short = Network.from_newick("((a,b),c);")
    with pytest.raises(DimpleError, match="missing"):
        validate_inputs(genes, scaffold, short)


def test_a_gene_tree_that_is_a_network_is_refused():
    hybrid = Node("#H0", is_reticulation=True)
    root, left, right = Node("Root"), Node("L"), Node("R")
    a, b, c = Node("a"), Node("b"), Node("c")
    net = Network()
    net.add_nodes([hybrid, root, left, right, a, b, c])
    net.add_edges([
        Edge(root, left), Edge(root, right),
        Edge(left, a), Edge(left, hybrid), Edge(hybrid, b),
        Edge(right, c), Edge(right, hybrid),
    ])
    with pytest.raises(DataError):
        GeneTrees(trees=[net])


def test_incomplete_gene_trees_are_not_an_error():
    genes = GeneTrees(trees=[Network.from_newick("((a,b),c);")])
    validate_inputs(genes, _scaffold(), _binary())


def test_a_missing_outgroup_label_is_a_no_op():
    scaffold = _scaffold()
    plain = partition_scaffold(scaffold, n_divisions=1, seed=0)
    labeled = partition_scaffold(scaffold, n_divisions=1, seed=0, outgroup="ZZZ")
    assert plain[0].divisions == labeled[0].divisions


def test_a_present_outgroup_is_removed_from_every_subset():
    scaffold = Network.from_newick("(OUT,(a,b,c,d));")
    jobs = partition_scaffold(scaffold, n_divisions=1, seed=0, outgroup="OUT")
    subsets = [subset for division in jobs[0].divisions for subset in division]
    assert subsets
    assert all("OUT" not in subset for subset in subsets)
    assert jobs[0].tob_leaves == frozenset("abcd")
