"""Compatible subset, overlap neighbor-joining, and the gene-tree distance matrix."""

import numpy as np
import pytest

from phynetpy.Network import Network
from phynetpy.data import GeneTrees

from phynetpy._dimple import (
    DimpleError,
    _Constraint,
    _outgroup_distances,
    gene_tree_distance_matrix,
    greedy_compatible,
    max_clique_compatible,
    merge_blob_networks,
    overlap_nj,
    trees_incompatible,
)
from dimple_fixtures import assert_phylogeny


def _siblings(net: Network, left: str, right: str) -> bool:
    return (
        net.get_parents(net.has_node_named(left))[0]
        is net.get_parents(net.has_node_named(right))[0]
    )


def test_a_quartet_conflict_keeps_one_tree():
    first = Network.from_newick("((a,b),(c,d));")
    second = Network.from_newick("((a,c),(b,d));")
    assert trees_incompatible(first, second)
    kept = greedy_compatible([first, second])
    assert kept == [first]


def test_overlapping_compatible_trees_keep_the_constrained_cherry():
    first = Network.from_newick("((a,b),(c,d));")
    second = Network.from_newick("((a,b),(c,e));")
    assert not trees_incompatible(first, second)
    assert greedy_compatible([first, second]) == [first, second]

    # a-c is the closest pair, so unconstrained NJ joins it. The constraint
    # ((a,b),(c,d)) forbids that cherry.
    labels = ["a", "b", "c", "d"]
    matrix = np.array([
        [0, 5, 0.1, 5],
        [5, 0, 5, 5],
        [0.1, 5, 0, 5],
        [5, 5, 5, 0],
    ], dtype=float)
    constrained, relaxed = overlap_nj([first], labels, matrix)
    free, _ = overlap_nj([], labels, matrix)
    assert relaxed == 0
    assert _siblings(constrained, "a", "b")
    assert not _siblings(constrained, "a", "c")
    assert _siblings(free, "a", "c")
    assert_phylogeny(constrained, list("abcd"))


def test_the_virtual_outgroup_roots_toward_the_near_taxon_and_is_deleted():
    labels = ["a", "b", "c"]
    matrix = np.array([[0, 2, 2], [2, 0, 2], [2, 2, 0]], dtype=float)
    merged, relaxed = overlap_nj(
        [Network.from_newick("((a,b),c);")],
        labels,
        matrix,
        outgroup_distances={"a": 0.2, "b": 10.0, "c": 10.0},
    )
    assert relaxed == 0
    assert "OUT" not in {leaf.label for leaf in merged.get_leaves()}
    # a is nearer the scaffold siblings, so it sits on the root side.
    depth = {}
    for label in labels:
        node = merged.has_node_named(label)
        hops = 0
        while merged.get_parents(node):
            hops += 1
            node = merged.get_parents(node)[0]
        depth[label] = hops
    assert depth["a"] < depth["b"]
    assert depth["a"] < depth["c"]
    assert_phylogeny(merged, list("abc"))


def test_a_taxon_in_no_constraint_tree_is_still_a_leaf():
    labels = ["a", "b", "c", "z"]
    matrix = np.ones((4, 4))
    np.fill_diagonal(matrix, 0.0)
    merged, _ = overlap_nj(
        [Network.from_newick("((a,b),c);")], labels, matrix,
    )
    assert "z" in {leaf.label for leaf in merged.get_leaves()}
    assert_phylogeny(merged, list("abcz"))


def test_a_pair_that_never_cooccurs_is_undefined_only_when_it_is_needed():
    missing = GeneTrees(trees=[
        Network.from_newick("(a:1,b:1);"),
        Network.from_newick("(a:1,c:1);"),
    ])
    with pytest.raises(DimpleError, match="never co-occur"):
        gene_tree_distance_matrix(missing, ["a", "b", "c"], strict=True)
    labels, matrix = gene_tree_distance_matrix(missing, ["a", "b", "c"], strict=False)
    assert np.isnan(matrix[labels.index("b"), labels.index("c")])

    present = GeneTrees(trees=[
        Network.from_newick("(a:1,b:1);"),
        Network.from_newick("(a:1,c:1);"),
        Network.from_newick("(b:2,c:2);"),
    ])
    labels, matrix = gene_tree_distance_matrix(present, ["a", "b", "c"])
    assert matrix[labels.index("a"), labels.index("b")] == pytest.approx(2)
    assert matrix[labels.index("b"), labels.index("c")] == pytest.approx(4)


def test_incompatible_constraints_are_counted_and_the_filter_avoids_them():
    first = Network.from_newick("((a,b),(c,d));")
    second = Network.from_newick("((a,c),(b,d));")
    labels = ["a", "b", "c", "d"]
    matrix = np.array([
        [0, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 0],
    ], dtype=float)
    _, relaxed = overlap_nj([first, second], labels, matrix)
    assert relaxed >= 1
    kept = greedy_compatible([first, second])
    _, filtered = overlap_nj(kept, labels, matrix)
    assert filtered == 0


def _rooted_clades(net: Network) -> set[frozenset[str]]:
    labels = {leaf.label for leaf in net.get_leaves()}
    clades = set()
    for node in net.V():
        below = frozenset(leaf.label for leaf in net.leaf_descendants(node))
        if 1 < len(below) < len(labels):
            clades.add(below)
    return clades


def test_outgroup_offset_roots_blob_correctly():
    # (((a,b),c),d) with sibling s above the blob. Path lengths are additive.
    # a=2.64, b=0.16, (a,b)=0.79, c=2.47, (a,b,c)=0.08, d=2.89, stem=1.28, s=2.24.
    labels = ["a", "b", "c", "d", "s"]
    pairs = {
        ("a", "b"): 2.80, ("a", "c"): 5.90, ("a", "d"): 6.40, ("a", "s"): 7.03,
        ("b", "c"): 3.42, ("b", "d"): 3.92, ("b", "s"): 4.55,
        ("c", "d"): 5.44, ("c", "s"): 6.07,
        ("d", "s"): 6.41,
    }
    matrix = np.zeros((5, 5))
    for (left, right), distance in pairs.items():
        i, j = labels.index(left), labels.index(right)
        matrix[i, j] = matrix[j, i] = distance
    blob = ["a", "b", "c", "d"]
    # scale=1.0 keeps the metric additive and roots exactly. The module
    # default is DIMPLE's 2.0, which wins on estimated gene trees (see
    # OUTGROUP_SCALE); on this exact matrix it misroots.
    additive = _outgroup_distances(blob, labels, matrix, frozenset({"s"}), scale=1.0)
    sub = matrix[np.ix_([0, 1, 2, 3], [0, 1, 2, 3])]
    rooted, _ = overlap_nj([], blob, sub, outgroup_distances=additive)
    assert _rooted_clades(rooted) == {frozenset({"a", "b"}), frozenset({"a", "b", "c"})}
    doubled = _outgroup_distances(blob, labels, matrix, frozenset({"s"}))
    assert doubled == {label: 2.0 * distance for label, distance in additive.items()}
    wrong, _ = overlap_nj([], blob, sub, outgroup_distances=doubled)
    assert _rooted_clades(wrong) != {frozenset({"a", "b"}), frozenset({"a", "b", "c"})}


def test_constraint_relaxed_join_removes_ghosts():
    constraint = _Constraint(Network.from_newick("((a,b),(c,d));"))
    constraint.join("a", "c", "NJ0")
    assert constraint.leaves == {"b", "d"}
    assert "NJ0" not in constraint.leaves
    assert constraint.cherries() == {frozenset({"b", "d"})}
    assert constraint.broken == 1


def test_constraint_cherry_join_unchanged():
    constraint = _Constraint(Network.from_newick("((a,b),(c,d));"))
    constraint.join("a", "b", "NJ0")
    assert constraint.leaves == {"NJ0", "c", "d"}
    assert constraint.cherries() == {
        frozenset({"NJ0", "c"}),
        frozenset({"NJ0", "d"}),
        frozenset({"c", "d"}),
    }
    assert constraint.broken == 0


def test_overlap_nj_relaxation_does_not_cascade():
    # a-c is the closest pair. It is a cherry of the second tree and not of
    # the first, so the first join relaxes exactly one constraint. The old
    # contract left a and c behind, and every later pair of b and d then
    # looked illegal too.
    first = Network.from_newick("((a,b),(c,d));")
    second = Network.from_newick("((a,c),(b,d));")
    labels = ["a", "b", "c", "d"]
    matrix = np.array([
        [0, 5, 0.1, 5],
        [5, 0, 5, 5],
        [0.1, 5, 0, 5],
        [5, 5, 5, 0],
    ], dtype=float)
    _, relaxed = overlap_nj([first, second], labels, matrix)
    assert relaxed == 1


def test_root_blob_rooted_by_triples():
    # a is far, so neighbor-joining attaches it last and the root is not
    # the ab|cd split. Every gene tree says that split is the root.
    gene = Network.from_newick("((a:8,b:0.1):0.1,(c:0.1,d:0.1):0.1);")
    genes = GeneTrees(trees=[gene, gene, gene, gene])
    merged, _, _ = merge_blob_networks(
        [
            Network.from_newick("((a,b),c);"),
            Network.from_newick("((c,d),b);"),
        ],
        genes,
        frozenset(),
    )
    assert _rooted_clades(merged) == {frozenset({"a", "b"}), frozenset({"c", "d"})}


def test_merge_tolerates_disjoint_siblings():
    # s1 and s2 never share a gene tree. Each co-occurs with every blob taxon.
    genes = GeneTrees(trees=[
        Network.from_newick("((a:1,b:1):1,(c:1,s1:1):1);"),
        Network.from_newick("((a:1,b:1):1,(c:1,s2:1):1);"),
    ])
    merged, _, _ = merge_blob_networks(
        [
            Network.from_newick("((a,b),c);"),
            Network.from_newick("((a,c),b);"),
        ],
        genes,
        frozenset({"s1", "s2"}),
    )
    assert {leaf.label for leaf in merged.get_leaves()} == {"a", "b", "c"}


def test_the_largest_compatible_clique_beats_first_fit():
    # The first tree conflicts with the other three on {a,b,c,d}; those three
    # agree with each other. First-fit keeps only the first tree; DIMPLE's
    # max-clique filter keeps the three.
    odd = Network.from_newick("(((a,b),(c,d)),e);")
    agree = [
        Network.from_newick("(((a,c),(b,d)),e);"),
        Network.from_newick("(((a,c),(b,d)),f);"),
        Network.from_newick("(((a,c),(b,d)),g);"),
    ]
    assert greedy_compatible([odd, *agree]) == [odd]
    assert max_clique_compatible([odd, *agree]) == agree
    assert max_clique_compatible([odd]) == [odd]


def test_the_distance_matrix_matches_dijkstra_path_lengths_and_measures_only_wanted_taxa():
    from phynetpy.GraphUtils import pairwise_leaf_distance
    from phynetpy._dimple import _leaf_path_lengths
    tree = Network.from_newick("(((a:1,b:2):0.5,c:3):1,(d:4,e:1):2);")
    want = {"a", "b", "c", "e"}
    ours = _leaf_path_lengths(tree, want)
    theirs = pairwise_leaf_distance(tree, use_branch_lengths=True)
    assert set(ours) == {tuple(sorted(pair)) for pair in theirs if set(pair) <= want}
    for pair, distance in theirs.items():
        if set(pair) <= want:
            assert abs(ours[tuple(sorted(pair))] - distance) < 1e-9
    # A tree missing a taxon contributes only to the pairs it has.
    partial = Network.from_newick("((a:1,c:1):1,e:1);")
    genes = GeneTrees(trees=[tree, partial])
    labels, matrix = gene_tree_distance_matrix(genes, ["a", "c", "e", "b"])
    i = {label: k for k, label in enumerate(labels)}
    assert abs(matrix[i["a"], i["c"]] - ((1 + 0.5 + 3) + 2) / 2) < 1e-9
    assert abs(matrix[i["a"], i["b"]] - 3.0) < 1e-9
    # A missing edge length counts as one unit.
    bare = Network.from_newick("((a,b),c);")
    assert _leaf_path_lengths(bare, {"a", "c"}) == {("a", "c"): 3.0}
