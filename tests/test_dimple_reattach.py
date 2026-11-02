"""Putting pruned clades back onto a blob."""

import pytest

from phynetpy.Network import Edge, Network, Node

from phynetpy._dimple import DimpleError, PrunedSubtree, reattach_prunes
from dimple_fixtures import assert_phylogeny, leaves


def _leaf_net(newick: str) -> Network:
    return Network.from_newick(newick)


def test_a_prune_is_a_sibling_of_the_cut_sibling_mrca():
    blob = Network.from_newick("((a,b),(c,d));")
    prune = PrunedSubtree(
        leaves=frozenset("ef"),
        anchor_leaves=frozenset("ab"),
        parent_leaves=frozenset("abef"),
        subtree=_leaf_net("(e,f);"),
        source_type="blob_item",
    )
    result = reattach_prunes(blob, [prune])
    assert_phylogeny(result, list("abcdef"))
    anchor = result.mrca({"a", "b"})
    grafted = result.mrca({"e", "f"})
    assert result.get_parents(anchor)[0] is result.get_parents(grafted)[0]


def test_attachment_at_a_reticulation_lands_inside_the_hybrid_clade():
    blob = Network.from_newick(
        "((a:1,(b:1)#H1[&gamma=0.7]:1):1,(#H1:1[&gamma=0.3],c:1):1);"
    )
    prune = PrunedSubtree(
        leaves=frozenset({"e"}),
        anchor_leaves=frozenset({"b"}),
        parent_leaves=frozenset({"b", "e"}),
        subtree=_leaf_net("(e);"),
        source_type="blob_item",
    )
    result = reattach_prunes(blob, [prune])
    hybrid = next(node for node in result.V() if node.is_reticulation())
    assert result.has_node_named("e") in result.get_subtree_at(hybrid)
    assert_phylogeny(result, list("abce"))


def test_stacked_reticulations_are_walked_through():
    # H1 and H2 both have two parents, so a degree-2 cleanup cannot delete
    # the lower one. The walk from H1 stops at b, under H2.
    from phynetpy._dimple import _below_retics

    h1 = Node("#H1", is_reticulation=True)
    h2 = Node("#H2", is_reticulation=True)
    root, left, right = Node("Root"), Node("L"), Node("R")
    a, b, c = Node("a"), Node("b"), Node("c")
    blob = Network()
    blob.add_nodes([h1, h2, root, left, right, a, b, c])
    blob.add_edges([
        Edge(root, left), Edge(root, right),
        Edge(left, a), Edge(left, h1), Edge(right, c), Edge(right, h1),
        Edge(h1, h2), Edge(right, h2), Edge(h2, b),
    ])
    for edge in list(blob.in_edges(h1)) + list(blob.in_edges(h2)):
        edge.set_gamma(0.5)
    assert _below_retics(blob, h1) is b
    result = reattach_prunes(blob, [PrunedSubtree(
        leaves=frozenset({"e"}),
        anchor_leaves=frozenset({"b"}),
        parent_leaves=frozenset({"b", "e"}),
        subtree=_leaf_net("(e);"),
        source_type="blob_item",
    )])
    lower = next(node for node in result.V() if node.label == "#H2")
    assert result.has_node_named("e") in result.get_subtree_at(lower)


def test_a_root_anchor_keeps_the_leaves():
    # An anchor covering the whole blob is its root; a new root is made.
    blob = Network.from_newick("((a,b),(c,d));")
    at_root = reattach_prunes(blob, [PrunedSubtree(
        leaves=frozenset({"e"}),
        anchor_leaves=frozenset("abcd"),
        parent_leaves=frozenset("abcde"),
        subtree=_leaf_net("(e);"),
        source_type="blob_item",
    )])
    assert_phylogeny(at_root, list("abcde"))


def test_a_leaf_in_both_the_blob_and_the_prune_is_not_duplicated():
    blob = Network.from_newick(
        "((a:1,(b:1)#H1[&gamma=0.9]:1):1,(#H1:1[&gamma=0.1],c:1):1);"
    )
    # b is already in the blob. The prune brings b back, so the copy in the
    # blob is removed first and the inheritance probabilities stay.
    result = reattach_prunes(blob, [PrunedSubtree(
        leaves=frozenset({"b"}),
        anchor_leaves=frozenset({"a"}),
        parent_leaves=frozenset({"a", "b"}),
        subtree=_leaf_net("(b);"),
        source_type="blob_item",
    )])
    assert leaves(result).count("b") == 1
    gammas = sorted(
        edge.get_gamma()
        for node in result.V() if result.in_degree(node) >= 2
        for edge in result.in_edges(node)
    )
    assert gammas == [0.1, 0.9]


def test_a_prune_waits_for_the_prune_it_anchors_on():
    blob = Network.from_newick("((a,b),(c,d));")
    outer = PrunedSubtree(
        leaves=frozenset("ef"),
        anchor_leaves=frozenset("ab"),
        parent_leaves=frozenset("ab"),
        subtree=_leaf_net("(e,f);"),
        source_type="blob_item",
        index=0,
    )
    inner = PrunedSubtree(
        leaves=frozenset("gh"),
        anchor_leaves=frozenset("ef"),
        parent_leaves=frozenset("efgh"),
        subtree=_leaf_net("(g,h);"),
        source_type="prune",
        index=1,
    )
    result = reattach_prunes(blob, [inner, outer])
    assert_phylogeny(result, list("abcdefgh"))
    assert result.get_parents(result.mrca({"g", "h"}))[0] is result.get_parents(
        result.mrca({"e", "f"})
    )[0]


def test_an_empty_anchor_or_a_disjoint_blob_anchor_raises():
    blob = Network.from_newick("((a,b),(c,d));")
    with pytest.raises(DimpleError, match="empty anchor"):
        reattach_prunes(blob, [PrunedSubtree(
            leaves=frozenset({"e"}),
            anchor_leaves=frozenset(),
            parent_leaves=frozenset(),
            subtree=_leaf_net("(e);"),
        )])
    with pytest.raises(DimpleError, match="no leaf"):
        reattach_prunes(blob, [PrunedSubtree(
            leaves=frozenset({"e"}),
            anchor_leaves=frozenset({"z"}),
            parent_leaves=frozenset({"z"}),
            subtree=_leaf_net("(e);"),
            source_type="blob_item",
        )])


def test_the_closer_cut_is_the_inner_sibling():
    # Scaffold: ((a,b),(c,d)). The cut of {e} has its parent at {a,b}, close
    # to anchor {a,b}. The cut of {f} has its parent at the root, farther.
    scaffold = Network.from_newick("((a,b),(c,d));")
    close = PrunedSubtree(
        leaves=frozenset({"e"}),
        anchor_leaves=frozenset("ab"),
        parent_leaves=frozenset("ab"),
        subtree=_leaf_net("(e);"),
        source_type="blob_item",
        index=0,
    )
    far = PrunedSubtree(
        leaves=frozenset({"f"}),
        anchor_leaves=frozenset("ab"),
        parent_leaves=frozenset("abcd"),
        subtree=_leaf_net("(f);"),
        source_type="blob_item",
        index=1,
    )
    result = reattach_prunes(Network.from_newick("((a,b),(c,d));"), [far, close], scaffold)
    assert_phylogeny(result, list("abcdef"))
    anchor = result.mrca({"a", "b"})
    assert result.get_parents(result.has_node_named("e"))[0] is result.get_parents(anchor)[0]
    assert result.get_parents(result.has_node_named("f"))[0] is not result.get_parents(anchor)[0]


def test_a_stuck_prune_is_hosted_where_most_of_its_anchor_is():
    # DIMPLE's host search. {g} anchors on {e, z}; z never exists, so no
    # ordinary pass can place it. Once {e,f} is on the blob, e is the host.
    blob = Network.from_newick("((a,b),(c,d));")
    first = PrunedSubtree(
        leaves=frozenset("ef"),
        anchor_leaves=frozenset("ab"),
        parent_leaves=frozenset("abef"),
        subtree=_leaf_net("(e,f);"),
        source_type="blob_item",
        index=0,
    )
    stuck = PrunedSubtree(
        leaves=frozenset({"g"}),
        anchor_leaves=frozenset({"e", "z"}),
        parent_leaves=frozenset({"e", "g", "z"}),
        subtree=_leaf_net("(g);"),
        source_type="prune",
        index=1,
    )
    result = reattach_prunes(blob, [stuck, first])
    assert_phylogeny(result, list("abcdefg"))
    assert result.get_parents(result.has_node_named("g"))[0] is result.get_parents(
        result.has_node_named("e")
    )[0]


def test_a_stuck_prune_can_ride_inside_another_waiting_prune():
    # Neither prune's anchor is complete. {g} shares two leaves with the
    # {e,f} clade and one with nothing else, so it is attached inside that
    # clade first; the clade then lands beside a, the one anchor leaf it has.
    blob = Network.from_newick("((a,b),(c,d));")
    carrier = PrunedSubtree(
        leaves=frozenset("ef"),
        anchor_leaves=frozenset({"a", "z"}),
        parent_leaves=frozenset("aefz"),
        subtree=_leaf_net("(e,f);"),
        source_type="prune",
        index=0,
    )
    rider = PrunedSubtree(
        leaves=frozenset({"g"}),
        anchor_leaves=frozenset({"e", "f", "z"}),
        parent_leaves=frozenset("efgz"),
        subtree=_leaf_net("(g);"),
        source_type="prune",
        index=1,
    )
    result = reattach_prunes(blob, [carrier, rider])
    assert_phylogeny(result, list("abcdefg"))
    pair = result.mrca({"e", "f"})
    assert result.get_parents(result.has_node_named("g"))[0] is result.get_parents(pair)[0]
    trio = result.mrca({"e", "f", "g"})
    assert result.get_parents(trio)[0] is result.get_parents(result.has_node_named("a"))[0]
    # The caller's record is not mutated by the ride-along.
    assert set(leaves(carrier.subtree)) == set("ef")


def test_a_prune_with_no_host_at_all_still_raises():
    blob = Network.from_newick("((a,b),(c,d));")
    with pytest.raises(DimpleError, match="could not be attached"):
        reattach_prunes(blob, [PrunedSubtree(
            leaves=frozenset({"e"}),
            anchor_leaves=frozenset({"z"}),
            parent_leaves=frozenset({"z"}),
            subtree=_leaf_net("(e);"),
            source_type="prune",
        )])
