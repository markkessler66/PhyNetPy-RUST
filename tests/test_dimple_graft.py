"""Grafting merged blobs back onto the tree of blobs."""

import pytest

from phynetpy.Network import Edge, Network, Node

from phynetpy._dimple import DimpleError, GraftPiece, graft_blobs, strip_outgroup
from dimple_fixtures import assert_phylogeny, leaves


def test_the_scaffold_node_with_the_blob_leaves_is_replaced():
    scaffold = Network.from_newick("(a,b,c,d);")
    result = graft_blobs(scaffold, [GraftPiece(
        frozenset("abcd"),
        Network.from_newick("((a,b),(c,d));"),
        frozenset(),
    )])
    assert_phylogeny(result, list("abcd"))
    assert result.mrca({"a", "b"}) is not result.mrca({"a", "b", "c", "d"})


def test_an_inner_blob_survives_the_outer_replacement():
    scaffold = Network.from_newick("(a,b,(q,(c,d,e)));")
    result = graft_blobs(scaffold, [
        GraftPiece(frozenset("cde"), Network.from_newick("((c,d),e);"), frozenset({"q"})),
        GraftPiece(
            frozenset({"a", "b", "q", "c", "d", "e"}),
            Network.from_newick("((a,b),q);"),
            frozenset(),
        ),
    ])
    assert_phylogeny(result, list("abcdeq"))
    clade = result.mrca({"c", "d", "e"})
    assert result.get_parents(clade)[0] is result.get_parents(result.has_node_named("q"))[0]


def test_a_colliding_internal_name_is_renamed_and_leaves_are_not():
    z, a, b, c = Node("z"), Node("a"), Node("b"), Node("c")
    blob, root = Node("Blob"), Node("Shared")
    scaffold = Network()
    scaffold.add_nodes([z, a, b, c, blob, root])
    scaffold.add_edges([
        Edge(root, z), Edge(root, blob), Edge(blob, a), Edge(blob, b), Edge(blob, c),
    ])
    piece = Network.from_newick("((a,b),c);")
    internal = next(node for node in piece.V() if piece.out_degree(node) == 2)
    piece.update_node_name(internal, "Shared")
    result = graft_blobs(scaffold, [GraftPiece(frozenset("abc"), piece, frozenset())])
    assert sum(1 for node in result.V() if node.label == "Shared") == 1
    assert leaves(result) == list("abcz")
    assert_phylogeny(result, list("abcz"))


def test_a_root_blob_becomes_the_network_and_a_repeated_leaf_raises():
    scaffold = Network.from_newick("(a,b,c,d);")
    result = graft_blobs(scaffold, [GraftPiece(
        frozenset("abcd"), Network.from_newick("((a,b),(c,d));"), frozenset(),
    )])
    assert_phylogeny(result, list("abcd"))

    with pytest.raises(DimpleError, match="duplicate"):
        # z stays on the scaffold when the {a,b,c} clade is replaced, and the
        # piece names z again.
        graft_blobs(Network.from_newick("((a,b,c),z);"), [GraftPiece(
            frozenset("abc"),
            Network.from_newick("((a,b),(c,z));"),
            frozenset(),
        )])


def test_stripping_an_outgroup_leaves_the_remaining_scaffold_leaves():
    scaffold = Network.from_newick("(OUT,(a,b,c,d));")
    stripped = strip_outgroup(scaffold, "OUT")
    result = graft_blobs(stripped, [GraftPiece(
        frozenset("abcd"),
        Network.from_newick("((a,b),(c,d));"),
        frozenset(),
    )])
    assert_phylogeny(result, list("abcd"))
    assert "OUT" not in leaves(result)


def test_nested_inner_blobs_re_hang_most_available_anchor_first():
    # Two polytomies nested under one ordinary node of the root polytomy.
    # A's external siblings are f and B's leaves; B's are A's leaves and f.
    # When the outer blob is grafted only f is present, so A goes first,
    # beside f, and B then anchors on the whole (A, f) clade. DIMPLE's
    # merge_to_tob orders the re-hang by available siblings the same way.
    scaffold = Network.from_newick("((((A1,A2,A3),f),(B1,B2,B3)),(c,d),e);")
    everything = frozenset("A1 A2 A3 f B1 B2 B3 c d e".split())
    outer = GraftPiece(everything, Network.from_newick("((f,c),(d,e));"), frozenset())
    a = GraftPiece(frozenset({"A1", "A2", "A3"}), Network.from_newick("((A1,A2),A3);"),
                   frozenset({"f", "B1", "B2", "B3"}))
    b = GraftPiece(frozenset({"B1", "B2", "B3"}), Network.from_newick("((B1,B2),B3);"),
                   frozenset({"A1", "A2", "A3", "f"}))
    result = graft_blobs(scaffold, [outer, a, b])
    assert_phylogeny(result, sorted(everything))
    a_root = result.mrca({"A1", "A2", "A3"})
    assert result.get_parents(a_root)[0] is result.get_parents(result.has_node_named("f"))[0]
    b_root = result.mrca({"B1", "B2", "B3"})
    af = result.mrca({"A1", "A2", "A3", "f"})
    assert result.get_parents(b_root)[0] is result.get_parents(af)[0]


def test_an_inner_blob_with_no_anchor_leaf_is_kept_beside_the_graft_root():
    # Each nested blob's siblings are only the other nested blob's leaves,
    # so neither anchor exists on the outer network. DIMPLE falls back to
    # a sibling of the graft root's first child rather than dropping taxa.
    scaffold = Network.from_newick("(((A1,A2,A3),(B1,B2,B3)),(c,d),e);")
    everything = frozenset("A1 A2 A3 B1 B2 B3 c d e".split())
    outer = GraftPiece(everything, Network.from_newick("((c,d),e);"), frozenset())
    a = GraftPiece(frozenset({"A1", "A2", "A3"}), Network.from_newick("((A1,A2),A3);"),
                   frozenset({"B1", "B2", "B3"}))
    b = GraftPiece(frozenset({"B1", "B2", "B3"}), Network.from_newick("((B1,B2),B3);"),
                   frozenset({"A1", "A2", "A3"}))
    result = graft_blobs(scaffold, [outer, a, b])
    assert_phylogeny(result, sorted(everything))
