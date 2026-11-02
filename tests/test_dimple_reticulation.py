"""Reticulation identity and displayed-tree selection.

Displayed-tree selection counts rooted triples. It does not call the MSC
pseudo-likelihood.
"""

from phynetpy.GraphUtils import (
    get_all_subtrees,
    group_reticulations,
    rooted_triples,
    same_reticulation,
)
from phynetpy.Network import Edge, Network, Node

from phynetpy._dimple import (
    collect_reticulation_candidates,
    reticulation_records,
    select_displayed_trees,
)

HYBRID = "((a:1,(b:1)#H1[&gamma=0.9]:1):1,(#H1:1[&gamma=0.1],c:1):1);"


def _hybrid(child, major, minor, major_gamma=0.9, minor_gamma=0.1, name="#H"):
    nodes = {label: Node(label) for label in (child, major, minor)}
    hybrid = Node(name, is_reticulation=True)
    major_parent = Node("M" + name)
    minor_parent = Node("N" + name)
    root = Node("R" + name)
    net = Network()
    net.add_nodes(list(nodes.values()) + [hybrid, major_parent, minor_parent, root])
    major_edge = Edge(major_parent, hybrid)
    minor_edge = Edge(minor_parent, hybrid)
    major_edge.set_gamma(major_gamma)
    minor_edge.set_gamma(minor_gamma)
    net.add_edges([
        Edge(root, major_parent), Edge(root, minor_parent),
        Edge(major_parent, nodes[major]), major_edge,
        Edge(hybrid, nodes[child]),
        Edge(minor_parent, nodes[minor]), minor_edge,
    ])
    return net


def test_parent_order_does_not_split_an_event_and_one_shared_leaf_groups_them():
    left = _hybrid("b", "x", "c", name="#H1")
    # Same child and same two sides, parents created in the other order.
    right = _hybrid("b", "c", "x", major_gamma=0.2, minor_gamma=0.8, name="#H2")
    left_sig = reticulation_records(left)[0]["signature"]
    right_sig = reticulation_records(right)[0]["signature"]
    assert same_reticulation(left_sig, right_sig)
    assert len(group_reticulations([
        {"signature": left_sig}, {"signature": right_sig},
    ])) == 1

    overlap = _hybrid("b", "y", "c", name="#H3")
    # Replace the {x}-side with {y,z} sharing only y would need a different
    # builder. A one-leaf overlap against {x,y} vs {y}:
    wide = _hybrid("b", "x", "c", name="#H4")
    # Manually compare signatures whose parent branches overlap in one leaf.
    wide_sig = reticulation_records(wide)[0]["signature"]
    narrow = (
        wide_sig[0],
        (wide_sig[1][0], (frozenset({"y"}),)),
    )
    # wide's minor side is {c}; narrow's second side is {y}. Those are disjoint,
    # so this pair must NOT group. The overlapping pair shares y.
    overlapping = (
        wide_sig[0],
        ((frozenset({"x", "y"}),), (frozenset({"c"}),)),
    )
    assert same_reticulation(wide_sig, overlapping)
    assert not same_reticulation(wide_sig, narrow)
    assert not same_reticulation(wide_sig, (
        wide_sig[0],
        ((frozenset({"p"}),), (frozenset({"q"}),)),
    ))


def test_the_child_clade_is_stored_and_is_not_what_equality_checks():
    record = reticulation_records(_hybrid("b", "a", "c"))[0]
    assert record["signature"][0] == (frozenset({"b"}),)
    other = (
        (frozenset({"z"}),),
        record["signature"][1],
    )
    assert same_reticulation(record["signature"], other)
    assert other[0] != record["signature"][0]


def test_in_degree_other_than_two_is_not_a_record():
    assert reticulation_records(Network.from_newick("((a,b),c);")) == []
    hybrid = Node("H", is_reticulation=True)
    parents = [Node(f"P{i}") for i in range(3)]
    root, leaf = Node("Root"), Node("leaf")
    net = Network()
    net.add_nodes([hybrid, root, leaf, *parents])
    net.add_edges([Edge(root, parent) for parent in parents])
    net.add_edges([Edge(parent, hybrid) for parent in parents])
    net.add_edges(Edge(hybrid, leaf))
    assert reticulation_records(net) == []


def test_literal_duplicates_are_kept_once_and_gamma_variants_collapse_to_the_first():
    two = (
        "(((a:1,(b:1)#H1[&gamma=0.7]:1):1,(#H1:1[&gamma=0.3],c:1):1):1,"
        "((d:1,(e:1)#H2[&gamma=0.6]:1):1,(#H2:1[&gamma=0.4],f:1):1):1);"
    )
    got = collect_reticulation_candidates([
        Network.from_newick(two), Network.from_newick(two),
    ])
    assert sorted(tuple(record["retic_leaves"]) for record in got) == [("b",), ("e",)]
    low = Network.from_newick(
        "((a:1,(b:1)#H1[&gamma=0.7]:1):1,(#H1:1[&gamma=0.3],c:1):1);"
    )
    high = Network.from_newick(
        "((a:1,(b:1)#H1[&gamma=0.6]:1):1,(#H1:1[&gamma=0.4],c:1):1);"
    )
    # DIMPLE keeps one representative per event (collect_unique_retics), so
    # a second gamma on the same event is not a separate candidate.
    variants = collect_reticulation_candidates([low, high])
    assert len(variants) == 1
    assert variants[0]["major_gamma"] == 0.7


def test_the_same_event_seen_by_overlapping_subsets_is_one_candidate_with_the_largest_child():
    # Both subnetworks put a hybrid between the {a} side and the {c} side.
    # The second also carries d inside the hybrid clade. Parent-side overlap
    # groups them; the larger child clade is the representative.
    narrow = Network.from_newick(
        "((a:1,(b:1)#H1[&gamma=0.7]:1):1,(#H1:1[&gamma=0.3],c:1):1);"
    )
    wide = Network.from_newick(
        "((a:1,(b:1,d:1)#H1[&gamma=0.6]:1):1,(#H1:1[&gamma=0.4],c:1):1);"
    )
    got = collect_reticulation_candidates([narrow, wide])
    assert len(got) == 1
    assert got[0]["retic_leaves"] == frozenset({"b", "d"})
    assert got[0]["network_taxa"] == frozenset("abcd")


def test_a_tree_is_returned_unchanged_and_one_displayed_tree_skips_counts():
    tree = Network.from_newick("((a,b),c);")
    assert select_displayed_trees([tree], None)[0] is tree

    # Flag a node but give it one parent, so there is a single displayed tree.
    root, marked, leaf, other = Node("Root"), Node("M", is_reticulation=True), Node("a"), Node("b")
    net = Network()
    net.add_nodes([root, marked, leaf, other])
    net.add_edges([Edge(root, marked), Edge(root, other), Edge(marked, leaf)])
    chosen = select_displayed_trees([net], None)
    assert len(chosen) == 1


def test_the_picker_keeps_the_tree_that_misses_fewer_triples_and_not_msc():
    network = Network.from_newick(HYBRID)
    other = Network.from_newick("((a,b),c);")
    chosen = select_displayed_trees([network, other], {("a", "b", "c"): 5})[0]
    assert ("a", "b", "c") in rooted_triples(chosen)
    assert ("b", "c", "a") not in rooted_triples(chosen)

    flipped = select_displayed_trees([network, other], {("b", "c", "a"): 5})[0]
    assert ("b", "c", "a") in rooted_triples(flipped)


def test_fewer_than_three_shared_taxa_is_a_total_miss_and_ties_keep_the_first():
    network = Network.from_newick(HYBRID)
    distant = Network.from_newick("((a,b),x);")
    first = get_all_subtrees(network)[0]
    chosen = select_displayed_trees(
        [network, distant], {("a", "b", "c"): 5},
    )[0]
    # Sibling order in Newick is not part of the tree. The triple set is.
    assert rooted_triples(chosen) == rooted_triples(first)

    tied = select_displayed_trees(
        [network, Network.from_newick("((a,b),c);")], {},
    )[0]
    assert rooted_triples(tied) == rooted_triples(first)


# Nested reticulations: the leaf sets are read on a displayed tree.

# Inner #H2's minor parent sits inside outer #H1's child clade; its major
# parent is outside. On the network, #H1's leaves include c through #H2's
# minor edge, and {b, c} is not a clade of any displayed tree.
CROSS = "((a:1,((b:1,(c:1)#H2[&gamma=0.4]:1):1)#H1[&gamma=0.6]:1):1,(#H1:1,(#H2:1,d:1):1):1);"
# #H2 -> #H1 -> a: a reticulation whose only child is a reticulation.
CHAIN = "((((a:1)#H1[&gamma=0.6]:1)#H2[&gamma=0.6]:1,((#H1:1,b:1):1,(#H2:1,c:1):1):1):1,e:1);"


def _by_child(records):
    return {tuple(sorted(record["retic_leaves"])): record for record in records}


def test_nested_records_read_clades_of_the_major_tree():
    records = _by_child(reticulation_records(Network.from_newick(CROSS)))
    outer, inner = records[("b",)], records[("c",)]
    # The network's leaf set under #H1 is {b, c}; the major tree's is {b}.
    assert outer["kept_sibling_leaves"] == frozenset({"a"})
    # #H1's minor edge attaches above the clade {c, d} once #H2 hangs from
    # its major parent, not above {d} alone and not inside its own child.
    assert outer["minor_sibling_leaves"] == frozenset({"c", "d"})
    assert inner["kept_sibling_leaves"] == frozenset({"d"})
    assert inner["minor_sibling_leaves"] == frozenset({"b"})
    assert outer["retic_depth"] == inner["retic_depth"] == 0


def test_a_chain_record_has_a_side_and_a_depth():
    records = reticulation_records(Network.from_newick(CHAIN))
    assert [tuple(sorted(r["retic_leaves"])) for r in records] == [("a",), ("a",)]
    lower = next(r for r in records if r["minor_sibling_leaves"] == frozenset({"b"}))
    upper = next(r for r in records if r["minor_sibling_leaves"] == frozenset({"c"}))
    # #H1's major parent #H2 has no other child: on the network that side
    # is empty. In the tree it attaches where #H2's parent stands.
    assert lower["kept_sibling_leaves"] == frozenset({"a", "b", "c"})
    assert upper["kept_sibling_leaves"] == frozenset({"a", "b", "c"})
    assert (upper["retic_depth"], lower["retic_depth"]) == (1, 0)


def test_level_one_records_are_the_network_leaf_sets():
    record = reticulation_records(_hybrid("b", "a", "c"))[0]
    assert record["retic_leaves"] == frozenset({"b"})
    assert record["kept_sibling_leaves"] == frozenset({"a"})
    assert record["minor_sibling_leaves"] == frozenset({"c"})
    assert record["retic_depth"] == 0
    # The LCA of the cycle is a parent: that side holds the child too, as before.
    lca = reticulation_records(Network.from_newick(
        "(((a:1,(b:1)#H1[&gamma=0.3]:1):1,#H1:1):1,c:1);"
    ))[0]
    assert lca["retic_leaves"] == frozenset({"b"})
    assert lca["kept_sibling_leaves"] == frozenset({"a", "b"})
    assert lca["minor_sibling_leaves"] == frozenset({"a"})


def test_a_displayed_tree_choice_names_the_kept_edges():
    from phynetpy._dimple import select_displayed_trees_with_choices
    network = Network.from_newick(HYBRID)
    other = Network.from_newick("((a,b),c);")
    trees, choices = select_displayed_trees_with_choices([network, other], {("a", "b", "c"): 5})
    (hybrid,) = [node for node in network.V() if network.in_degree(node) == 2]
    kept = choices[0][hybrid]
    assert ("a", "b", "c") in rooted_triples(trees[0])
    # The kept edge is the one whose side the chosen tree displays: b beside a.
    assert {leaf.label for leaf in network.leaf_descendants(kept.src)} == {"a", "b"}
    assert choices[1] is None
    _, flipped = select_displayed_trees_with_choices([network, other], {("b", "c", "a"): 5})
    assert flipped[0][hybrid] is not kept
    assert {leaf.label for leaf in network.leaf_descendants(flipped[0][hybrid].src)} == {"b", "c"}
    assert select_displayed_trees_with_choices([Network.from_newick("((a,b),c);")])[1] == [None]
