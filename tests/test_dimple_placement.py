"""Reticulation placement: which edge, which gamma, and when it is kept."""

from phynetpy.Network import Edge, Network, Node

from phynetpy._dimple import REDUNDANCY_THRESHOLD, place_reticulations


def _hybrid(child="b", major="a", minor="c", major_gamma=0.9, minor_gamma=0.1, name="#H1"):
    nodes = {label: Node(label) for label in (child, major, minor)}
    hybrid = Node(name, is_reticulation=True)
    major_parent, minor_parent, root = Node("M" + name), Node("N" + name), Node("R" + name)
    net = Network()
    net.add_nodes(list(nodes.values()) + [hybrid, major_parent, minor_parent, root])
    major_edge, minor_edge = Edge(major_parent, hybrid), Edge(minor_parent, hybrid)
    major_edge.set_gamma(major_gamma)
    minor_edge.set_gamma(minor_gamma)
    net.add_edges([
        Edge(root, major_parent), Edge(root, minor_parent),
        Edge(major_parent, nodes[major]), major_edge,
        Edge(hybrid, nodes[child]),
        Edge(minor_parent, nodes[minor]), minor_edge,
    ])
    return net


def _incoming(net):
    """(gamma, sibling leaves) for every edge into a reticulation."""
    rows = []
    for node in net.V():
        if net.in_degree(node) < 2:
            continue
        for edge in net.in_edges(node):
            siblings = set()
            for child in net.get_children(edge.src):
                if child is node:
                    continue
                siblings.update(leaf.label for leaf in net.leaf_descendants(child))
            rows.append((edge.get_gamma(), frozenset(siblings)))
    return rows


def _accept_if_reticulated(base_count):
    def score(net):
        count = sum(1 for node in net.V() if net.in_degree(node) >= 2)
        return 1.0 if count > base_count else 0.0
    return score


def test_exact_match_is_used_and_a_missing_side_is_skipped():
    source = _hybrid()
    backbone = Network.from_newick("((a,b),c);")
    placed, trials = place_reticulations(backbone, [source], _accept_if_reticulated(0))
    assert trials[0].accepted
    assert trials[0].edge_leaves == frozenset({"c"})
    assert any(gamma == 0.1 and siblings == frozenset({"c"}) for gamma, siblings in _incoming(placed))
    assert any(gamma == 0.9 and siblings == frozenset({"a"}) for gamma, siblings in _incoming(placed))

    # No taxon of the source's minor side exists on this backbone.
    nowhere = Network.from_newick("((a,b),d);")
    before = nowhere.newick()
    skipped, skip_trials = place_reticulations(nowhere, [source], _accept_if_reticulated(0))
    assert skipped.newick() == before
    assert skip_trials[0].accepted is False
    assert skip_trials[0].edge_leaves == frozenset()


def test_gammas_follow_the_side_the_backbone_kept():
    source = _hybrid()
    standard, _ = place_reticulations(
        Network.from_newick("((a,b),c);"), [source], _accept_if_reticulated(0),
    )
    flipped, _ = place_reticulations(
        Network.from_newick("((b,c),a);"), [source], _accept_if_reticulated(0),
    )
    # Standard: the new edge carries the minor gamma 0.1, on the {c} side.
    assert (0.1, frozenset({"c"})) in _incoming(standard)
    assert (0.9, frozenset({"a"})) in _incoming(standard)
    # Flipped: the backbone kept {c}, so the new edge is the {a} side at 0.9.
    assert (0.9, frozenset({"a"})) in _incoming(flipped)
    assert (0.1, frozenset({"c"})) in _incoming(flipped)


def test_a_rejected_trial_does_not_change_the_network_and_a_nonfinite_score_raises():
    source = _hybrid()
    backbone = Network.from_newick("((a,b),c);")
    snapshot = sorted(
        (edge.src.label, edge.dest.label, edge.get_length(), edge.get_gamma())
        for edge in backbone.E()
    )
    returned, trials = place_reticulations(backbone, [source], lambda net: 0.0)
    assert returned is backbone
    assert trials[0].accepted is False
    assert snapshot == sorted(
        (edge.src.label, edge.dest.label, edge.get_length(), edge.get_gamma())
        for edge in returned.E()
    )
    try:
        place_reticulations(Network.from_newick("((a,b),c);"), [source], lambda net: float("inf"))
    except RuntimeError:
        return
    raise AssertionError("a non-finite base score was treated as 'add nothing'")


def test_the_kept_network_is_the_one_that_was_scored():
    # The backbone already has a reticulation, so the child has two parents.
    # The recorded edge is the one the winning trial scored.
    backbone = Network.from_newick(
        "((a:1,(b:1)#H1[&gamma=0.7]:1):1,(#H1:1[&gamma=0.3],c:1):1);"
    )
    source = _hybrid(major_gamma=0.8, minor_gamma=0.2, name="#H9")
    base = sum(1 for node in backbone.V() if backbone.in_degree(node) >= 2)

    def score(net):
        return 200.0 if sum(1 for node in net.V() if net.in_degree(node) >= 2) > base else 100.0

    placed, trials = place_reticulations(backbone, [source], score)
    assert trials[0].accepted
    assert trials[0].score_before == 100.0
    assert trials[0].score_after == 200.0
    assert trials[0].edge_leaves in {siblings for _, siblings in _incoming(placed)}
    assert placed is not backbone


def test_a_small_gain_near_an_existing_reticulation_is_rejected():
    assert REDUNDANCY_THRESHOLD == 0.02
    backbone = Network.from_newick(
        "((a:1,(b:1)#H1[&gamma=0.7]:1):1,(#H1:1[&gamma=0.3],c:1):1);"
    )
    source = _hybrid(major_gamma=0.8, minor_gamma=0.2, name="#H9")
    base = sum(1 for node in backbone.V() if backbone.in_degree(node) >= 2)

    def small(net):
        extra = sum(1 for node in net.V() if net.in_degree(node) >= 2) > base
        return 101.0 if extra else 100.0

    _, near = place_reticulations(backbone, [source], small)
    assert near[0].accepted is False
    assert near[0].score_after == 101.0

    def large(net):
        extra = sum(1 for node in net.V() if net.in_degree(node) >= 2) > base
        return 200.0 if extra else 100.0

    _, cleared = place_reticulations(backbone, [source], large)
    assert cleared[0].accepted is True

    # The same small gain on a tree, far from any reticulation, is enough.
    _, far = place_reticulations(
        Network.from_newick("((a,b),c);"),
        [source],
        lambda net: 101.0 if any(net.in_degree(node) >= 2 for node in net.V()) else 100.0,
    )
    assert far[0].accepted is True


def test_an_accepted_reticulation_is_found_on_the_copied_network():
    # The node created by an accepted insertion belongs to that graph. The
    # next candidate is scored on a copy, so the redundancy check has to
    # find the reticulation again from its child leaves.
    from phynetpy._dimple import _edge_distance, _near_accepted

    placed, trials = place_reticulations(
        Network.from_newick("((a,b),c);"), [_hybrid()], _accept_if_reticulated(0),
    )
    assert trials[0].accepted
    copied, _ = placed.copy()
    stale = next(node for node in placed.V() if placed.in_degree(node) >= 2)
    anchor = copied.has_node_named("a")
    assert _edge_distance(copied, anchor, stale) > 2
    assert _near_accepted(copied, anchor, frozenset({"b"}), [frozenset({"b"})])


def _retic_over_ab():
    """Network on {a,b,c,d} whose reticulation sits above {a,b}."""
    nodes = {label: Node(label) for label in "abcd"}
    ab = Node("AB")
    hybrid = Node("#H", is_reticulation=True)
    major, minor, root = Node("M"), Node("N"), Node("R")
    net = Network()
    net.add_nodes(list(nodes.values()) + [ab, hybrid, major, minor, root])
    major_edge, minor_edge = Edge(major, hybrid), Edge(minor, hybrid)
    major_edge.set_gamma(0.8)
    minor_edge.set_gamma(0.2)
    net.add_edges([
        Edge(root, major), Edge(root, minor),
        Edge(major, nodes["c"]), major_edge,
        Edge(hybrid, ab), Edge(ab, nodes["a"]), Edge(ab, nodes["b"]),
        Edge(minor, nodes["d"]), minor_edge,
    ])
    return net


def test_retic_skipped_when_child_not_clade():
    source = _retic_over_ab()
    # {a,b} is not a clade: their MRCA also contains c.
    skipped, trials = place_reticulations(
        Network.from_newick("((a,(b,c)),d);"),
        [source],
        lambda net: 1.0,
    )
    assert trials[0].accepted is False
    assert trials[0].reason == "child_not_clade"
    assert sum(1 for node in skipped.V() if skipped.in_degree(node) >= 2) == 0

    tried, tried_trials = place_reticulations(
        Network.from_newick("(((a,b),c),d);"),
        [source],
        lambda net: 0.0,
    )
    assert tried_trials[0].reason == ""
    assert tried_trials[0].accepted is False


def test_an_event_whose_child_sits_in_the_next_side_is_placed_first():
    first = _hybrid("b", "a", "c", name="#HA")
    second = _hybrid("d", "e", "b", major_gamma=0.7, minor_gamma=0.3, name="#HB")
    backbone = Network.from_newick("((((a,b),c),(d,e)));")

    def score(net):
        return float(sum(1 for node in net.V() if net.in_degree(node) >= 2))

    # second is listed first. Dependency order still places b before d,
    # and the second placement is scored on the tree the first one changed.
    _, trials = place_reticulations(backbone, [second, first], score)
    assert [trial.child_leaves for trial in trials] == [
        frozenset({"b"}), frozenset({"d"}),
    ]
    assert all(trial.accepted for trial in trials)


def test_min_gain_rejects_a_gain_that_dimples_rule_accepts():
    from phynetpy._dimple import selection_min_gain
    source = _hybrid(major_gamma=0.8, minor_gamma=0.2, name="#H9")
    tree = Network.from_newick("((a,b),c);")

    def gain_of_two(net):
        return 102.0 if any(net.in_degree(node) >= 2 for node in net.V()) else 100.0

    _, dimple_rule = place_reticulations(tree, [source], gain_of_two)
    assert dimple_rule[0].accepted is True
    # AIC charges 3 parameters per reticulation; a gain of 2 does not pay.
    assert selection_min_gain("aic", 1000) == 3.0
    _, charged = place_reticulations(tree, [source], gain_of_two, min_gain=selection_min_gain("aic", 1000))
    assert charged[0].accepted is False
    assert selection_min_gain("elbow", 1000) == 0.0 and selection_min_gain(None, 1000) == 0.0
    assert abs(selection_min_gain("bic", 1000) - 1.5 * __import__("math").log(1000)) < 1e-9


def test_the_merger_scorer_agrees_with_the_python_triplet_scorer():
    from phynetpy.data import GeneTrees
    from phynetpy._dimple import _msc_scorer, restrict_gene_trees
    from phynetpy._mcmc_gt import _populate_default_branch_lengths
    from phynetpy.infer import compute_gene_tree_triplets, score_species_network_triplets
    genes = GeneTrees(trees=[
        Network.from_newick(newick) for newick in
        ["((a,b),(c,d));"] * 6 + ["((a,c),(b,d));"] * 3 + ["((a,d),(b,c));"] * 1
    ])
    taxa = frozenset("abcd")
    scorer = _msc_scorer(genes, taxa)
    table = compute_gene_tree_triplets(genes, genes.resolved_mapping(), species_labels=sorted(taxa))
    for newick in ("((a,b),(c,d));", "((a:1,(b:1)#H1[&gamma=0.7]:1):1,(#H1:1[&gamma=0.3],(c:1,d:1):1):1);"):
        net = Network.from_newick(newick)
        _populate_default_branch_lengths(net)
        expected = score_species_network_triplets(net, table).log_pseudo_likelihood
        assert abs(scorer(Network.from_newick(newick)) - expected) < 1e-6
    # Trees already on exactly the wanted taxa are shared, not copied.
    kept = restrict_gene_trees(genes, sorted(taxa))
    originals = {id(tree) for tree in genes.trees}
    assert all(id(tree) in originals for tree in kept.trees)


def _major_tree(net):
    from phynetpy.GraphUtils import get_all_subtrees
    return get_all_subtrees(net)[0]


def _same_network(truth, merged):
    from phynetpy.GraphUtils import hardwired_cluster_distance
    from phynetpy.ReticulationComparison import compare_networks
    comparison = compare_networks(truth, merged)
    return (
        hardwired_cluster_distance(truth, merged, normalize=True) == 0
        and comparison.precision == 1.0 and comparison.recall == 1.0
    )


def _count(net):
    return float(sum(1 for node in net.V() if net.in_degree(node) >= 2))


def test_nested_reticulations_are_rebuilt_on_the_major_tree():
    # Inner #H2's minor parent is inside outer #H1's child clade. On the
    # network #H1's child is {b, c}, which the major tree does not show.
    cross = Network.from_newick(
        "((a:1,((b:1,(c:1)#H2[&gamma=0.4]:1):1)#H1[&gamma=0.6]:1):1,(#H1:1,(#H2:1,d:1):1):1);"
    )
    merged, trials = place_reticulations(_major_tree(cross), [cross], _count)
    assert all(trial.accepted for trial in trials) and len(trials) == 2
    assert _same_network(cross, merged)


def test_a_chain_of_reticulations_is_rebuilt_upper_one_first():
    chain = Network.from_newick(
        "((((a:1)#H1[&gamma=0.6]:1)#H2[&gamma=0.6]:1,((#H1:1,b:1):1,(#H2:1,c:1):1):1):1,e:1);"
    )
    merged, trials = place_reticulations(_major_tree(chain), [chain], _count)
    # The upper reticulation (minor side {c}) is inserted first, so the
    # lower one lands in the edge below it.
    assert [sorted(trial.edge_leaves) for trial in trials] == [["c"], ["b"]]
    assert _same_network(chain, merged)


def test_an_inserted_hybrid_edge_does_not_leak_leaves_into_a_later_clade():
    from phynetpy._dimple import _placement_view
    cross = Network.from_newick(
        "((a:1,((b:1,(c:1)#H2[&gamma=0.4]:1):1)#H1[&gamma=0.6]:1):1,(#H1:1,(#H2:1,d:1):1):1);"
    )
    merged, _ = place_reticulations(_major_tree(cross), [cross], _count)
    inserted = {
        (edge.src.label, edge.dest.label)
        for node in merged.V() if merged.in_degree(node) >= 2
        for edge in merged.in_edges(node) if edge.src.label.startswith("UID")
    }
    view = _placement_view(merged, inserted)
    assert sum(1 for node in view.V() if view.in_degree(node) >= 2) == 0
    assert {leaf.label for leaf in view.get_leaves()} == set("abcd")
    assert _placement_view(merged, set()) is merged
