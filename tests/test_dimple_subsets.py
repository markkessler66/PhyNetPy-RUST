"""Restricting gene trees and the start tree, and the subset infer call."""

from unittest.mock import patch

import pytest

from phynetpy.Network import Network
from phynetpy.data import GeneTrees
from phynetpy.infer import InferenceResult

from phynetpy._dimple import DimpleError, restrict_gene_trees, restrict_start


def _chain(labels):
    newick = labels[0]
    for label in labels[1:]:
        newick = f"({newick},{label})"
    return Network.from_newick(newick + ";")


def test_a_gene_tree_missing_taxa_is_cut_to_the_overlap():
    full = Network.from_newick("((a,b),(c,d));")
    partial = Network.from_newick("((a,b),c);")
    genes = GeneTrees(trees=[full, partial], mapping={lab: [lab] for lab in "abcd"})
    restricted = restrict_gene_trees(genes, ["a", "b", "d"])
    # The partial tree has only a and b in the subset, so it is omitted.
    assert len(restricted.trees) == 1
    kept = next(iter(restricted.trees))
    assert {leaf.label for leaf in kept.get_leaves()} == {"a", "b", "d"}
    assert "c" not in restricted.resolved_mapping()
    assert restricted.resolved_mapping()["a"] == ["a"]


def test_every_gene_tree_missing_the_subset_raises():
    genes = GeneTrees(trees=[Network.from_newick("((a,b),c);")])
    with pytest.raises(DimpleError, match="No gene tree"):
        restrict_gene_trees(genes, ["d", "e", "f"])


def test_the_restricted_start_is_binary_on_exactly_the_subset():
    start = Network.from_newick("(((a,b),c),d);")
    restricted = restrict_start(start, ["a", "b", "c"])
    assert {leaf.label for leaf in restricted.get_leaves()} == {"a", "b", "c"}
    for node in restricted.V():
        if restricted.out_degree(node):
            assert restricted.out_degree(node) == 2
    with pytest.raises(DimpleError, match="not binary"):
        restrict_start(Network.from_newick("(a,b,c,d);"), ["a", "b", "c", "d"])
    with pytest.raises(DimpleError, match="missing"):
        restrict_start(start, ["a", "b", "z"])


def test_each_subset_is_three_annealing_runs():
    scaffold = Network.from_newick("(a,b,c,d,e,f);")
    start = _chain(list("abcdef"))
    genes = GeneTrees(trees=[start])
    seen = []

    def fake_infer(data, **kwargs):
        seen.append(kwargs)
        labels = sorted({leaf.label for tree in data.trees for leaf in tree.get_leaves()})
        return InferenceResult(best=_chain(labels), score=-1.0, method="stub")

    with patch("phynetpy.infer.infer", fake_infer):
        from phynetpy.infer import infer_dimple
        result = infer_dimple(
            genes, scaffold, start,
            max_subset_size=4, n_divisions=1, max_reticulations=2,
            seed=0, num_iter=1,
        )
    # Two subsets, three annealing runs each. num_iter=1 overrides the
    # 5000-iteration default; the method and the seeds do not.
    assert len(seen) == 6
    assert {call["max_reticulations"] for call in seen} == {2}
    assert {call["method"] for call in seen} == {"sa"}
    assert {call["num_iter"] for call in seen} == {1}
    assert {call["schedule"] for call in seen} == {"geometric_reheat"}
    assert sorted(call["seed"] for call in seen) == [0, 0, 1, 1, 2, 2]
    assert result.method == "DIMPLE"
    assert len(result.raw.blobs[0].subsets) == 2


# ── Reticulation count per subset: ground truth and model selection ──


def _subset_genes(n=100):
    return GeneTrees(trees=[Network.from_newick("((a,b),c);") for _ in range(n)])


def _stub(score_by_k):
    """A fake infer/score pair: k reticulations cost nothing to find."""
    seen = []

    def fake_infer(data, **kwargs):
        k = kwargs["max_reticulations"]
        seen.append(kwargs)
        if k == 0:
            net = Network.from_newick("((a,b),c);")
        else:
            net = Network.from_newick(
                "((a:1,(b:1)#H1[&gamma=0.7]:1):1,(#H1:1[&gamma=0.3],c:1):1);"
            )
        return InferenceResult(best=net, score=score_by_k[k], method="stub")

    def fake_score(net, data, **kwargs):
        return score_by_k[sum(1 for node in net.V() if node.is_reticulation())]

    return fake_infer, fake_score, seen


def test_the_true_bound_counts_only_reticulations_triplets_can_see():
    from phynetpy._dimple import true_reticulation_bound
    truth = Network.from_newick("((((A,(B)#H1),(#H1,C)),E),D);")
    assert true_reticulation_bound(truth, ["A", "B", "C"]) == 1
    # Restricted to B, C, D the cycle collapses to three nodes: a tree to triplets.
    assert true_reticulation_bound(truth, ["B", "C", "D"]) == 0
    assert true_reticulation_bound(truth, ["A", "C", "D"]) == 0
    with pytest.raises(DimpleError, match="lacks subset taxa"):
        true_reticulation_bound(truth, ["A", "B", "Z"])


def test_aic_and_bic_keep_the_tree_when_the_gain_is_below_the_penalty():
    from phynetpy._dimple import _infer_subset
    start = _chain(list("abc"))
    fake_infer, fake_score, seen = _stub({0: -1000.0, 1: -998.0})
    with patch("phynetpy.infer.infer", fake_infer), patch("phynetpy.infer.score", fake_score):
        found = _infer_subset(_subset_genes(), start, frozenset("abc"), 1,
                              {"n_runs": 2, "num_iter": 1}, model_selection="aic")
    assert found.k_selected == 0 and found.selection == "aic" and found.bound == 1
    assert found.score == -1000.0
    assert sorted((c["max_reticulations"], c["seed"]) for c in seen) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert [row["k"] for row in found.sweep] == [0, 1]
    assert [row["n_retics"] for row in found.sweep] == [0, 1]
    with patch("phynetpy.infer.infer", fake_infer), patch("phynetpy.infer.score", fake_score):
        found = _infer_subset(_subset_genes(), start, frozenset("abc"), 1,
                              {"n_runs": 1, "num_iter": 1}, model_selection="bic")
    assert found.k_selected == 0


def test_a_large_gain_selects_the_reticulation_and_the_elbow_tries_one_extra_cap():
    from phynetpy._dimple import _infer_subset
    start = _chain(list("abc"))
    fake_infer, fake_score, seen = _stub({0: -1000.0, 1: -950.0})
    with patch("phynetpy.infer.infer", fake_infer), patch("phynetpy.infer.score", fake_score):
        found = _infer_subset(_subset_genes(), start, frozenset("abc"), 1,
                              {"n_runs": 1, "num_iter": 1}, model_selection="aic")
    assert found.k_selected == 1
    assert sum(1 for node in found.network.V() if node.is_reticulation()) == 1

    # The elbow sweeps 0, 1, 2 for a cap of 1; [50, 1] bends at 1.
    fake_infer, fake_score, seen = _stub({0: -1000.0, 1: -950.0, 2: -949.0})
    with patch("phynetpy.infer.infer", fake_infer), patch("phynetpy.infer.score", fake_score):
        found = _infer_subset(_subset_genes(), start, frozenset("abc"), 1,
                              {"n_runs": 1, "num_iter": 1}, model_selection="elbow")
    assert sorted(c["max_reticulations"] for c in seen) == [0, 1, 2]
    assert found.k_selected == 1
    # A curve that keeps rising picks the extra cap, which is clamped to the bound.
    fake_infer, fake_score, seen = _stub({0: -1000.0, 1: -950.0, 2: -905.0})
    with patch("phynetpy.infer.infer", fake_infer), patch("phynetpy.infer.score", fake_score):
        found = _infer_subset(_subset_genes(), start, frozenset("abc"), 1,
                              {"n_runs": 1, "num_iter": 1}, model_selection="elbow")
    assert found.k_selected == 1


def test_without_a_rule_the_cap_is_used_as_before_and_an_unknown_rule_raises():
    from phynetpy._dimple import _infer_subset
    start = _chain(list("abc"))
    fake_infer, fake_score, seen = _stub({0: -1000.0, 1: -998.0})
    with patch("phynetpy.infer.infer", fake_infer), patch("phynetpy.infer.score", fake_score):
        found = _infer_subset(_subset_genes(), start, frozenset("abc"), 1, {"n_runs": 3, "num_iter": 1})
    assert found.k_selected == 1 and found.selection is None and found.sweep is None
    assert [c["max_reticulations"] for c in seen] == [1, 1, 1]
    with pytest.raises(DimpleError, match="model_selection"):
        _infer_subset(_subset_genes(), start, frozenset("abc"), 1, {}, model_selection="aicc")


def test_a_true_network_sets_each_subsets_cap_to_its_detectable_count():
    scaffold = Network.from_newick("(a,b,c,d,e,f);")
    start = _chain(list("abcdef"))
    genes = GeneTrees(trees=[start])
    # One reticulation among a, b, c; none elsewhere.
    truth = Network.from_newick("((((a,(b)#H1),(#H1,c)),(d,e)),f);")
    seen = []

    def fake_infer(data, **kwargs):
        labels = sorted({leaf.label for tree in data.trees for leaf in tree.get_leaves()})
        seen.append((frozenset(labels), kwargs["max_reticulations"]))
        return InferenceResult(best=_chain(labels), score=-1.0, method="stub")

    with patch("phynetpy.infer.infer", fake_infer):
        from phynetpy.infer import infer_dimple
        from phynetpy._dimple import true_reticulation_bound
        result = infer_dimple(
            genes, scaffold, start, max_subset_size=4, n_divisions=1,
            max_reticulations=1, seed=0, num_iter=1, true_network=truth,
        )
    assert seen
    for taxa, cap in seen:
        assert cap == true_reticulation_bound(truth, sorted(taxa))
    for record in result.raw.blobs[0].subsets:
        assert record.bound == true_reticulation_bound(truth, sorted(record.taxa))
        assert record.k_selected == record.bound and record.selection is None


def test_gain_rule_uses_the_threshold_and_warm_starts_each_cap_from_the_one_below():
    from phynetpy._dimple import SELECTION_GAIN_THRESHOLD, _infer_subset, selection_min_gain
    from phynetpy.infer import Start
    assert SELECTION_GAIN_THRESHOLD == 50.0
    assert selection_min_gain("gain", 1000) == 50.0
    start = _chain(list("abc"))
    # Gain 40 is below the threshold: the tree is kept even though AIC would take it.
    fake_infer, fake_score, seen = _stub({0: -1000.0, 1: -960.0})
    with patch("phynetpy.infer.infer", fake_infer), patch("phynetpy.infer.score", fake_score):
        found = _infer_subset(_subset_genes(), start, frozenset("abc"), 1,
                              {"n_runs": 2, "num_iter": 1}, model_selection="gain")
    assert found.k_selected == 0 and found.selection == "gain"
    # Cap-1 runs start from the cap-0 result (a tree), not from the start tree object.
    cap1 = [c for c in seen if c["max_reticulations"] == 1]
    assert cap1 and all(isinstance(c["start"], Start) for c in cap1)
    assert all(not any(n.is_reticulation() for n in c["start"].network.V()) for c in cap1)
    cap0 = [c for c in seen if c["max_reticulations"] == 0]
    # Every cap-0 run starts from the one restricted start tree; cap-1 runs do not.
    restricted = {id(c["start"].network) for c in cap0}
    assert len(restricted) == 1
    assert all(id(c["start"].network) not in restricted for c in cap1)
    # Gain 60 clears it.
    fake_infer, fake_score, seen = _stub({0: -1000.0, 1: -940.0})
    with patch("phynetpy.infer.infer", fake_infer), patch("phynetpy.infer.score", fake_score):
        found = _infer_subset(_subset_genes(), start, frozenset("abc"), 1,
                              {"n_runs": 1, "num_iter": 1}, model_selection="gain")
    assert found.k_selected == 1
    # The fixed-cap path still starts every run from the start tree (DIMPLE's protocol).
    fake_infer, fake_score, seen = _stub({0: -1000.0, 1: -940.0})
    with patch("phynetpy.infer.infer", fake_infer), patch("phynetpy.infer.score", fake_score):
        _infer_subset(_subset_genes(), start, frozenset("abc"), 1, {"n_runs": 2, "num_iter": 1})
    assert len({id(c["start"].network) for c in seen}) == 1


# ── Chain portfolios ─────────────────────────────────────────────────


def test_subset_chains_expand_flat_settings_and_portfolios():
    from phynetpy._dimple import SUBSET_ITERS, SUBSET_RUNS, subset_chains

    flat = subset_chains({"seed": 7, "num_iter": 10})
    assert [c["seed"] for c in flat] == [7 + i for i in range(SUBSET_RUNS)]
    assert {c["method"] for c in flat} == {"sa"}
    assert {c["num_iter"] for c in flat} == {10}
    assert all("schedule" in c and "phase_patience" not in c for c in flat)

    portfolio = subset_chains({
        "seed": 3,
        "chains": [
            {"method": "sa", "num_iter": 20, "n_runs": 2},
            {"method": "hc", "num_iter": 30, "n_runs": 1},
        ],
    })
    assert [(c["method"], c["num_iter"], c["seed"], c["chain"]) for c in portfolio] == [
        ("sa", 20, 3, 0), ("sa", 20, 4, 1), ("hc", 30, 5, 2),
    ]
    assert portfolio[2]["phase_patience"] > 0 and "schedule" not in portfolio[2]
    assert subset_chains({"chains": [{"n_runs": 1}]})[0]["num_iter"] == SUBSET_ITERS
    with pytest.raises(DimpleError):
        subset_chains({"chains": []})


def test_a_portfolio_keeps_the_best_scoring_chain_and_records_all_of_them():
    from phynetpy._dimple import _infer_subset

    start = Network.from_newick("((a,b),c);")
    seen = []

    def fake_infer(data, **kwargs):
        seen.append(kwargs)
        # The hill climb scores best; annealing seeds score by their seed.
        value = -900.0 if kwargs["method"] == "hc" else -1000.0 + kwargs["seed"]
        return InferenceResult(best=Network.from_newick("((a,b),c);"), score=value, method="stub")

    def fake_score(net, data, **kwargs):
        return -1.0

    with patch("phynetpy.infer.infer", fake_infer), patch("phynetpy.infer.score", fake_score):
        found = _infer_subset(_subset_genes(), start, frozenset("abc"), 1, {
            "num_iter": 1,
            "chains": [{"method": "sa", "n_runs": 2}, {"method": "hc", "n_runs": 1}],
        })
    assert [c["method"] for c in seen] == ["sa", "sa", "hc"]
    assert "chain" not in seen[0] and "phase_patience" not in seen[0]
    assert found.score == -1.0  # re-scored, not the search statistic
    assert [c["chosen"] for c in found.chains] == [False, False, True]
    assert [c["score"] for c in found.chains] == [-1000.0, -999.0, -900.0]
