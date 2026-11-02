"""Parity report: distances decide, score gaps are recorded."""

import os
from pathlib import Path

import pytest

from phynetpy.Network import Network

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import dimple_parity_report as parity


def test_identical_networks_are_identical():
    tree = Network.from_newick("((a,b),(c,d));")
    other = Network.from_newick("((a,b),(c,d));")
    label, distances, comparison = parity.classify_pair(tree, other)
    assert label == "identical"
    assert all(value == 0 for value in distances.values())
    assert comparison.D_ret_hat == 0


def test_one_extra_reticulation_is_a_different_reticulation():
    tree = Network.from_newick("((a,b),(c,d));")
    network = Network.from_newick(
        "((a:1,(b:1)#H1[&gamma=0.7]:1):1,(#H1:1[&gamma=0.3],(c:1,d:1):1):1);"
    )
    # The network is the reference, so a tree that dropped its reticulation
    # has recall below 1.
    label, _, comparison = parity.classify_pair(network, tree)
    assert label == "different reticulations"
    assert comparison.D_ret_hat > 0
    assert comparison.recall < 1


def test_renamed_internals_are_still_identical():
    left = Network.from_newick("((a,b),(c,d));")
    right = Network.from_newick("((a,b),(c,d));")
    internal = next(node for node in right.V() if right.out_degree(node) == 2)
    right.update_node_name(internal, "Renamed")
    label, distances, comparison = parity.classify_pair(left, right)
    assert label == "identical"
    assert comparison.D_ret_hat == 0
    assert all(value == 0 for value in distances.values())


def test_unequal_scores_are_reported_and_do_not_fail():
    text = parity.score_note(1.0, 2.5, -3.0)
    assert "1.0" in text and "2.5" in text and "-3.0" in text
    assert "search gap" in text
    absent = parity.score_note(1.0, 1.0, None)
    assert "absent" in absent


def test_a_directory_without_metadata_still_parses_the_final_network(tmp_path):
    (tmp_path / "gene_trees.tre").write_text("((a,b),(c,d));\n")
    (tmp_path / "tob_rooted.tre").write_text("(a,b,c,d);\n")
    (tmp_path / "astral.tre").write_text("((a,b),(c,d));\n")
    (tmp_path / "dimple_network.nwk").write_text("((a,b),(c,d));\n")
    loaded = parity.load_dimple_directory(tmp_path)
    assert loaded["final"] is not None
    assert {leaf.label for leaf in loaded["final"].get_leaves()} == set("abcd")
    assert loaded["blobs"] == []


@pytest.mark.skipif(
    not os.environ.get("DIMPLE_PARITY_DIR"),
    reason="Set DIMPLE_PARITY_DIR to a DIMPLE output directory to compare runs.",
)
def test_live_directory_matches_leaf_sets(tmp_path):
    directory = Path(os.environ["DIMPLE_PARITY_DIR"])
    text, summary = parity.run_parity(directory, num_iter=1)
    assert summary["leaf_sets_match"]
    out = tmp_path / "dimple_parity.md"
    out.write_text(text)
    assert "Final network:" in out.read_text()
