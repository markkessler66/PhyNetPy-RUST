"""Tiny networks and the structural check shared by the DIMPLE tests."""

from phynetpy.Network import Network


def leaves(net: Network) -> list[str]:
    """Sorted leaf labels of ``net``."""
    return sorted(leaf.label for leaf in net.get_leaves())


def assert_phylogeny(net: Network, expected: list[str]) -> None:
    """One root, no cycle, the expected leaves once, binary reticulations.

    A degree-2 chain (in-degree 1 and out-degree 1) is a failed cleanup.
    """
    found = leaves(net)
    assert found == sorted(expected)
    assert len(found) == len(set(found))
    assert net.is_acyclic()
    roots = [
        node for node in net.V()
        if net.in_degree(node) == 0 and net.out_degree(node) > 0
    ]
    assert len(roots) == 1
    for node in net.V():
        indegree = net.in_degree(node)
        outdegree = net.out_degree(node)
        if node.is_reticulation() or indegree >= 2:
            assert indegree == 2 and outdegree == 1
        assert not (indegree == 1 and outdegree == 1)
