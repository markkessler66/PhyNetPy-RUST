"""Network transactions: journaled mutation with exact rollback."""

import numpy as np
import pytest

from phynetpy.Network import Edge, Network, Node
from phynetpy.ModelGraph import Model
from phynetpy import ModelMove as mm
from phynetpy._mcmc_gt import _populate_default_branch_lengths


def _fingerprint(net: Network):
    def time_of(node):
        try:
            return node.get_time()
        except Exception:
            return None
    nodes = tuple(sorted((n.label, n.is_reticulation(), time_of(n)) for n in net.V()))
    edges = tuple(sorted((e.src.label, e.dest.label, e.get_length(), e.get_gamma()) for e in net.E()))
    return nodes, edges, tuple(sorted(l.label for l in net.get_leaves())), net.root().label


def _ids(net: Network):
    return {id(n) for n in net.V()} | {id(e) for e in net.E()}


def test_rollback_restores_topology_attributes_and_the_same_objects():
    net = Network.from_newick("((a:1,(b:1)#H1[&gamma=0.7]:1):1,(#H1:1[&gamma=0.3],c:1):1);")
    before, ids = _fingerprint(net), _ids(net)
    net.begin_transaction()
    hybrid = next(e for e in net.E() if e.dest.is_reticulation())
    hybrid.set_gamma(0.1)
    hybrid.set_length(5.0)
    net.has_node_named("a").set_time(3.0)
    net.remove_nodes(net.has_node_named("b"))
    net.update_node_name(net.has_node_named("c"), "zz")
    net.clean()
    assert _fingerprint(net) != before
    net.rollback_transaction()
    assert _fingerprint(net) == before
    assert _ids(net) == ids
    assert not net.in_transaction()


def test_an_unset_time_is_unset_again_after_rollback():
    net = Network()
    a, b = Node("a"), Node("b")
    net.add_nodes(a, b)
    net.add_edges(Edge(a, b))
    with pytest.raises(Exception):
        b.get_time()
    net.begin_transaction()
    b.set_time(2.0)
    assert b.get_time() == 2.0
    net.rollback_transaction()
    with pytest.raises(Exception):
        b.get_time()


def test_commit_keeps_changes_and_nested_transactions_unwind_separately():
    net = Network.from_newick("((a,b),(c,d));")
    net.begin_transaction()
    extra = Node("e")
    net.add_nodes(extra)
    net.add_edges(Edge(net.root(), extra))
    net.begin_transaction()
    net.remove_nodes(net.has_node_named("a"))
    assert net.has_node_named("a") is None
    net.rollback_transaction()                 # inner only
    assert net.has_node_named("a") is not None and net.has_node_named("e") is not None
    assert net.in_transaction()
    net.commit_transaction()
    assert not net.in_transaction()
    assert net.has_node_named("e") is not None
    # Idle calls are no-ops.
    net.rollback_transaction()
    net.commit_transaction()
    assert net.has_node_named("e") is not None


def test_changes_outside_a_transaction_are_not_journaled():
    net = Network.from_newick("((a,b),(c,d));")
    net.remove_nodes(net.has_node_named("a"))
    net.begin_transaction()
    net.rollback_transaction()
    assert net.has_node_named("a") is None


def test_copies_do_not_inherit_ownership_and_free_elements_are_silent():
    net = Network.from_newick("((a:1,b:1):1,c:1);")
    twin = net.copy()[0]
    assert all(n._owner is twin for n in twin.V()) and all(e._owner is twin for e in twin.E())
    assert all(n._owner is net for n in net.V())
    loose = Edge(Node("x"), Node("y"))
    loose.set_gamma(0.4)                       # no owner, no error
    net.begin_transaction()
    net.remove_edge(next(iter(net.E())))
    removed = [e for e in twin.E()]            # the twin is untouched
    assert len(removed) == len(list(net.E())) + 1
    net.rollback_transaction()


@pytest.mark.parametrize("move_cls", [
    mm.AddReticulation, mm.RemoveReticulation, mm.FlipReticulation, mm.SwitchParentage,
    mm.SPR, mm.ChangeReticSource, mm.RelocateReticulation, mm.ChangeReticDest,
])
def test_every_topology_move_rolls_back_exactly(move_cls):
    tree = Network.from_newick("((((a:1,b:1):1,c:1):1,(d:1,e:1):1):1,(((f:1,g:1):1,h:1):1,(i:1,j:1):1):1);")
    reticulate = Network.from_newick(
        "((((a:1,(b:1)#H1[&gamma=0.6]:1):1,(#H1:1,c:1):1):1,(d:1,e:1):1):1,"
        "(((f:1,g:1):1,h:1):1,(i:1,j:1):1):1);"
    )
    for base in (tree, reticulate):
        _populate_default_branch_lengths(base)
        for seed in range(12):
            net = base.copy()[0]
            model = Model(rng=np.random.default_rng(seed))
            model.network = net
            before, ids = _fingerprint(net), _ids(net)
            move = move_cls()
            try:
                move.execute(model)
            finally:
                move.undo(model)
            assert _fingerprint(model.network) == before
            assert _ids(model.network) == ids
            assert not model.network.in_transaction()
