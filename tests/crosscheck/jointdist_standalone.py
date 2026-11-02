"""jointdist, single-file build: a joint reticulation-and-tree dissimilarity for phylogenetic networks.

Generated from the jointdist package by tools/build_standalone.py; do not edit by hand.
Usage:  python jointdist_standalone.py truth.nwk estimate.nwk [--csv]
Python: import jointdist_standalone as J; J.score(J.parse(nwk1), J.parse(nwk2))
Requires numpy, scipy and phynetpy==0.6.0.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, OrderedDict
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment



# ============================================================================
# network.py
# ============================================================================
"""Validation of input networks and the light rooted-DAG type the measure works on.

A phylogenetic network is read with PhyNetPy and validated once; every block and
remainder is then a ``Graph``: a rooted DAG with labelled leaves, detached from
PhyNetPy, with small-integer node ids. Two leaves of a Graph may share a label
(two markers of reticulations with the same descendant taxa), which is what makes
the structural instantiations well defined on multi-labelled trees.
"""



# --------------------------------------------------------------- PhyNetPy side
def topological_order(net) -> list:
    """Kahn's topological sort of all nodes; raises ValueError on a cycle."""
    nodes = list(net.V())
    indeg = {v: 0 for v in nodes}
    for v in nodes:
        for c in net.get_children(v):
            indeg[c] = indeg.get(c, 0) + 1
    ready = [v for v in nodes if indeg[v] == 0]
    order = []
    while ready:
        v = ready.pop()
        order.append(v)
        for c in net.get_children(v):
            indeg[c] -= 1
            if indeg[c] == 0:
                ready.append(c)
    if len(order) != len(nodes):
        raise ValueError(f"network is not a DAG: {len(nodes) - len(order)} nodes lie on a cycle")
    return order


def leaves(net) -> list:
    """The leaf nodes: every node without children (an isolated root included)."""
    return [v for v in net.V() if net.out_degree(v) == 0]


def validate(net) -> None:
    """Refuse a network the measure is not defined on. Raises ValueError.

    Checked in order: nonempty with exactly one root; acyclic; every leaf labelled
    and no label used twice; every reticulation has exactly two distinct parents
    and exactly one child.
    """
    nodes = list(net.V())
    if not nodes:
        raise ValueError("empty network")
    topological_order(net)                                   # raises on a cycle
    roots = [v for v in nodes if net.in_degree(v) == 0]
    if len(roots) != 1:
        raise ValueError(f"network has {len(roots)} roots; a rooted network has exactly one")
    labels = [n.label for n in leaves(net)]
    if any(l is None or l == "" for l in labels):
        raise ValueError("network has an unlabelled leaf")
    dup = sorted(l for l, c in Counter(labels).items() if c > 1)
    if dup:
        raise ValueError(f"duplicate leaf labels: {dup}")
    for h in nodes:
        if net.in_degree(h) < 2:
            continue
        name = h.label if h.label else "(unlabelled)"
        parents = net.get_parents(h)
        if len(parents) != 2 or parents[0] is parents[1]:
            raise ValueError(f"reticulation {name} has {len(parents)} parent edges; "
                             f"the measure is defined for binary reticulations with two distinct parents")
        if net.out_degree(h) == 0:
            raise ValueError(f"reticulation {name} is a leaf (two parents, no child); "
                             f"usually a duplicated taxon label merged by the parser")
        if net.out_degree(h) != 1:
            raise ValueError(f"reticulation {name} has {net.out_degree(h)} children; "
                             f"the measure is defined for one child per reticulation")


def hybrids(net) -> list:
    """The reticulation nodes, in one fixed order."""
    return [n for n in net.V() if net.in_degree(n) >= 2]


def taxa(net) -> frozenset:
    """The taxon set X(N): the leaf labels."""
    return frozenset(v.label for v in leaves(net))


# ------------------------------------------------------------------------ Graph
class Graph:
    """A rooted DAG with labelled leaves.

    ``children[v]`` is a tuple of child ids without repeats, ``label[v]`` the label
    of leaf ``v``. ``root`` is None only for the empty graph.
    """
    __slots__ = ("root", "children", "label", "_leaves", "_order", "_clusters")

    def __init__(self, root, children: dict, label: dict):
        self.root, self.children, self.label = root, children, label
        self._leaves = self._order = self._clusters = None

    def __len__(self):
        return len(self.children)

    @property
    def leaves(self) -> frozenset:
        """The label set L(S) (distinct labels)."""
        if self._leaves is None:
            self._leaves = frozenset(self.label.values())
        return self._leaves

    def order(self) -> list:
        """Nodes root first, every node before its children."""
        if self._order is None:
            if self.root is None:
                self._order = []
            else:
                indeg = Counter(c for cs in self.children.values() for c in cs)
                out, ready = [], [self.root]
                while ready:
                    v = ready.pop()
                    out.append(v)
                    for c in self.children[v]:
                        indeg[c] -= 1
                        if indeg[c] == 0:
                            ready.append(c)
                if len(out) != len(self.children):
                    raise ValueError("graph is not a DAG, or has nodes unreachable from the root")
                self._order = out
        return self._order

    def cluster_multiset(self) -> Counter:
        """Hardwired clusters of the FULL structure: one record per vertex.

        A cluster is the multiset of labels of the leaf vertices reachable from
        the vertex (a leaf reached along two paths counts once; two leaves with
        the same label count twice). Records are keyed by the immutable set of
        (label, multiplicity) pairs, so labels need not be sortable.
        """
        if self._clusters is None:
            reach, out = {}, Counter()
            for v in reversed(self.order()):
                if v in self.label:
                    leaves = frozenset([v])
                else:
                    leaves = frozenset().union(*(reach[c] for c in self.children[v]))
                reach[v] = leaves
                out[frozenset(Counter(self.label[leaf] for leaf in leaves).items())] += 1
            self._clusters = out
        return self._clusters

    def node_classes(self, interner: dict) -> Counter:
        """Nakhleh's vertex-equivalence classes of the FULL structure.

        Two vertices are equivalent when they are leaves with the same label, or
        when their children can be matched one-to-one into equivalent vertices.
        Returns a Counter of class ids; classes are interned in ``interner`` so two
        graphs keyed through the SAME dict share class ids.
        """
        key, out = {}, Counter()
        for v in reversed(self.order()):
            if v in self.label:
                sig = ("leaf", self.label[v])
            else:
                sig = ("node", tuple(sorted(key[c] for c in self.children[v])))
            k = interner.setdefault(sig, len(interner))
            key[v] = k
            out[k] += 1
        return out


def _dedupe(cs):
    out = []
    for c in cs:
        if c not in out:
            out.append(c)
    return tuple(out)


EMPTY = Graph(None, {}, {})


def network_graph(net):
    """The validated network as a Graph, and the node-to-id map."""
    order = topological_order(net)
    id_of = {v: i for i, v in enumerate(order)}
    children = {id_of[v]: _dedupe(tuple(id_of[c] for c in net.get_children(v))) for v in order}
    label = {id_of[v]: v.label for v in order if not net.get_children(v)}
    return Graph(id_of[order[0]], children, label), id_of


def subgraph(G: Graph, root, exclude=frozenset()) -> Graph:
    """Everything reachable from ``root`` without entering a node of ``exclude``."""
    if root in exclude:
        return EMPTY
    seen, stack = {root}, [root]
    while stack:
        v = stack.pop()
        for c in G.children[v]:
            if c not in exclude and c not in seen:
                seen.add(c)
                stack.append(c)
    children = {v: tuple(c for c in G.children[v] if c in seen) for v in seen}
    label = {v: G.label[v] for v in seen if v in G.label}
    return Graph(root, children, label)


def parent_block(G: Graph, root, exclude=frozenset()) -> Graph:
    """The vertices on a path from ``root`` to a taxon leaf once ``exclude`` is deleted.

    Nothing is suppressed: a parent left unary by the deletion stays. Empty when
    no path from the parent to a taxon leaf remains.
    """
    S = subgraph(G, root, exclude)
    if S.root is None:
        return EMPTY
    keep = set()
    for v in reversed(S.order()):
        if v in S.label or any(c in keep for c in S.children[v]):
            keep.add(v)
    if S.root not in keep:
        return EMPTY
    children = {v: tuple(c for c in S.children[v] if c in keep) for v in keep}
    label = {v: S.label[v] for v in keep if v in S.label}
    return Graph(S.root, children, label)


# ============================================================================
# instantiations.py
# ============================================================================
"""The three instantiations of the structure dissimilarity d(S, S') in [0, 1].

Each takes two ``Graph``s (tripartition blocks, or the two marked remainders) and
compares their COMPLETE structures: no restriction to shared leaves, no weighting.
Both empty -> 0; exactly one empty -> 1; isomorphic -> 0.

  leaf     d = Jaccard distance of the label sets, |L xor L'| / |L u L'|
  cluster  d = HC(S, S') = |C xor C'| / |C u C'| over the multisets of hardwired
           clusters of the full structures, one cluster per vertex
  nakhleh  d = L / U over Nakhleh's vertex-equivalence classes of the full
           structures, L = sum_q |c_S(q) - c_S'(q)| and U = sum_q max(c_S(q), c_S'(q))
           (equivalently 4m / (|V| + |V'| + 2m) with m = L / 2 Nakhleh's measure)
"""



def jaccard(P: frozenset, Q: frozenset) -> float:
    return len(P ^ Q) / len(P | Q) if (P or Q) else 0.0


def d_leaf(S: Graph, T: Graph) -> float:
    return jaccard(S.leaves, T.leaves)


def d_cluster(S: Graph, T: Graph) -> float:
    C1, C2 = S.cluster_multiset(), T.cluster_multiset()
    sym = union = 0
    for c in C1.keys() | C2.keys():
        a, b = C1[c], C2[c]
        sym += abs(a - b)
        union += max(a, b)
    return sym / union if union else 0.0


def nakhleh_counts(S: Graph, T: Graph) -> tuple[int, int]:
    """(L, U): summed class-count difference and summed class-count maximum."""
    interner = {}
    K1, K2 = S.node_classes(interner), T.node_classes(interner)
    L = U = 0
    for k in K1.keys() | K2.keys():
        a, b = K1[k], K2[k]
        L += abs(a - b)
        U += max(a, b)
    return L, U


def d_nakhleh(S: Graph, T: Graph) -> float:
    L, U = nakhleh_counts(S, T)
    return L / U if U else 0.0


KINDS = ("leaf", "cluster", "nakhleh")
DSET = {"leaf": d_leaf, "cluster": d_cluster, "nakhleh": d_nakhleh}
SUFFIX = {"leaf": "l", "cluster": "c", "nakhleh": "n"}


def dset_of(kind: str):
    if kind not in DSET:
        raise ValueError(f"unknown instantiation {kind!r}; choose from {KINDS}")
    return DSET[kind]


# ============================================================================
# tripartition.py
# ============================================================================
"""Tripartitions and the reticulation term.

Every reticulation h of N gives one tripartition (A, B, C): B is the block below
h (h kept as its root); A and C are the blocks below h's two parents once h is
deleted, pruned to the vertices on a path from that parent to a taxon leaf.
Nothing is suppressed, and identical tripartitions stay separate entries.

Two tripartitions are compared block by block with the parents unordered,

    delta(t, t') = d(B, B') + min(d(A, A') + d(C, C'), d(A, C') + d(C, A'))   in [0, 3].

The smaller collection is padded with dummy tripartitions to m = max(r1, r2)
(real-dummy cost 3, dummy-dummy cost 0) and

    D_ret = min over bijections M of  sum_t delta(t, M(t))

by one Hungarian solve on the m x m matrix.
"""



Triple = tuple[Graph, Graph, Graph]
DUMMY_COST = 3.0


def tripartitions_of(net, G: Graph, id_of: dict) -> list[Triple]:
    """One (A, B, C) per reticulation of a VALIDATED network given as (G, id_of) from ``network_graph``."""
    out = []
    for h in hybrids(net):
        hid = id_of[h]
        u, v = (id_of[p] for p in net.get_parents(h))
        out.append((parent_block(G, u, exclude={hid}), subgraph(G, hid), parent_block(G, v, exclude={hid})))
    return out


def tripartitions(net) -> list[Triple]:
    """One (A, B, C) per reticulation, in the order of ``network.hybrids``. Validates first."""
    validate(net)
    return tripartitions_of(net, *network_graph(net))


def cost(t1: Triple, t2: Triple, dset) -> float:
    A, B, C = t1
    D, E, F = t2
    return dset(B, E) + min(dset(A, D) + dset(C, F), dset(A, F) + dset(C, D))


def cost_matrix(t1: list, t2: list, kind: str) -> np.ndarray:
    """The r1 x r2 matrix of delta between the real tripartitions."""
    d = dset_of(kind)
    return np.array([[cost(a, b, d) for b in t2] for a in t1], dtype=float).reshape(len(t1), len(t2))


def padded(delta: np.ndarray, m: int) -> np.ndarray:
    """Pad to m x m: real-dummy 3, dummy-dummy 0."""
    r1, r2 = delta.shape
    out = np.zeros((m, m), dtype=float)
    out[:r1, :r2] = delta
    out[:r1, r2:] = DUMMY_COST
    out[r1:, :r2] = DUMMY_COST
    return out


def d_ret_raw(delta: np.ndarray) -> float:
    """D_ret (unnormalised) from the real cost matrix: pad to max(r1, r2) and solve."""
    m = max(delta.shape)
    if m == 0:
        return 0.0
    P = padded(delta, m)
    rows, cols = linear_sum_assignment(P)
    return float(P[rows, cols].sum())


# ============================================================================
# remainder.py
# ============================================================================
"""The remainder T(N): what is left of a network once every reticulation is removed.

Delete the union of the blocks below all reticulations. For every original edge
(p, h) from a surviving vertex p into a reticulation h, hang one marker leaf on p
labelled by the taxon set below h in the ORIGINAL network. Taxa keep their names
(in a separate namespace). Nothing is suppressed: unary vertices stay.

The remainder is built per network, independently of any matching, so two
reticulations with the same descendant taxa get the same marker label, and a
reticulation whose descendant set differs between the networks gets different
labels. Repeated labels are kept as distinct leaf vertices.
"""



def remainder_of(net, G: Graph, id_of: dict) -> Graph:
    """T(N) of a VALIDATED network given as (G, id_of) from ``network_graph``."""
    H = {id_of[h] for h in hybrids(net)}

    below = {}                                   # descendant taxa, read before anything is deleted
    for v in reversed(G.order()):
        if v in G.label:
            below[v] = frozenset([G.label[v]])
        else:
            below[v] = frozenset().union(*(below[c] for c in G.children[v]))
    marker_label = {h: ("marker", tuple(sorted(below[h]))) for h in H}

    deleted = set()                              # union of the blocks below every reticulation
    for h in H:
        stack = [h]
        while stack:
            v = stack.pop()
            if v in deleted:
                continue
            deleted.add(v)
            stack.extend(G.children[v])

    children, label = {}, {}
    next_id = len(G.children)
    for v in G.order():
        if v in deleted:
            continue
        kids = []
        for c in G.children[v]:
            if c in H:                           # one marker per original edge (v, h), v surviving
                m = next_id
                next_id += 1
                children[m] = ()
                label[m] = marker_label[c]
                kids.append(m)
            else:
                assert c not in deleted, "non-reticulation child of a surviving vertex inside a deleted block"
                kids.append(c)
        children[v] = tuple(kids)
        if v in G.label:
            label[v] = ("taxon", G.label[v])
    return Graph(G.root, children, label)


def remainder(net) -> Graph:
    """T(N) of a network. Validates first."""
    validate(net)
    return remainder_of(net, *network_graph(net))


# ============================================================================
# io.py
# ============================================================================
"""Reading networks. Extended Newick is parsed with PhyNetPy (``phynetpy.IO``).

Two checks run on the string before PhyNetPy sees it, because the parser would
otherwise change the topology silently: an unnamed leaf would receive an invented
name, and an internal-node label equal to a taxon name would be merged with that
taxon into a reticulation. Internal labels other than ``#H`` tags carry no
information for the measure and are removed. Every failure raises ValueError.
"""


_UNNAMED_LEAF = re.compile(r"[(,]\s*[:,)]")              # "(a,)" "(,b)" "(a,:0.5)"
_INTERNAL_LABEL = re.compile(r"\)\s*([^,:;()\[\]#\s]+)")   # ")label" but not ")#H1"


def clean_newick(newick: str) -> str:
    """The Newick string with internal-node labels (not ``#H`` tags) removed.

    Raises ValueError on an empty string or an unnamed leaf.
    """
    s = newick.strip()
    if not s:
        raise ValueError("empty Newick string")
    if not s.endswith(";"):
        s += ";"
    if _UNNAMED_LEAF.search(s):
        raise ValueError("Newick has an unnamed leaf; every taxon must be labelled")
    return _INTERNAL_LABEL.sub(")", s)


def parse(newick: str):
    """A PhyNetPy ``Network`` from an extended-Newick string (reticulations as #H tags).

    Raises ValueError when the string cannot be parsed or parses to nothing
    (a single-taxon Newick such as ``a;`` is not a network to compare).
    """
    from phynetpy import IO
    s = clean_newick(newick)
    try:
        net = IO.read_newick(s)
    except Exception as e:                       # PhyNetPy raises IOError and others
        raise ValueError(f"cannot parse Newick: {e}") from e
    if net is None or not list(net.V()):
        raise ValueError("Newick parsed to an empty network (a single taxon is not a network)")
    return net


def read_file(path: str):
    """The first network in a Newick file."""
    with open(path) as fh:
        text = fh.read().strip()
    if not text:
        raise ValueError(f"{path}: empty file")
    return parse(text.splitlines()[0])


# ============================================================================
# measure.py
# ============================================================================
"""The joint dissimilarity D_joint = D_ret + 3 D_tree and its normalised forms.

For two networks on the same taxon set and an instantiation alpha in
{leaf, cluster, nakhleh}:

    D_ret^alpha   = min_M sum_t delta^alpha(t, M(t))      raw, in [0, 3m], m = max(r1, r2)
    D_tree^alpha  = d^alpha(T(N1), T(N2))                   in [0, 1]
    D_joint^alpha = D_ret^alpha + 3 D_tree^alpha            raw, in [0, 3(m + 1)]

    normalised:  D_ret / 3m (0 when m = 0)   and   D_joint / 3(m + 1)

``score`` returns every quantity for all three instantiations in one dict.
"""



VERSION = "1.0.0"
FIELDS = ("D_ret", "D_ret_normalized", "D_tree", "D_joint", "D_joint_normalized")


@dataclass(frozen=True)
class Prepared:
    """A network reduced to what the measure needs: its tripartitions, remainder and taxa."""
    tripartitions: list
    remainder: object
    taxa: frozenset


def prepare(net) -> Prepared:
    """Validate once, build the Graph once, and derive tripartitions and remainder from it."""
    validate(net)
    G, id_of = network_graph(net)
    return Prepared(tripartitions_of(net, G, id_of), remainder_of(net, G, id_of), taxa(net))


def score_prepared(a: Prepared, b: Prepared, kinds=KINDS) -> dict:
    if a.taxa != b.taxa:
        raise ValueError("the two networks must have the same taxon set")
    m = max(len(a.tripartitions), len(b.tripartitions))
    out = {"m": m, "r1": len(a.tripartitions), "r2": len(b.tripartitions)}
    for kind in kinds:
        s = SUFFIX[kind]
        ret = d_ret_raw(cost_matrix(a.tripartitions, b.tripartitions, kind))
        tree = dset_of(kind)(a.remainder, b.remainder)
        joint = ret + 3.0 * tree
        vals = (ret, ret / (3.0 * m) if m else 0.0, tree, joint, joint / (3.0 * (m + 1)))
        out.update({f"{f}_{s}": v for f, v in zip(FIELDS, vals)})
    return out


def score(net1, net2) -> dict:
    """All fields of ``columns()`` for two PhyNetPy networks."""
    return score_prepared(prepare(net1), prepare(net2))


def columns() -> list:
    return ["m", "r1", "r2"] + [f"{f}_{SUFFIX[k]}" for k in KINDS for f in FIELDS]


class Scorer:
    """``score`` on Newick strings, caching each network's preparation (bounded LRU)."""

    def __init__(self, max_networks: int = 128):
        self.max_networks = max_networks
        self.cache = OrderedDict()
        self.hits = self.misses = 0

    def get(self, newick: str) -> Prepared:
        key = newick.strip()
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        prepared = prepare(parse(key))
        self.misses += 1
        self.cache[key] = prepared
        if len(self.cache) > self.max_networks:
            self.cache.popitem(last=False)
        return prepared

    def score(self, newick1: str, newick2: str) -> dict:
        return score_prepared(self.get(newick1), self.get(newick2))


# ============================================================================
# cli.py
# ============================================================================
"""Command line: ``python -m jointdist truth.nwk estimate.nwk [--csv]``."""




def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="jointdist", description="Joint reticulation-and-tree dissimilarity of two phylogenetic networks.")
    ap.add_argument("network1", help="extended-Newick file (first line is read)")
    ap.add_argument("network2", help="extended-Newick file (first line is read)")
    ap.add_argument("--csv", action="store_true", help="one CSV row with every column instead of a table")
    a = ap.parse_args(argv)
    try:
        r = score(read_file(a.network1), read_file(a.network2))
    except (ValueError, OSError) as e:
        print(f"jointdist: {e}", file=sys.stderr)
        return 2
    if a.csv:
        w = csv.DictWriter(sys.stdout, fieldnames=columns())
        w.writeheader()
        w.writerow(r)
        return 0
    print(f"reticulations: {r['r1']} and {r['r2']}  (m = {r['m']})")
    print(f"{'':10s}" + "".join(f"{f:>20s}" for f in FIELDS))
    for kind in KINDS:
        s = SUFFIX[kind]
        print(f"{kind:10s}" + "".join(f"{r[f'{f}_{s}']:20.6f}" for f in FIELDS))
    return 0


if __name__ == "__main__":
    sys.exit(main())


if __name__ == "__main__":
    sys.exit(main())
