#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
PhyNetPy walkthrough: a typical session, start to finish.

This script does what a biologist does on a new data set, in order:

1. Read a species network and look at the objects it is made of.
2. Write it out (Newick, NEXUS, PhyloNet flavour) and read it back.
3. Edit it: rename a tip, drop a taxon, add and remove a hybridization,
   list the trees it displays, and compare the edited copies to the original.
4. Get gene trees: simulate a small set under the multispecies network
   coalescent, inspect them, write them to a file, read them back.
5. Score candidate networks against the gene trees.
6. Run a short inference from the gene trees and compare the answer to the
   truth.

Everything runs in a few seconds. Run it from the repository root:

    python examples/biologist_walkthrough.py

Output files go to runs/walkthrough/.
"""

from pathlib import Path

# ── Data structures ────────────────────────────────────────────────────────
from phynetpy.Network import Network, Node, Edge
from phynetpy.models import BranchLengthUnit, MSC, convert_network_branch_lengths

# ── I/O ────────────────────────────────────────────────────────────────────
from phynetpy.IO import (
    read_newick, read_newick_file, read_nexus,
    write_newick, write_newick_file, write_nexus,
    convert_newick, detect_newick_standard,
)

# ── Network analysis and editing ──────────────────────────────────────────
from phynetpy.GraphUtils import (
    ascii, count_reticulations, level, is_tree,
    get_displayed_trees, dominant_tree, induced_subnetwork_by_taxa,
    add_hybrid, pairwise_leaf_distance, compare_networks, joint_dissimilarity,
)

# ── Data, inference, scoring ──────────────────────────────────────────────
from phynetpy.data import GeneTrees
from phynetpy.criteria import Likelihood, PseudoLikelihood
from phynetpy.infer import infer, score, simulate


OUT = Path("runs") / "walkthrough"
OUT.mkdir(parents=True, exist_ok=True)


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# ══════════════════════════════════════════════════════════════════════════
# 1. Read a species network and look at its parts
# ══════════════════════════════════════════════════════════════════════════
banner("1. A species network from extended Newick")

# Five species. B's lineage is a hybrid: 65% of its ancestry comes from the
# A side (#H1's first parent edge), 35% from the D side. Branch lengths are
# expected substitutions per site and the tree is ultrametric (every tip is
# 0.18 from the root).
TRUE_NEWICK = (
    "(((((A:0.04,B:0.04)AB:0.03)#H1:0.02[&gamma=0.65],C:0.09)ABC:0.04,"
    "(#H1:0.04[&gamma=0.35],D:0.11)DR:0.02)R:0.05,E:0.18)Root;"
)
net = read_newick(TRUE_NEWICK)
net.set_branch_length_unit(BranchLengthUnit.SUBSTITUTIONS_PER_SITE)

print(ascii(net))
print(f"Nodes ({len(net.V())}): {sorted(n.label for n in net.V())}")
print(f"Edges: {len(net.E())}   Leaves: {sorted(l.label for l in net.get_leaves())}")
print(f"Root: {net.root().label}   Reticulations: {count_reticulations(net)}   "
      f"Level: {level(net)}   Is a tree? {is_tree(net)}")

# A Node knows its name, whether it is a reticulation, and its time (height
# from the root, filled in from branch lengths when parsing).
hybrid = next(n for n in net.V() if n.is_reticulation())
print(f"\nReticulation node {hybrid.label}: time from root = {hybrid.get_time():.3f}")

# An Edge carries its two endpoints, a length, and (on reticulation edges)
# the inheritance probability gamma.
for e in net.in_edges(hybrid):
    print(f"  in-edge {e.src.label} -> {e.dest.label}: "
          f"length {e.get_length():.3f}, gamma {e.get_gamma():.2f}")

# Patristic distances between tips (keyed by tip label).
d = pairwise_leaf_distance(net, use_branch_lengths=True)
print("\nClosest tip pairs:")
for (u, v), dist in sorted(d.items(), key=lambda kv: kv[1])[:3]:
    print(f"  d({u}, {v}) = {dist:.3f}")


# ══════════════════════════════════════════════════════════════════════════
# 2. Write it out, read it back
# ══════════════════════════════════════════════════════════════════════════
banner("2. I/O round trips")

nwk = write_newick(net, annotate=True)         # with lengths and gammas
print(f"Newick (PhyNetPy style, {detect_newick_standard(nwk)}):\n  {nwk}")
print(f"PhyloNet style:\n  {convert_newick(nwk, standard='Phylonet')}")

# write_newick_file / write_nexus store topology only (one tree per line /
# a TREES block). To keep branch lengths and gammas, write the annotated
# string yourself; read_newick_file parses either form.
newick_path = OUT / "species_network.nwk"
nexus_path = OUT / "species_network.nex"
newick_path.write_text(nwk + "\n")
write_nexus([net], str(nexus_path), tree_prefix="species")
print(f"\nWrote {newick_path} (with lengths) and {nexus_path} (topology)")

back_from_newick = read_newick_file(str(newick_path))[0]
back_from_nexus = read_nexus(str(nexus_path))[0]
tip_heights = {l.label: round(l.get_time(), 6) for l in back_from_newick.get_leaves()}
print(f"Read back from Newick: same topology? {net.is_isomorphic(back_from_newick)}; "
      f"tip heights kept? {set(tip_heights.values()) == {0.18}}")
print(f"Read back from NEXUS : same topology? {net.is_isomorphic(back_from_nexus)}")


# ══════════════════════════════════════════════════════════════════════════
# 3. Edit the network
# ══════════════════════════════════════════════════════════════════════════
banner("3. Editing networks")

# copy() returns the copy and the old->new node map; edits never touch `net`.
edited, _ = net.copy()

# Rename a tip (e.g. a sample code to a species name).
edited.update_node_name(edited.has_node_named("E"), "Outgroup")
print(f"Renamed E -> Outgroup: leaves are now {sorted(l.label for l in edited.get_leaves())}")

# Keep only some taxa: the induced subnetwork drops the rest and cleans up.
four_taxa = induced_subnetwork_by_taxa(net, ["A", "B", "C", "D"])
print(f"\nRestricted to A,B,C,D: {write_newick(four_taxa)}")
print(f"  reticulations kept: {count_reticulations(four_taxa)}")

# Remove the minor hybrid edge: the network collapses to a tree.
tree_only, _ = net.copy()
h = next(n for n in tree_only.V() if n.is_reticulation())
minor_edge = min(tree_only.in_edges(h), key=lambda e: e.get_gamma())
tree_only.remove_edge(minor_edge)
tree_only.clean()                              # suppress degree-2 nodes
print(f"\nMinor hybrid edge removed: {write_newick(tree_only)}")
print(f"  is a tree? {is_tree(tree_only)}")

# Build a tree by hand from Nodes and Edges, then add a hybridization.
root, i1, i2 = Node("Root", t=0.0), Node("I1", t=0.5), Node("I2", t=0.5)
x, y, z = Node("X", t=1.0), Node("Y", t=1.0), Node("Z", t=1.0)
hand = Network()
hand.add_nodes(root, i1, i2, x, y, z)
hand.add_edges([
    Edge(root, i1, length=0.5), Edge(root, i2, length=0.5),
    Edge(i1, x, length=0.5), Edge(i1, y, length=0.5), Edge(i2, z, length=0.5),
])
hand.set_branch_length_unit(BranchLengthUnit.SUBSTITUTIONS_PER_SITE)
add_hybrid(hand, source=hand.get_edge(i2, z), destination=hand.get_edge(i1, y),
           t_src=0.75, t_dest=0.75)           # gene flow from Z's lineage into Y
# add_hybrid creates the reticulation; set how much ancestry each parent contributes.
new_hybrid = next(n for n in hand.V() if n.is_reticulation())
for e in hand.in_edges(new_hybrid):
    e.set_gamma(0.3 if e.src.label == "UID_0" else 0.7)   # 30% from Z's lineage
print(f"\nHand-built tree with a hybrid edge added: {write_newick(hand, annotate=True)}")
print(ascii(hand))

# A network with k reticulations displays up to 2^k trees.
shown = get_displayed_trees(net)
print(f"Trees displayed by the species network ({len(shown)}):")
for t in shown:
    print(f"  {write_newick(t)}")
print(f"Dominant tree (major edges only): {write_newick(dominant_tree(net))}")

# Compare networks: how different is each edited copy from the original?
print("\nDistances from the original network:")
for name, other in [("tree (hybrid removed)", tree_only), ("restricted to 4 taxa", four_taxa)]:
    if set(l.label for l in other.get_leaves()) != set(l.label for l in net.get_leaves()):
        print(f"  {name:24s}: different taxon sets, compare on shared taxa instead")
        continue
    cmp = compare_networks(net, other)
    jd = joint_dissimilarity(net, other, kind="cluster")
    print(f"  {name:24s}: reticulation dissimilarity {cmp.D_ret:.3f}, "
          f"joint dissimilarity {jd.D_joint_normalized:.3f}")


# ══════════════════════════════════════════════════════════════════════════
# 4. Gene trees: simulate, inspect, save, load
# ══════════════════════════════════════════════════════════════════════════
banner("4. Gene trees")

THETA = 0.02    # population mutation rate (4N mu), in substitutions per site
mapping = {sp: [sp] for sp in ["A", "B", "C", "D", "E"]}   # one allele per species

gts = simulate(MSC(theta=THETA), network=net, n=60, data="gene_trees",
               mapping=mapping, seed=7)
print(f"Simulated {len(gts.trees)} gene trees along the species network")
print(f"Alleles: {sorted(gts.taxa_names)}")

print("\nSupport for the true clades across the gene trees:")
for clade, frac in sorted(gts.cluster_support().items(), key=lambda kv: -kv[1])[:5]:
    print(f"  {{{', '.join(sorted(clade))}}}: {frac:.2f}")

consensus = gts.build_majority_rule_consensus_tree(threshold=0.5)
print(f"\nMajority-rule consensus: {write_newick(consensus)}")

gene_tree_path = OUT / "gene_trees.nwk"
write_newick_file(list(gts.trees), str(gene_tree_path))
loaded = GeneTrees.from_file(gene_tree_path, mapping)   # Newick inferred from .nwk
print(f"Wrote {len(gts.trees)} trees to {gene_tree_path}; read back {len(loaded.trees)}")


# ══════════════════════════════════════════════════════════════════════════
# 5. Score candidate networks against the gene trees
# ══════════════════════════════════════════════════════════════════════════
banner("5. Scoring")

# Gene-tree likelihoods work in coalescent units (2N generations); convert.
true_coal = convert_network_branch_lengths(net, BranchLengthUnit.COALESCENT_2N, theta=THETA)
tree_coal = convert_network_branch_lengths(tree_only, BranchLengthUnit.COALESCENT_2N, theta=THETA)

for name, candidate in [("true network", true_coal), ("tree without the hybrid", tree_coal)]:
    ll = score(candidate, gts, model=MSC(), criterion=Likelihood())
    pl = score(candidate, gts, model=MSC(), criterion=PseudoLikelihood())
    print(f"{name:24s}: log likelihood {ll:10.3f}   log pseudo-likelihood {pl:10.3f}")


# ══════════════════════════════════════════════════════════════════════════
# 6. Infer a network from the gene trees
# ══════════════════════════════════════════════════════════════════════════
banner("6. Inference (short pseudo-likelihood search)")

# A deliberately short search (num_iter=300) so the script stays quick; the
# move statistics the search prints are its end-of-run diagnostics. A real
# analysis uses the default budget (num_iter=700 or more) and several runs.
result = infer(gts, model=MSC(), criterion=PseudoLikelihood(),
               max_reticulations=1, num_iter=300, print_comparison=False)
print(result)
print(f"Best network: {write_newick(result.best, annotate=True)}")
print(f"Reticulations found: {count_reticulations(result.best)}")

true_pl = score(true_coal, gts, model=MSC(), criterion=PseudoLikelihood())
cmp = compare_networks(net, result.best)
print(f"\nFound {result.score:.3f} vs true network {true_pl:.3f} (log pseudo-likelihood)")
print(f"Against the truth: reticulation dissimilarity {cmp.D_ret:.3f}, "
      f"same topology? {net.is_isomorphic(result.best)}")
print(f"Major tree of the result: {write_newick(dominant_tree(result.best))}")
print(f"Major tree of the truth : {write_newick(dominant_tree(net))}")
print("\nDone. Files are in", OUT.resolve())
