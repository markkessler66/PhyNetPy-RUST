#! /usr/bin/env python
# -*- coding: utf-8 -*-

##############################################################################
##  -- PhyNetPy --
##  Library for the Development and use of Phylogenetic Network Methods
##
##  Copyright 2025 Mark Kessler, Luay Nakhleh.
##  All rights reserved.
##
##  See "LICENSE.txt" for terms and conditions of usage.
##
##  If you use this work or any portion thereof in published work,
##  please cite it as:
##
##     Mark Kessler, Luay Nakhleh. 2025.
##
##############################################################################

"""
Author : Mark Kessler
Last Edit : 2/12/26
First Included in Version : 1.1.0

SNP Data Simulator for Phylogenetic Networks.

Simulates biallelic (SNP) data by drawing one exact MSNC genealogy per
unlinked site and evolving a stationary 2-state continuous-time Markov chain
down it. This includes ILS, reticulation routing, and multiple samples per
species under the same model used by the Bryant marker likelihood.

The network generation (node relabelling, reticulation grafting) and the
per-site simulation run in the compiled core (``phynetpy._core``); every
random draw still comes from the seeded ``numpy.random.Generator`` in the
same order as before, so a seeded run reproduces.

Usage:
    from PhyNetPy.SNPSimulator import simulate, random_network

    # Generate a random level-2 network with 50 taxa
    net = random_network(n=50, level=2, seed=42)

    # Simulate 10000 SNP sites over it
    sim = simulate(n=50, s=10000, net=net, seed=42)

    # Write to nexus file for use with SNP_LIKELIHOOD
    sim.write_nexus("stress_test_50taxa.nex")

Docs   - [x]
Tests  - [x]
Design - [x]
"""

from __future__ import annotations

from dataclasses import dataclass
import operator
import warnings

import numpy as np

from . import _core
from .Network import Network, Node, Edge
from .BirthDeath import Yule
from ._sim_markers import simulate_biallelic_markers
from ._units import BranchLengthUnit, BranchThetaKey


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SimulatedSNPData:
    """
    Container for simulated SNP data and the generating network.

    Attributes:
        network (Network): The phylogenetic network used for simulation.
        data (dict[str, list[int]]): Mapping from taxon name to a list of
                                      red allele counts per site.
        n_taxa (int): Number of leaf taxa.
        n_sites (int): Number of simulated SNP sites.
        samples (dict[str, int]): Number of sampled gene copies per taxon.
        u (float): Mutation rate from red to green allele.
        v (float): Mutation rate from green to red allele.
        theta (float): Population mutation rate ``4*N*mu``.
        seed (int): Random seed used for simulation.
    """
    network: Network
    data: dict[str, list[int]]
    n_taxa: int
    n_sites: int
    samples: dict[str, int]
    u: float
    v: float
    theta: float
    seed: int | None

    def taxa_names(self) -> list[str]:
        """
        List the taxa present in the simulated data, in sorted order.

        Args:
            N/A
        Returns:
            list[str]: Taxon names, sorted lexicographically.
        """
        return sorted(self.data.keys())

    def write_nexus(self, filepath: str) -> None:
        """
        Write the simulated data and network to a NEXUS file compatible
        with SNP_LIKELIHOOD.

        The output file contains:
            - TAXA block with taxon labels
            - DATA block with one hexadecimal red-allele count per site
            - TREES block with the network in extended newick format
              (including branch lengths and gamma annotations)

        Args:
            filepath (str): Path where the nexus file will be written.
        Returns:
            None
        Raises:
            ValueError: If any taxon's count vector is not exactly n_sites
                        long, or if any red count falls outside 0..15.
        """
        if any(len(counts) != self.n_sites for counts in self.data.values()):
            raise ValueError("every taxon must have exactly n_sites counts.")
        if any(
            count < 0 or count > 15
            for counts in self.data.values()
            for count in counts
        ):
            raise ValueError(
                "NEXUS SNP rows encode counts as one hexadecimal character, "
                "so every red count must be in 0..15."
            )

        taxa = self.taxa_names()

        # annotate=True keeps branch lengths and gammas in the TREES block,
        # which SNP_LIKELIHOOD needs to rebuild the generating network.
        nwk = self.network.newick(annotate = True)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("#NEXUS\n\n")

            # TAXA block
            f.write("BEGIN TAXA;\n")
            f.write(f"DIMENSIONS NTAX={len(taxa)};\n")
            f.write(f"TAXLABELS {' '.join(taxa)};\n")
            f.write("END;\n\n")

            # DATA block
            f.write("BEGIN DATA;\n")
            f.write(f"  Dimensions nchar={self.n_sites};\n")
            f.write("  Format datatype=snp missing=? gap=- matchchar=.;\n")
            f.write("  Matrix\n")
            for taxon in taxa:
                counts = self.data[taxon]
                seq = "".join(format(count, "X") for count in counts)
                f.write(f"    {taxon} {seq}\n")
            f.write("  ;\nEND;\n\n")

            # TREES block
            f.write("BEGIN TREES;\n")
            f.write(f"Tree net = {nwk}\n")
            f.write("END;\n")


# ═══════════════════════════════════════════════════════════════════════════════
# NETWORK GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def random_network(
    n: int,
    level: int = 1,
    birth_rate: float = 1.0,
    gamma_range: tuple[float, float] = (0.2, 0.8),
    seed: int | None = None
) -> Network:
    """
    Generate a random phylogenetic network with n taxa and a given number
    of reticulations.

    First generates a random tree using a Yule (pure-birth) process, then
    rebuilds the topology with clean node names. Finally, grafts reticulation
    edges to reach the desired reticulation count.

    Note: The ``level`` parameter specifies the number of reticulation nodes
    to add, not the network level in the strict sense (max reticulations per
    biconnected component). Depending on placement, the resulting network's
    true level may be less than this value.

    Args:
        n (int): Number of leaf taxa. A tree on n leaves has only n - 2 edges
                 joining two internal nodes, and a reticulation consumes two
                 non-nested ones, so n >= 5 in practice for level >= 1.
        level (int): Number of reticulation nodes to add. Defaults to 1.
        birth_rate (float): Birth rate for the Yule process. Larger values
                            produce shorter branch lengths. Defaults to 1.0.
        gamma_range (tuple[float, float]): Range from which inheritance
                                            probabilities (gamma) are
                                            uniformly drawn. Defaults to
                                            (0.2, 0.8).
        seed (int | None): Random seed for reproducibility. Defaults to None.

    Raises:
        ValueError: If n < 2 or level < 0.

    Warns:
        UserWarning: If the topology runs out of legal insertion sites before
                     `level` reticulations are placed. The network is returned
                     with however many were added.

    Returns:
        Network: A phylogenetic network with n leaves and at most `level`
                 reticulation nodes.
    """
    if n < 2:
        raise ValueError("Need at least 2 taxa.")
    if level < 0:
        raise ValueError("Network level must be non-negative.")

    rng = np.random.default_rng(seed)

    # The Yule draw, the relabelling and the reticulation placements all run
    # in the compiled core, drawing from ``rng`` in the same order as before.
    return _core.snp_random_network(
        rng, int(n), int(level), float(birth_rate),
        float(gamma_range[0]), float(gamma_range[1]),
    )


def _clean_node_names(net: Network) -> None:
    """
    Relabel a simulator-generated tree in place to PhyNetPy naming conventions:
    leaves become T1..Tn (ordered by their original label), the root becomes
    Root, and the remaining internal nodes become I1..Ik ordered by depth.

    Renaming goes through :meth:`Network.update_node_name`, which rehashes the
    node in every name-keyed collection (``Node.__hash__`` is name-based, so a
    bare ``set_name`` would strand the node in the node/edge sets).

    Args:
        net (Network): A tree-like network from a Yule/CBDP simulator. Modified
                       in place.
    Returns:
        None
    """
    _core.snp_clean_node_names(net)


def _add_reticulation(
    net: Network,
    rng: np.random.Generator,
    gamma: float,
    retic_id: int
) -> bool:
    """
    Add one reticulation edge to an existing network.

    Strategy:
        1. Find all internal edges (between non-leaf, non-retic nodes).
        2. Pick two edges that are NOT in an ancestor-descendant relationship
           (to avoid creating cycles).
        3. Bisect each edge to create two new internal nodes.
        4. The first new node becomes a tree node; the second becomes the
           hybrid (#H) node with two parents.

    This deliberately does not delegate to ``GraphUtils.add_hybrid``, which
    performs the same bisection but leaves the four new edges without lengths
    and takes no gamma. Both are required here: the simulator evolves a
    coalescent down these branches, and ``write_nexus`` has to serialize them.

    Args:
        net (Network): The network to modify (in-place).
        rng (np.random.Generator): Random number generator.
        gamma (float): Inheritance probability for the reticulation.
        retic_id (int): Integer identifier for naming the hybrid node.

    Returns:
        bool: True if a reticulation was successfully added, False if
              not enough candidate edges were found.
    """
    return _core.snp_add_reticulation(net, rng, float(gamma), int(retic_id))


# ═══════════════════════════════════════════════════════════════════════════════
# SNP SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def simulate(
    n: int,
    s: int,
    net: Network,
    samples: dict[str, int] | None = None,
    u: float = 1.0,
    v: float = 1.0,
    theta: float = 0.02,
    branch_thetas: dict[BranchThetaKey, float] | None = None,
    seed: int | None = None
) -> SimulatedSNPData:
    """
    Simulate SNP (biallelic marker) data over a phylogenetic network.

    Draws an independent MSNC genealogy per site, evolves the exact two-state
    mutation process down it, and aggregates sampled gene copies into red
    allele counts. This matches the model consumed by ``BiMarkers`` for one
    or multiple samples per taxon.

    Args:
        n (int): Expected number of taxa — used only for validation.
                 The actual taxa come from the network's leaves.
        s (int): Number of SNP sites to simulate.
        net (Network): The phylogenetic network to simulate data on.
        samples (dict[str, int] | None): Number of sampled gene copies per
                                          taxon. Keys must match leaf names.
                                          Defaults to 1 per taxon.
        u (float): Mutation rate from red allele to green. Defaults to 1.0.
        v (float): Mutation rate from green allele to red. Defaults to 1.0.
        theta (float): Population mutation rate ``4*N*mu``.
        branch_thetas (dict | None): Fixed per-population theta overrides.
        seed (int | None): Random seed for reproducibility. Defaults to None.

    Raises:
        ValueError: If the number of leaves in the network doesn't match n,
                    or if sample keys don't match leaf names.

    Returns:
        SimulatedSNPData: Container with simulated data, network, and metadata.
    """
    leaves = net.get_leaves()
    leaf_names = sorted([leaf.label for leaf in leaves])

    if len(leaf_names) != n:
        raise ValueError(
            f"Network has {len(leaf_names)} leaves but n={n} was specified."
        )

    if samples is None:
        samples = {name: 1 for name in leaf_names}
    else:
        if set(samples) != set(leaf_names):
            raise ValueError(
                f"Sample keys {set(samples)} don't match "
                f"leaf names {set(leaf_names)}."
            )
        try:
            samples = {
                name: operator.index(samples[name]) for name in leaf_names
            }
        except TypeError as exc:
            raise ValueError("sample counts must be integers.") from exc
        if any(count <= 0 for count in samples.values()):
            raise ValueError("sample counts must be positive.")

    mapping = {
        name: [f"{name}_{i}" for i in range(samples[name])]
        for name in leaf_names
    }
    data = simulate_biallelic_markers(
        net,
        s,
        mapping,
        theta=theta,
        u=u,
        v=v,
        branch_thetas=branch_thetas,
        rng=np.random.default_rng(seed),
    )

    return SimulatedSNPData(
        network=net,
        data=data,
        n_taxa=n,
        n_sites=s,
        samples=samples,
        u=u,
        v=v,
        theta=theta,
        seed=seed,
    )
