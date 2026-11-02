#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compare a DIMPLE directory to infer_dimple.

Network distance is the result. MPL scores are written down and do not
decide whether the two runs agree. The directory is the colleague's
DIMPLE output; this script does not vendor it.

    python scripts/dimple_parity_report.py /path/to/dimple_out

Writes reports/dimple_parity.md and prints the same summary.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from phynetpy.GraphUtils import (
    count_reticulations,
    displayed_tree_distance,
    hardwired_cluster_distance,
    mu_distance,
    robinson_foulds_distance,
    softwired_cluster_distance,
    tripartition_distance,
)
from phynetpy.Network import Network
from phynetpy.ReticulationComparison import compare_networks
from phynetpy.data import GeneTrees
from phynetpy.infer import infer_dimple

_SCORE_PATTERNS = (
    re.compile(r"Total log probability[:\s]+(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", re.I),
    re.compile(r"log(?:arithmic)? pseudo-?likelihood[:\s]+(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", re.I),
)


def classify_pair(reference: Network, inferred: Network) -> tuple[str, dict, object]:
    """Distance class of one DIMPLE network against one PhyNetPy network.

    Args:
        reference (Network): The DIMPLE network.
        inferred (Network): The PhyNetPy network.

    Returns:
        tuple[str, dict, NetworkComparison]: The class (``identical``,
        ``same reticulations, different backbone``, or
        ``different reticulations``), the distance dictionary, and
        ``compare_networks``'s result.
    """
    distances = {
        "hardwired_cluster": hardwired_cluster_distance(reference, inferred, normalize=True),
        "softwired_cluster": softwired_cluster_distance(reference, inferred, normalize=True),
        "mu": mu_distance(reference, inferred),
        "tripartition": tripartition_distance(reference, inferred, normalize=True),
        "displayed_tree": displayed_tree_distance(reference, inferred, normalize=True),
        "robinson_foulds": robinson_foulds_distance(reference, inferred, normalize=True),
    }
    comparison = compare_networks(reference, inferred)
    zeros = all(value == 0 for value in distances.values())
    if zeros and comparison.D_ret_hat == 0:
        label = "identical"
    elif comparison.D_ret_hat == 0:
        label = "same reticulations, different backbone"
    else:
        label = "different reticulations"
    return label, distances, comparison


def score_note(dimple_under_phynetpy, phynetpy_score, printed) -> str:
    """Three scores, when they exist. Inequality is text, not a failure.

    Args:
        dimple_under_phynetpy: PhyNetPy's score of the DIMPLE network.
        phynetpy_score: PhyNetPy's score of its own network.
        printed: The score PhyloNet printed, or None.

    Returns:
        str: One line naming both gaps.
    """
    parts = [
        f"PhyNetPy(DIMPLE network)={dimple_under_phynetpy}",
        f"PhyNetPy(PhyNetPy network)={phynetpy_score}",
    ]
    if (
        dimple_under_phynetpy is not None
        and phynetpy_score is not None
    ):
        parts.append(
            f"search gap={phynetpy_score - dimple_under_phynetpy}"
        )
    if printed is None:
        parts.append("PhyloNet printed=(absent)")
    else:
        parts.append(f"PhyloNet printed={printed}")
        if dimple_under_phynetpy is not None:
            parts.append(
                f"numeric gap={dimple_under_phynetpy - printed}"
            )
    return "; ".join(parts)


def _printed_score(text: str):
    """A PhyloNet log-probability from ``text``, or None."""
    for pattern in _SCORE_PATTERNS:
        match = pattern.search(text)
        if match:
            return float(match.group(1))
    return None


def _read_newick(path: Path) -> Network:
    return Network.from_newick(path.read_text().strip().splitlines()[0])


def load_dimple_directory(directory: Path) -> dict:
    """Read the flowchart files under ``directory``.

    Missing pieces are None or empty. The reader does not run DIMPLE.

    Args:
        directory (Path): A DIMPLE output directory, or the directory that
            contains ``gene_trees.tre``, ``tob_rooted.tre``, and ``astral.tre``.

    Returns:
        dict: Inputs, the final network, per-blob taxon sets, subset
        networks, and any printed scores that were found.
    """
    directory = Path(directory)
    found = {
        "gene_trees": None,
        "tree_of_blobs": None,
        "start_tree": None,
        "final": None,
        "blobs": [],
        "printed_scores": [],
    }
    for name, key in (
        ("gene_trees.tre", "gene_trees"),
        ("tob_rooted.tre", "tree_of_blobs"),
        ("astral.tre", "start_tree"),
        ("dimple_network.nwk", "final"),
    ):
        path = directory / name
        if not path.is_file():
            alt = directory.parent / name
            path = alt if alt.is_file() else path
        if path.is_file() and key != "final":
            if key == "gene_trees":
                found[key] = path
            else:
                found[key] = _read_newick(path)
        elif path.is_file():
            found[key] = _read_newick(path)

    divisions = directory / "divisions"
    if not divisions.is_dir():
        divisions = directory
    for meta in sorted(divisions.glob("blob*/run_*/subnetworks_output_metadata.csv")):
        blob_name = meta.parents[1].name
        rows = list(csv.DictReader(meta.open()))
        subnet_lines = []
        subnets_path = meta.parent / "subgenes-out" / "subnets.txt"
        if subnets_path.is_file():
            subnet_lines = subnets_path.read_text().splitlines()
        subsets = []
        for index, row in enumerate(rows):
            if row.get("type") != "blob_group":
                continue
            leaves = frozenset(
                label for label in row.get("all_leaves", "").split(",") if label
            )
            network = None
            if index < len(subnet_lines) and subnet_lines[index].strip():
                network = Network.from_newick(subnet_lines[index].strip())
            subsets.append({"leaves": leaves, "network": network})
        merged = None
        for candidate in (
            meta.parents[1] / "merged_network.nwk",
            meta.parent / "merged_network.nwk",
        ):
            if candidate.is_file():
                merged = _read_newick(candidate)
                break
        if merged is None:
            for candidate in meta.parents[1].glob("*_combined.nwk"):
                merged = _read_newick(candidate)
                break
        found["blobs"].append({
            "name": blob_name,
            "subsets": subsets,
            "merged": merged,
        })
        for text_path in meta.parent.rglob("*.txt"):
            printed = _printed_score(text_path.read_text(errors="replace"))
            if printed is not None:
                found["printed_scores"].append(printed)
    return found


def _phy_score(network: Network, genes: GeneTrees, taxa) -> float | None:
    """PhyNetPy pseudo-likelihood of ``network`` on ``taxa``, or None if it cannot be scored."""
    from phynetpy._dimple import restrict_gene_trees
    from phynetpy.criteria import PseudoLikelihood
    from phynetpy.infer import score
    from phynetpy.models import MSC

    try:
        subset = restrict_gene_trees(genes, sorted(taxa))
        return float(score(
            network, subset, model=MSC(), criterion=PseudoLikelihood(),
        ))
    except Exception:
        return None


def render_report(blocks: list[str]) -> str:
    """Join report blocks into one markdown document."""
    return "\n".join(blocks).rstrip() + "\n"


def run_parity(
    directory: Path,
    *,
    max_subset_size: int = 12,
    n_divisions: int = 15,
    max_reticulations: int = 1,
    seed: int = 0,
    outgroup: str | None = "OUT",
    num_iter: int | None = None,
) -> tuple[str, dict]:
    """Rerun infer_dimple on a DIMPLE directory and describe the differences.

    Args:
        directory (Path): DIMPLE output directory.
        max_subset_size (int): S used for that run.
        n_divisions (int): K used for that run.
        max_reticulations (int): Reticulation bound used for that run.
        seed (int): Partition seed used for that run.
        outgroup (str | None): Scaffold leaf that run stripped.
        num_iter (int | None): Forwarded to each subset search when set.

    Returns:
        tuple[str, dict]: The markdown report and a summary with
        ``leaf_sets_match`` and class counts.

    Raises:
        FileNotFoundError: If an input or the final network is missing.
        ValueError: If the two final networks do not have the same leaves.
    """
    loaded = load_dimple_directory(Path(directory))
    if loaded["gene_trees"] is None or loaded["tree_of_blobs"] is None or loaded["start_tree"] is None:
        raise FileNotFoundError(
            "Need gene_trees.tre, tob_rooted.tre, and astral.tre."
        )
    if loaded["final"] is None:
        raise FileNotFoundError("Need dimple_network.nwk.")

    search = {}
    if num_iter is not None:
        search["num_iter"] = num_iter
    lines = [
        line.strip() for line in Path(loaded["gene_trees"]).read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    genes = GeneTrees(trees=[Network.from_newick(line) for line in lines])

    result = infer_dimple(
        genes,
        loaded["tree_of_blobs"],
        loaded["start_tree"],
        max_subset_size=max_subset_size,
        n_divisions=n_divisions,
        max_reticulations=max_reticulations,
        seed=seed,
        outgroup=outgroup,
        **search,
    )
    dimple_leaves = {leaf.label for leaf in loaded["final"].get_leaves()}
    phy_leaves = {leaf.label for leaf in result.best.get_leaves()}
    if dimple_leaves != phy_leaves:
        raise ValueError(
            "Final leaf sets differ: "
            f"only DIMPLE {sorted(dimple_leaves - phy_leaves)[:8]}, "
            f"only PhyNetPy {sorted(phy_leaves - dimple_leaves)[:8]}."
        )

    blocks = ["# DIMPLE parity", ""]
    grouped: dict[str, dict] = {}
    for blob in loaded["blobs"]:
        slot = grouped.setdefault(
            blob["name"], {"name": blob["name"], "subsets": [], "merged": None},
        )
        slot["subsets"].extend(blob["subsets"])
        if blob["merged"] is not None:
            slot["merged"] = blob["merged"]
    dimple_blobs = [grouped[name] for name in sorted(grouped)]
    dimple_parts = {
        frozenset(item["leaves"] for item in blob["subsets"])
        for blob in dimple_blobs
    }
    phy_parts = {
        frozenset(subset.taxa for subset in blob.subsets)
        for blob in result.raw.blobs
    }
    parts_match = dimple_parts == phy_parts if dimple_blobs else None
    if parts_match is None:
        blocks.append("Partitions: DIMPLE metadata was not in this directory.")
    elif parts_match:
        blocks.append("Partitions: the subset taxon sets match.")
    else:
        blocks.append(
            "Partitions: the subset taxon sets differ, so subset networks "
            "are not paired. The final networks are still compared."
        )
    blocks.append("")

    counts = {
        "identical": 0,
        "same reticulations, different backbone": 0,
        "different reticulations": 0,
    }

    def add_pair(title, reference, inferred, taxa, printed=None):
        if reference is None or inferred is None:
            blocks.append(f"## {title}: not paired")
            blocks.append("")
            return
        ref_leaves = {leaf.label for leaf in reference.get_leaves()}
        inf_leaves = {leaf.label for leaf in inferred.get_leaves()}
        if ref_leaves != inf_leaves:
            blocks.append(
                f"## {title}: not paired (leaf sets {sorted(ref_leaves)} vs {sorted(inf_leaves)})"
            )
            blocks.append("")
            return
        label, distances, comparison = classify_pair(reference, inferred)
        counts[label] += 1
        dimple_score = _phy_score(reference, genes, taxa)
        phy_score = _phy_score(inferred, genes, taxa)
        blocks.append(f"## {title}: {label}")
        blocks.append("")
        blocks.append(
            f"reticulations DIMPLE={count_reticulations(reference)} "
            f"PhyNetPy={count_reticulations(inferred)}"
        )
        blocks.append(
            "distances: " + ", ".join(f"{name}={value}" for name, value in distances.items())
        )
        blocks.append(
            f"D_ret_hat={comparison.D_ret_hat} precision={comparison.precision} "
            f"recall={comparison.recall} f1={comparison.f1}"
        )
        blocks.append("scores: " + score_note(dimple_score, phy_score, printed))
        blocks.append("")
        return label

    if parts_match:
        phy_by_taxa = {
            subset.taxa: subset
            for blob in result.raw.blobs
            for subset in blob.subsets
        }
        for blob in dimple_blobs:
            for item in blob["subsets"]:
                phy = phy_by_taxa.get(item["leaves"])
                add_pair(
                    f"Subset {blob['name']} {sorted(item['leaves'])}",
                    item["network"],
                    None if phy is None else phy.network,
                    item["leaves"],
                )
        for dimple_blob in dimple_blobs:
            phy_merged = next(
                (
                    blob.merged for blob in result.raw.blobs
                    if frozenset(subset.taxa for subset in blob.subsets)
                    == frozenset(item["leaves"] for item in dimple_blob["subsets"])
                ),
                None,
            )
            if dimple_blob["merged"] is not None and phy_merged is not None:
                add_pair(
                    f"Blob {dimple_blob['name']}",
                    dimple_blob["merged"],
                    phy_merged,
                    set().union(*(item["leaves"] for item in dimple_blob["subsets"])),
                )

    printed = loaded["printed_scores"][0] if len(loaded["printed_scores"]) == 1 else None
    final_class = add_pair(
        "Final network", loaded["final"], result.best, dimple_leaves, printed,
    )
    blocks.append(
        "counts: " + ", ".join(f"{name}={count}" for name, count in counts.items())
    )
    text = render_report(blocks)
    return text, {
        "leaf_sets_match": True,
        "partitions_match": parts_match,
        "counts": counts,
        "final_class": final_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare DIMPLE output to infer_dimple.")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--out", type=Path, default=Path("reports/dimple_parity.md"))
    parser.add_argument("--size", type=int, default=12)
    parser.add_argument("--divisions", type=int, default=15)
    parser.add_argument("--reticulations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outgroup", default="OUT")
    parser.add_argument("--num-iter", type=int, default=None)
    args = parser.parse_args()
    text, summary = run_parity(
        args.directory,
        max_subset_size=args.size,
        n_divisions=args.divisions,
        max_reticulations=args.reticulations,
        seed=args.seed,
        outgroup=args.outgroup or None,
        num_iter=args.num_iter,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(text)
    print(summary)


if __name__ == "__main__":
    main()
