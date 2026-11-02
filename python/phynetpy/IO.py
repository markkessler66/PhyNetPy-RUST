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
Last Stable Edit : 2/6/26
First Included in Version : 1.1.0
Approved for Release: No.

IO module for PhyNetPy. Handles reading and writing of various phylogenetic
file formats, starting with FASTA. Serves as the central I/O hub for all
supported file formats.

The parsers and writers run in the compiled core (``phynetpy._core``); this
module keeps the public API, the ``IOError`` class and the glue that turns
parsed records into ``DataSequence`` / ``MSA`` / ``GeneTrees`` objects.
"""

from __future__ import annotations
import os
import traceback
import warnings
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from . import _core
from .MSA import DataSequence, MSA
from .Network import Edge, Network, Node
from .GeneTrees import GeneTrees


#####################
#### Error Class ####
#####################

class IOError(Exception):
    """
    Exception raised when file I/O operations fail within PhyNetPy.
    """
    def __init__(self, message: str = "An I/O error occurred") -> None:
        """
        Initialize an IOError with a descriptive message.

        Args:
            message (str): Custom error message describing the I/O failure.
        """
        self.message = message
        super().__init__(self.message)


# The core raises this class for the failures the readers and writers report.
_core._register_io_error(IOError)


###############################
#### FASTA Reading Functions ##
###############################

def read_fasta_records(filepath: str) -> List[DataSequence]:
    """
    Read a FASTA file and return a list of DataSequence objects.
    
    This is the lower-level reader that returns raw DataSequence objects 
    without wrapping them in an MSA. Useful for attaching sequences directly 
    to Node objects in an existing Network (via Node.set_seq()).

    A FASTA file looks like:
        >sequence_name_1
        ATCGATCGATCG...
        >sequence_name_2
        GCTAGCTAGCTA...

    Each record becomes a DataSequence where:
        - name = the FASTA header (sequence ID)
        - seq  = list of characters from the sequence string

    Args:
        filepath (str): Path to a FASTA file (.fasta, .fas, .fa, .fna, .ffn, 
                         .faa).

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If the file cannot be parsed or it contains no valid 
                 sequences.

    Returns:
        list[DataSequence]: A list of DataSequence objects, one per FASTA 
                            record.
    """
    records: List[DataSequence] = []
    for idx, (name, seq) in enumerate(_core.io_read_fasta_records(filepath)):
        records.append(DataSequence(list(seq), name, gid=idx))
    return records


def read_fasta(
    filepath: str,
    grouping: Optional[Dict[str, list]] = None,
    grouping_auto_detect: bool = False) -> MSA:
    """
    Read a FASTA file and return an MSA object containing all sequences.

    This function parses a FASTA file, converts each record into a 
    DataSequence, and wraps them in an MSA for downstream phylogenetic 
    analyses such as distance calculations, alignment inspection, or 
    model-based inference.

    Args:
        filepath (str): Path to a FASTA file (.fasta, .fas, .fa, .fna, 
                         .ffn, .faa).
        grouping (dict[str, list], optional): A mapping from group names to 
                                              lists of sequence names that 
                                              belong to that group. If 
                                              provided, sequences will be 
                                              assigned group IDs accordingly. 
                                              Defaults to None.
        grouping_auto_detect (bool, optional): If True, attempt to 
                                               automatically group sequences 
                                               by name similarity. 
                                               Defaults to False.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If the file cannot be parsed or contains no valid sequences.

    Returns:
        MSA: A Multiple Sequence Alignment object containing all parsed 
             sequences.
    """
    records = read_fasta_records(filepath)
    
    # Build the MSA from the data list
    msa = MSA(data=records)
    
    # Apply grouping if provided
    if grouping is not None:
        # Rebuild group assignments based on the provided grouping map
        gid = 0
        msa.grouping = grouping
        msa.hash = {}
        msa.name2gid = {}
        
        for group_name, members in grouping.items():
            msa.name2gid[group_name] = gid
            msa.hash[gid] = []
            
            for rec in records:
                if rec.get_name() in members:
                    rec.gid = gid
                    msa.hash[gid].append(rec)
            
            gid += 1
        
        msa.groups = len(grouping)
    elif grouping_auto_detect:
        # Use the MSA's retroactive grouping to auto-detect groups
        msa.retroactive_group()
        msa.groups = len(msa.hash)
    
    return msa


###############################
#### FASTA Writing Functions ##
###############################

FASTA_LINE_WIDTH: int = 80
"""Standard FASTA line width for sequence wrapping."""


def write_fasta(msa: MSA, filepath: str, line_width: int = FASTA_LINE_WIDTH) -> None:
    """
    Write an MSA object to a FASTA file.

    Each DataSequence in the MSA is written as a FASTA record:
        >sequence_name
        ATCGATCG...  (wrapped at line_width characters)

    Args:
        msa (MSA): The Multiple Sequence Alignment to write.
        filepath (str): The output file path. Will be created or overwritten.
        line_width (int, optional): Number of characters per sequence line. 
                                     Standard FASTA convention is 80. 
                                     Defaults to 80.

    Raises:
        IOError: If the MSA has no records to write, or if the file 
                 cannot be written.
        ValueError: If line_width is less than 1.

    Returns:
        None
    """
    _core.io_write_fasta(msa, os.fspath(filepath), int(line_width))


def write_fasta_from_network(
    network: Network, 
    filepath: str, 
    line_width: int = FASTA_LINE_WIDTH) -> None:
    """
    Extract sequences from the leaf nodes of a Network and write them to a 
    FASTA file.

    Only leaf nodes that have an associated DataSequence (set via 
    Node.set_seq()) will be written. The node label becomes the FASTA 
    header, and the attached sequence becomes the FASTA sequence.

    This is useful when a Network has been annotated with molecular data 
    and the user wants to export just the sequence data.

    Args:
        network (Network): A phylogenetic network whose leaf nodes may 
                           carry DataSequence objects.
        filepath (str): The output FASTA file path.
        line_width (int, optional): Characters per line for sequence 
                                     wrapping. Defaults to 80.

    Raises:
        IOError: If no leaf nodes in the network have sequence data 
                 attached, or if the file cannot be written.
        ValueError: If line_width is less than 1.

    Returns:
        None
    """
    if line_width < 1:
        raise ValueError("line_width must be >= 1")
    
    leaves = network.get_leaves()
    
    # Collect leaf nodes that have sequence data
    seq_records: List[DataSequence] = []
    for leaf in leaves:
        seq = leaf.get_seq()
        if seq is not None:
            # Use the node label as the sequence name if the DataSequence 
            # doesn't already have one
            if seq.get_name() == "" or seq.get_name() is None:
                seq.name = str(leaf)
            seq_records.append(seq)
    
    if len(seq_records) == 0:
        raise IOError(
            "No leaf nodes in the network have sequence data attached. "
            "Use Node.set_seq(DataSequence) to attach sequences before "
            "writing."
        )
    
    # Build a temporary MSA to reuse the write_fasta function
    temp_msa = MSA(data=seq_records)
    write_fasta(temp_msa, filepath, line_width=line_width)


############################
#### VCF Reading Functions #
############################

def read_vcf(
    filepath: str,
    grouping: Optional[Dict[str, list]] = None,
    missing_value: str = "?") -> MSA:
    """
    Read a VCF (Variant Call Format) file and return an MSA object.

    Each sample in the VCF becomes a DataSequence whose sequence is the 
    vector of ALT allele counts across all variant sites. This maps 
    directly to the SNP/BiMarkers pipeline used in PhyNetPy.

    A typical VCF file looks like::

        ##fileformat=VCFv4.1
        ##INFO=<...>
        #CHROM  POS  ID  REF  ALT  QUAL  FILTER  INFO  FORMAT  Samp1  Samp2
        chr1    100  .   A    T    30    PASS    .     GT      0/0    0/1
        chr1    200  .   G    C    50    PASS    .     GT      1/1    0/1

    Genotype encoding:
        - 0/0 -> 0  (homozygous reference, 0 copies of ALT allele)
        - 0/1 -> 1  (heterozygous, 1 copy of ALT allele)
        - 1/1 -> 2  (homozygous alternate, 2 copies of ALT allele)
        - ./. -> missing_value  (missing genotype)

    Args:
        filepath (str): Path to a VCF file (.vcf).
        grouping (dict[str, list], optional): A mapping from group/species 
                                              names to lists of sample names 
                                              that belong to that group. 
                                              Used for the BiMarkers 
                                              pipeline where multiple 
                                              individuals map to a single 
                                              species. Defaults to None.
        missing_value (str, optional): The character to use for missing 
                                       genotype data (./.). Defaults to "?".

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If the file cannot be parsed or contains no variant data.

    Returns:
        MSA: A Multiple Sequence Alignment where each DataSequence 
             represents one sample's genotype vector across all sites.
    """
    sample_names, sample_genotypes = _core.io_read_vcf_records(
        os.fspath(filepath), missing_value
    )
    
    # Build DataSequence objects
    records: List[DataSequence] = []
    
    if grouping is not None:
        # Assign group IDs based on the grouping map
        gid_map: Dict[str, int] = {}
        gid = 0
        for group_name, members in grouping.items():
            for member in members:
                gid_map[member] = gid
            gid += 1
        
        for sname in sample_names:
            seq = sample_genotypes[sname]
            assigned_gid = gid_map.get(sname, -1)
            if assigned_gid == -1:
                warnings.warn(
                    f"Sample '{sname}' not found in grouping map; "
                    f"assigned to its own group."
                )
                assigned_gid = gid
                gid += 1
            records.append(DataSequence(seq, sname, gid=assigned_gid))
    else:
        for idx, sname in enumerate(sample_names):
            seq = sample_genotypes[sname]
            records.append(DataSequence(seq, sname, gid=idx))
    
    msa = MSA(data=records)
    
    # Apply grouping to MSA if provided
    if grouping is not None:
        msa.grouping = grouping
        msa.groups = len(grouping)
    
    return msa


def read_vcf_metadata(filepath: str) -> Dict[str, Any]:
    """
    Read only the metadata and header from a VCF file without loading 
    all variant data. Useful for inspecting what samples and fields 
    are available before a full parse.

    Args:
        filepath (str): Path to a VCF file.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If the file cannot be read.

    Returns:
        dict[str, Any]: A dictionary containing:
            - "fileformat": The VCF version string
            - "metadata_lines": List of all ## header lines
            - "sample_names": List of sample column names
            - "info_fields": List of INFO field IDs
            - "format_fields": List of FORMAT field IDs
            - "filter_fields": List of FILTER field IDs
            - "contig_fields": List of contig IDs
    """
    return _core.io_read_vcf_metadata(os.fspath(filepath))


############################
#### VCF Writing Functions #
############################

def write_vcf(
    msa: MSA,
    filepath: str,
    chrom: str = "chr1",
    start_pos: int = 1,
    ref_allele: str = "A",
    alt_allele: str = "T") -> None:
    """
    Write an MSA of SNP/allele-count data to a simplified VCF file.

    This produces a minimal VCF where each site in the MSA becomes a 
    variant record, and each DataSequence becomes a sample column. The 
    allele count values (0, 1, 2, ...) are converted back to VCF genotype 
    notation (e.g., 0/0, 0/1, 1/1).

    Note: Because the MSA does not store chromosome position, reference 
    alleles, or other VCF-specific metadata, this output is a simplified 
    reconstruction. It is suitable for round-tripping SNP data or creating 
    test files, but will not preserve full VCF metadata from an original 
    file.

    Args:
        msa (MSA): The MSA containing allele count data (values like 
                    0, 1, 2 per site).
        filepath (str): The output VCF file path.
        chrom (str, optional): Chromosome name for all records. 
                                Defaults to "chr1".
        start_pos (int, optional): Starting position for the first 
                                    variant. Each subsequent variant 
                                    increments by 1. Defaults to 1.
        ref_allele (str, optional): Reference allele character. 
                                     Defaults to "A".
        alt_allele (str, optional): Alternate allele character. 
                                     Defaults to "T".

    Raises:
        IOError: If the MSA has no records, or the file cannot be written.

    Returns:
        None
    """
    _core.io_write_vcf(
        msa, os.fspath(filepath), str(chrom), start_pos,
        str(ref_allele), str(alt_allele),
    )


##################################
#### Newick Reading Functions ####
##################################

def read_newick(newick_str: str) -> Network:
    """
    Parse a single newick/extended-newick string into a PhyNetPy Network.

    Supports standard newick features (branch lengths, internal node names) 
    as well as the extended newick format for phylogenetic networks 
    (reticulation nodes prefixed with '#', gamma inheritance comments).

    Examples of accepted strings::

        ((A:0.1,B:0.2):0.3,C:0.4);
        ((A:0.1,(B:0.2)#H1:0.3):0.4,(#H1:0.5,C:0.6):0.7);

    Args:
        newick_str (str): A newick or extended-newick string. Trailing 
                          semicolons are handled automatically.

    Raises:
        IOError: If the string cannot be parsed.

    Returns:
        Network: A PhyNetPy Network object with the same topology, names, 
                 and branch lengths as described in the newick string.
    """
    return _core.io_read_newick(newick_str)


####################################
#### GeneTrees Read Helpers ########
####################################

def _ensure_rooted(network: Network) -> Network:
    """
    Ensure a Network representing a tree is rooted (bifurcation at the root).

    If the root has exactly 2 children the tree is already rooted and is
    returned unchanged.  If the root has 3+ children (trifurcation, i.e.
    unrooted representation) the first child subtree is arbitrarily chosen
    as the outgroup: a new root is created with two children -- the
    outgroup and a new internal node parenting the remaining subtrees.

    Args:
        network (Network): A tree-topology Network.

    Returns:
        Network: The (possibly modified) rooted Network.
    """
    root = network.root()
    children = network.get_children(root)

    if len(children) <= 2:
        return network

    warnings.warn(
        f"Tree with root '{root.label}' has {len(children)} children "
        f"(unrooted). Auto-rooting by picking the first child as outgroup."
    )

    outgroup = children[0]
    remaining = children[1:]

    new_root = Node("Root")
    new_internal = Node(f"I_{root.label}")

    child_lengths: Dict[Node, float] = {}
    for child in children:
        edge = network.get_edge(root, child)
        child_lengths[child] = edge.get_length()

    for child in children:
        network.remove_edge(network.get_edge(root, child))
    network.remove_nodes(root)

    network.add_nodes(new_root, new_internal)

    outgroup_edge = Edge(new_root, outgroup, length=child_lengths[outgroup])
    network.add_edges(outgroup_edge)

    network.add_edges(Edge(new_root, new_internal))

    for child in remaining:
        network.add_edges(Edge(new_internal, child, length=child_lengths[child]))

    return network


def _validate_tree_topology(network: Network, label: str = "") -> List[str]:
    """
    Validate that a Network has tree topology suitable for a gene tree.

    Checks:
      - No reticulation nodes (in-degree >= 2).
      - All internal nodes are binary (exactly 2 children).

    Args:
        network (Network): The network to validate.
        label (str): An optional name/index used in warning messages.

    Returns:
        list[str]: Warning messages (empty if topology is valid).
    """
    issues: List[str] = []
    prefix = f"Tree '{label}': " if label else ""

    for node in network.V():
        if node.is_reticulation():
            issues.append(
                f"{prefix}node '{node.label}' is a reticulation node "
                f"(in-degree >= 2). Gene trees must be strict trees."
            )

    for node in network.V():
        n_children = len(network.get_children(node))
        if n_children == 0:
            continue
        if n_children != 2:
            issues.append(
                f"{prefix}internal node '{node.label}' has {n_children} "
                f"children (expected 2 for a binary tree)"
            )

    return issues


def _networks_to_genetrees(
    networks: List[Network],
    species_gene_mapping: Optional[Dict[str, List[str]]] = None,
    naming_rule: Optional[Any] = None) -> GeneTrees:
    """
    Convert a list of Network objects into a GeneTrees container,
    enforcing rooting and validating tree topology along the way.

    Args:
        networks: Parsed Network objects.
        species_gene_mapping: Explicit species -> gene labels dict.
        naming_rule: Callable for deriving species from gene labels.

    Returns:
        GeneTrees: A validated, rooted gene tree collection.
    """
    from .GeneTrees import GeneTrees

    rooted: List[Network] = []
    for i, net in enumerate(networks):
        topology_issues = _validate_tree_topology(net, label=str(i + 1))
        for issue in topology_issues:
            warnings.warn(issue)

        net = _ensure_rooted(net)
        rooted.append(net)

    kwargs: Dict[str, Any] = {"gene_tree_list": rooted}
    if species_gene_mapping is not None:
        kwargs["species_gene_mapping"] = species_gene_mapping
    if naming_rule is not None and species_gene_mapping is None:
        kwargs["naming_rule"] = naming_rule

    gt = GeneTrees(**kwargs)

    if species_gene_mapping is not None:
        mapping_issues = gt.validate_mapping()
        for issue in mapping_issues:
            warnings.warn(f"GeneTrees mapping: {issue}")

    return gt


def _restrict_network_to_taxa(
    net: Network,
    restrict: Sequence[str],
    min_leaves: int,
) -> Optional[Network]:
    """
    Induce a subnetwork containing only leaves whose names appear in *restrict*.

    Args:
        net: Parsed network or tree.
        restrict: Candidate leaf names (typically species or gene labels).
        min_leaves: If fewer than this many names from *restrict* are present
            as leaves in *net*, return None.

    Returns:
        Induced network, or None if too few target leaves are present.
    """
    from .GraphUtils import induced_subnetwork_by_taxa

    present = sorted({name for name in restrict if net.has_node_named(name)})
    if len(present) < min_leaves:
        return None
    return induced_subnetwork_by_taxa(net, present)


####################################
#### Newick Reading Functions ######
####################################

def read_newick_file(
    filepath: Union[str, Path],
    return_type: Literal["networks", "genetrees"] = "networks",
    species_gene_mapping: Optional[Dict[str, List[str]]] = None,
    naming_rule: Optional[Callable[..., Any]] = None,
    *,
    restrict_to_taxa: Optional[Sequence[str]] = None,
    min_leaves_after_restrict: int = 1,
) -> Union[List[Network], GeneTrees]:
    """
    Read a file containing one or more newick strings (one per line) 
    and parse each into a PhyNetPy Network.

    Blank lines and lines starting with '#' are skipped.

    Args:
        filepath: Path to a file containing newick strings.
        return_type (str): ``"networks"`` (default) returns a list of
            Network objects.  ``"genetrees"`` validates each network as
            a rooted binary tree and wraps them in a GeneTrees object.
        species_gene_mapping (dict, optional): Explicit species -> gene
            label mapping.  Only used when *return_type* is
            ``"genetrees"``.
        naming_rule (Callable, optional): Gene-label-to-species callable.
            Only used when *return_type* is ``"genetrees"`` and no
            explicit mapping is given.
        restrict_to_taxa (Sequence[str], optional): If set, each parsed
            network is replaced by the subnetwork induced on those leaf
            labels that appear in both the network and this sequence
            (via :func:`GraphUtils.induced_subnetwork_by_taxa`). Lines
            where fewer than ``min_leaves_after_restrict`` of these
            labels are present are skipped with a warning.
        min_leaves_after_restrict (int): Minimum number of ``restrict_to_taxa``
            labels that must be present on a tree after restriction.
            Ignored when ``restrict_to_taxa`` is None. Default ``1``.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If no valid newick strings are found, or parsing fails.
        ValueError: If ``restrict_to_taxa`` is an empty sequence.

    Returns:
        list[Network] | GeneTrees: Parsed phylogenetic data.
    """
    path = os.fspath(filepath)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Newick file not found: {path}")

    restrict_tuple: Optional[Tuple[str, ...]] = None
    if restrict_to_taxa is not None:
        restrict_tuple = tuple(restrict_to_taxa)
        if not restrict_tuple:
            raise ValueError("restrict_to_taxa must be non-empty when provided")

    networks: List[Network] = []

    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                try:
                    line = convert_newick(line, standard="PhyNetPy")
                    net = read_newick(line)
                    if restrict_tuple is not None:
                        restricted = _restrict_network_to_taxa(
                            net,
                            restrict_tuple,
                            min_leaves_after_restrict,
                        )
                        if restricted is None:
                            warnings.warn(
                                f"Line {line_num}: skipped after restrict_to_taxa "
                                f"(fewer than {min_leaves_after_restrict} "
                                f"matching leaves)."
                            )
                            continue
                        net = restricted
                    networks.append(net)
                except Exception as e:
                    warnings.warn(
                        f"Line {line_num}: Failed to parse newick string "
                        f"'{line[:50]}...': {str(e)}"
                    )
    except OSError as e:
        raise IOError(f"Failed to read newick file '{path}': {str(e)}")

    if not networks:
        raise IOError(
            f"No valid newick strings found in '{path}'."
        )

    if return_type == "genetrees":
        return _networks_to_genetrees(
            networks,
            species_gene_mapping=species_gene_mapping,
            naming_rule=naming_rule,
        )

    return networks


##################################
#### Newick Writing Functions ####
##################################

def write_newick(network: Network, annotate: bool = False) -> str:
    """
    Convert a PhyNetPy Network into a newick string.

    Delegates to the Network's built-in ``newick()`` method, which 
    produces extended-newick notation for networks with reticulation 
    nodes.

    Args:
        network (Network): A PhyNetPy Network object.
        annotate (bool, optional): If True, emit branch lengths (``:len``) 
                                    and inheritance probabilities 
                                    (``[&gamma=g]``) as well as the 
                                    topology. Defaults to False.

    Returns:
        str: The newick representation of the network, ending with ';'.
    """
    return network.newick(annotate)

def write_newick_file(
    networks: List[Network],
    filepath: str) -> None:
    """
    Write one or more Networks to a file as newick strings, one per line.

    Args:
        networks (list[Network]): Networks to write.
        filepath (str): Output file path. Will be created or overwritten.

    Raises:
        IOError: If the list is empty or the file cannot be written.

    Returns:
        None
    """
    _core.io_write_newick_file(networks, os.fspath(filepath))


##################################
#### Nexus Reading Functions #####
##################################

def read_nexus(
    filepath: str,
    validate_input: bool = False,
    print_validation_summary: bool = False,
    return_type: Literal["networks", "genetrees"] = "networks",
    species_gene_mapping: Optional[Dict[str, List[str]]] = None,
    naming_rule: Optional[Callable[..., Any]] = None) -> Union[List[Network], GeneTrees]:
    """
    Read a nexus file and parse all trees/networks in the TREES block 
    into PhyNetPy Network objects.

    This replicates the core functionality of ``NetworkParser`` as a 
    standalone function, making it easy to call without instantiating a 
    class.

    A typical nexus file looks like::

        #NEXUS
        BEGIN TAXA;
            DIMENSIONS NTAX=3;
            TAXALABELS A B C;
        END;
        BEGIN TREES;
            Tree t1 = ((A:0.1,B:0.2):0.3,C:0.4);
            Tree t2 = ((B:0.1,C:0.2):0.3,A:0.4);
        END;

    Args:
        filepath (str): Path to a nexus file (.nex, .nexus).
        validate_input (bool, optional): If True, run NexusValidator on 
            the file before parsing. Defaults to False.
        print_validation_summary (bool, optional): If True and 
            validate_input is True, print the validation summary.  
            Defaults to False.
        return_type (str): ``"networks"`` (default) returns a list of
            Network objects.  ``"genetrees"`` validates each network as
            a rooted binary tree and wraps them in a GeneTrees object.
        species_gene_mapping (dict, optional): Explicit species -> gene
            label mapping. Only used when *return_type* is
            ``"genetrees"``.
        naming_rule (Callable, optional): Gene-label-to-species callable.
            Only used when *return_type* is ``"genetrees"`` and no
            explicit mapping is given.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If the file cannot be parsed or contains no trees.

    Returns:
        list[Network] | GeneTrees: Parsed phylogenetic data.
    """
    filepath = os.fspath(filepath)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Nexus file not found: {filepath}")

    # Optional validation
    validation_summary = None
    if validate_input:
        try:
            from .Validation import NexusValidator
            validator = NexusValidator()
            validation_summary = validator.validate(filepath)
            if print_validation_summary:
                print(validation_summary)
        except Exception as e:
            warnings.warn(f"Validation failed: {str(e)}")

    # Split the file into blocks and pull the TREES block's tree lines
    try:
        tree_lines = _core.io_nexus_tree_lines(filepath)
    except Exception as e:
        traceback.print_exc()
        raise IOError(
            f"NexusReader library could not find or parse '{filepath}': "
            f"{str(e)}"
        )

    if tree_lines is None:
        raise IOError(f"No trees listed in nexus file '{filepath}'.")

    networks: List[Network] = _core.io_networks_from_nexus_trees(tree_lines)

    if not networks:
        raise IOError(
            f"No valid trees could be parsed from nexus file '{filepath}'."
        )

    if return_type == "genetrees":
        return _networks_to_genetrees(
            networks,
            species_gene_mapping=species_gene_mapping,
            naming_rule=naming_rule,
        )

    return networks


def read_nexus_msa(filepath: str) -> MSA:
    """
    Read the sequence data (DATA/CHARACTERS block) from a nexus file 
    and return it as an MSA object.

    This is a convenience wrapper around the MSA constructor's built-in 
    nexus parsing. Use this when you want the alignment data rather than 
    the tree topology.

    Args:
        filepath (str): Path to a nexus file containing a DATA or 
                        CHARACTERS block.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If no sequence data is found.

    Returns:
        MSA: The parsed Multiple Sequence Alignment.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Nexus file not found: {filepath}")

    try:
        return MSA(filename=filepath)
    except Exception as e:
        raise IOError(
            f"Failed to read sequence data from nexus file '{filepath}': "
            f"{str(e)}"
        )


##################################
#### Nexus Writing Functions #####
##################################


def write_nexus(
    networks: List[Network],
    filepath: str,
    taxa: Optional[Set[str]] = None,
    tree_prefix: str = "net",
    overwrite: bool = True,
    phylonet_cmds: Optional[List[str]] = None) -> None:
    """
    Write one or more Networks to a nexus file with TAXA and TREES blocks.

    This replicates the functionality of the ``NexusTemplate`` class 
    as a standalone function.

    The generated file follows the standard nexus format::

        #NEXUS

        BEGIN TAXA;
        DIMENSIONS NTAX=3;
        TAXALABELS
        A
        B
        C
        ;
        END;
        BEGIN TREES;
        Tree net1 = ((A:0.1,B:0.2):0.3,C:0.4);
        Tree net2 = ...;
        END;

    Args:
        networks (list[Network]): The networks to write.
        filepath (str): Output file path. 
        taxa (set[str], optional): An explicit set of taxa labels. If 
                                    None, taxa are inferred from the 
                                    newick strings. Defaults to None.
        tree_prefix (str, optional): Label prefix for each tree line. 
                                      Defaults to "net".
        overwrite (bool, optional): If False, raises IOError if the file 
                                     already exists. Defaults to True.
        phylonet_cmds (list[str], optional): A list of PhyloNet commands 
                                             to include in a PHYLONET 
                                             block. Defaults to None.

    Raises:
        IOError: If the list is empty, or the file cannot be written, 
                 or the file already exists and overwrite is False.

    Returns:
        None
    """
    _core.io_write_nexus(
        networks, os.fspath(filepath), taxa, tree_prefix, overwrite,
        phylonet_cmds,
    )


##########################################
#### Newick Standard Conversion Utilities
##########################################

# Regex pattern for a number in newick (integer, float, or scientific notation)
_NUM_RE = r'\d*\.?\d+(?:[eE][+-]?\d+)?'

# Valid target standards for convert_newick
_VALID_STANDARDS = {"PhyNetPy", "Phylonet", "Beast"}


def detect_newick_standard(newick_str: str) -> str:
    """
    Auto-detect which newick convention a string uses based on its 
    formatting.

    The detection heuristic is:
        1. If the string contains ``#Name:len::gamma`` double-colon 
           notation on a reticulation node → **Phylonet**
        2. If the string starts with ``[&R]`` or ``[&U]`` → **Beast**
        3. If the string contains ``[&...gamma=...]`` → **PhyNetPy**
        4. Otherwise (plain newick) → **PhyNetPy** (default)

    Args:
        newick_str (str): A newick or extended-newick string.

    Returns:
        str: One of ``"Phylonet"``, ``"Beast"``, or ``"PhyNetPy"``.
    """
    return _core.io_detect_newick_standard(newick_str)


def convert_newick(newick_str: str, standard: str = "PhyNetPy") -> str:
    """
    Convert a newick/extended-newick string between different software 
    conventions.

    The three supported standards differ primarily in how they encode 
    inheritance probabilities (gamma) on reticulation edges:

    **PhyNetPy** uses BioPython-style bracket comments::

        ((C:.1,(B:.05)#H0[&gamma=.7]:.05)I1:.1,(A:.1,#H0:.05)I2:.1)I3;

    **Phylonet** uses Rich Newick double-colon notation::

        ((C:.1,(B:.05)#H0:.05::.7)I1:.1,(A:.1,#H0:.05)I2:.1)I3;

    **Beast** uses the same annotation as PhyNetPy but prefixes the 
    string with ``[&R]`` for rooted trees (or ``[&U]`` for unrooted)::

        [&R] ((C:.1,(B:.05)#H0[&gamma=.7]:.05)I1:.1,(A:.1,#H0:.05)I2:.1)I3;

    The function auto-detects the input convention and converts to the 
    target. Non-gamma metadata (e.g. ``[&posterior=0.95]``) on 
    non-reticulation nodes is preserved in all conversions.

    Args:
        newick_str (str): A newick or extended-newick string in any of 
                          the three conventions.
        standard (str, optional): Target convention. One of 
                                   ``"PhyNetPy"`` (default), 
                                   ``"Phylonet"``, or ``"Beast"``.

    Raises:
        ValueError: If ``standard`` is not one of the three valid 
                    options.
        IOError: If the input string is empty.

    Returns:
        str: The newick string reformatted for the target software.
    """
    return _core.io_convert_newick(newick_str, standard)
