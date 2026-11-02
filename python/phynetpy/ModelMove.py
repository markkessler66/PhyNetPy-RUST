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
Last Stable Edit : 4/24/26
First Included in Version : 1.0.0

Network proposal moves for MCMC and simulated-annealing search.

This module defines the proposal kernel primitives used by both the
Bayesian MCMC driver and the MPL simulated-annealing driver
(:mod:`src.MPL`).  Each move subclasses :class:`Move` and implements
``execute`` / ``undo`` / ``same_move`` / ``log_hastings_ratio``.  The kernel
wraps these into a weighted selection scheme; only the move classes
themselves need to understand how to mutate and restore the network.

Module layout
-------------

The file is organised in the following sections (search the banner
headers to jump between them):

    1. RNG HELPER METHODS
         Reproducible random sampling utilities (``_stable_sorted``,
         ``_rng_pick``, ``_rng_pick_weighted``) that every move should
         use instead of the stdlib ``random`` module.

    2. EXCEPTION CLASS
         :class:`MoveError` for fatal proposal failures.

    3. HELPER FUNCTIONS
         Shared graph-surgery utilities (edge splitting / merging,
         random edge sampling) used by multiple move classes.

    4. MOVE CLASS
         Abstract base :class:`Move` describing the contract every
         proposal must satisfy.

    5. TOPOLOGICAL MOVES
         :class:`SPR` (subtree-prune-and-regraft with distance-weighted
         regraft) and small tree-topology moves.

    6. PARAMETER-LEVEL MOVES
         :class:`_suppress_deg2`, :class:`_prune_orphan_chain`,
         :class:`ChangeNodeHeight`, :class:`ChangeInheritanceProb`,
         :class:`ChangeReticSource`, :class:`RelocateReticulation`,
         :class:`ChangeReticDest`, :class:`FlipReticulation`,
         :class:`AddReticulation`, :class:`RemoveReticulation`.

Every move is expected to leave ``model.network`` in a structurally
valid state (single root, no cycles, reticulations with in-degree 2,
no orphan internal nodes).  When a proposal cannot be realised, the
move either returns silently or rolls back via its ``undo_info``
snapshot so the caller can treat the outcome as a no-op.
"""

from __future__ import annotations
import math
from abc import ABC, abstractmethod
import numpy as np
from typing import TYPE_CHECKING, Any, Sequence

# Relative imports
from .Network import Network, Edge, Node
from . import _core
from . import _network_moves as _nm
from .GraphUtils import count_reticulations, level as _network_level, _clone_net

if TYPE_CHECKING:
    from .ModelGraph import Model

# Floor for branch lengths entering the SPR log-Jacobian, so a zero /
# missing length never produces a ``log(0)`` in the Hastings ratio.
_SPR_MIN_LEN: float = 1e-9


############################
#### RNG HELPER METHODS ####
############################
#
# All network moves previously used the stdlib ``random`` module, which pulls
# from a process-wide RNG that is not seeded by our SA/HC drivers.  That made
# proposal content non-reproducible across runs even with identical seeds.
# The helpers below route every random choice through ``model.rng`` (a
# ``numpy.random.Generator``) that is seeded once per search by
# ``MPL.search()`` using ``numpy.random.SeedSequence``.

def _stable_key(item: Any) -> Any:
    """Return a deterministic sort key for a ``Node`` or ``Edge``.

    ``Network`` stores nodes and edges in raw ``set`` containers, whose
    iteration order is hash-driven (i.e. memory-address-driven for
    user-defined classes).  Callers that sample from ``net.V()`` /
    ``net.E()`` therefore see different orderings across processes even
    with an identical seed.  Sorting by label before sampling fixes
    that without changing the graph data structures themselves.

    Falls back to ``id(item)`` for unknown types so we never raise.
    """
    if hasattr(item, "src") and hasattr(item, "dest"):
        src_lbl = getattr(item.src, "label", "") or ""
        dst_lbl = getattr(item.dest, "label", "") or ""
        return (str(src_lbl), str(dst_lbl))
    lbl = getattr(item, "label", None)
    if lbl is not None:
        return str(lbl)
    return id(item)


def _stable_sorted(seq: Sequence[Any]) -> list[Any]:
    """Return ``seq`` sorted by ``_stable_key``; tolerant of empty input."""
    if not seq:
        return []
    return sorted(seq, key=_stable_key)


def _rng_pick(rng: np.random.Generator, seq: Sequence[Any]) -> Any:
    """Uniformly sample one element from ``seq`` using ``rng``.

    ``seq`` is first sorted by ``_stable_key`` so the selection is
    reproducible across processes regardless of how the caller built
    the list.  Returns ``None`` when ``seq`` is empty so callers can
    short-circuit degenerate topologies without extra branching.
    """
    if not seq:
        return None
    ordered = _stable_sorted(seq)
    return ordered[int(rng.integers(0, len(ordered)))]


def _rng_pick_weighted(
    rng: np.random.Generator,
    seq: Sequence[Any],
    weights: Sequence[float],
) -> Any:
    """Weighted sample of one element from ``seq`` using ``rng``.

    Pairs ``seq`` with ``weights`` and sorts the pairs by
    ``_stable_key`` of the element so the draw is reproducible across
    processes.  Weights must be non-negative; they are normalised
    internally.  Returns ``None`` when ``seq`` is empty or the weight
    sum is non-positive.
    """
    if not seq:
        return None
    pairs = sorted(
        zip(seq, weights),
        key=lambda pw: _stable_key(pw[0]),
    )
    ordered_seq = [p for p, _ in pairs]
    w = np.asarray([w for _, w in pairs], dtype=float)
    total = float(w.sum())
    if not np.isfinite(total) or total <= 0.0:
        return ordered_seq[int(rng.integers(0, len(ordered_seq)))]
    probs = w / total
    idx = int(rng.choice(len(ordered_seq), p=probs))
    return ordered_seq[idx]


#########################
#### EXCEPTION CLASS ####
#########################

class MoveError(Exception):
    """
    This exception is raised whenever there is a fatal error in executing a 
    network move.
    """
    def __init__(self, message: str = "Error executing a move") -> None:
        """
        Initializes a MoveError object.

        Args:
            message (str, optional): Custom error message. Defaults to
                                     "Error executing a move".
        Returns:
            N/A
        """
        self.message = message
        super().__init__(self.message)

##########################
#### HELPER FUNCTIONS ####
##########################

def insert_node_in_edge(edge: Edge, node: Node, net: Network) -> None:
    """
    Given an edge, a -> b, place a node c, such that a -> c -> b.
    This requires the deletion of edge a -> b, then the addition of edges
    a -> c and c -> b.

    Gamma preservation.  When ``b`` is a reticulation, the original
    edge ``a -> b`` carries an inheritance probability ``gamma`` that
    represents one component of ``b``'s parental-mass split.  After
    the insertion the new in-edge to ``b`` is ``c -> b`` (``a -> c``
    is just a fresh tree edge feeding the new internal node), so the
    original gamma is propagated to ``c -> b``.

    Args:
        edge (Edge): An edge, a -> b
        node (Node): A node, c.
        net (Network): The network that contains nodes a, b, and c
    Returns:
        N/A
    """
    _core.insert_node_in_edge(edge, node, net)


def _suppress_deg2(net: Network, node: Node) -> None:
    """Collapse a degree-2 passthrough node into a single edge.

    ``node`` must have in-degree 1 and out-degree 1; if not, this is
    a no-op.  The two incident edges are merged into one whose
    branch length is the sum of the originals.  Gamma is inherited
    from the child-side edge when present, falling back to the
    parent-side edge.

    Args:
        net: Network containing ``node``.
        node: Candidate passthrough node.
    """
    _core._suppress_deg2(net, node)


def _prune_orphan_chain(net: Network, node: Node) -> None:
    """Iteratively remove non-root dead-end internal nodes.

    Walks up from ``node``, removing each dead-end and its incoming
    edges and collapsing any parents that become degree-2
    passthroughs.  The root is never pruned.

    Args:
        net: Network to repair.
        node: Starting dead-end node.
    """
    _core._prune_orphan_chain(net, node)


##################
#### MOVE API ####
##################

class Move(ABC):
    """Abstract base class for every network proposal move.

    A :class:`Move` is a stateful object whose ``execute`` method
    mutates the ``Network`` held by a :class:`Model` and whose
    ``undo`` method restores the pre-execute state.  The contract
    is:

      * ``execute(model)`` performs the proposal and records just
        enough information to roll it back: either compact deltas in
        ``undo_info`` (continuous moves), or, for topology moves, an
        open ``Network`` transaction (``net.begin_transaction()``) whose
        journal undoes every change on ``rollback_transaction()``. No
        move copies the network.
      * ``undo(model)`` returns the network to its pre-execute
        state.  Must be cheap and side-effect-free on anything
        else.
      * ``same_move(model)`` re-plays the proposal on a clone of
        the model, used by the parallel MCMC driver.
      * ``log_hastings_ratio()`` returns ``log[ q(x | x') / q(x' | x) ]``
        (in log space, ``0.0`` for symmetric moves) plus any
        reversible-jump log-Jacobian; see the method-level
        docstring below.

    Subclasses are expected to leave the network in a
    structurally valid state (single root, acyclic, reticulations
    with in-degree 2, no orphan internals).  When a proposal is
    infeasible (e.g. no reticulations present for a retic-flip),
    the subclass should return silently rather than raise.
    """

    def __init__(self) -> None:
        """Initialise per-instance bookkeeping fields."""
        self.model = None
        self.undo_info = None
        self.same_move_info = None
        # Log proposal-asymmetry correction for this proposal, computed
        # in ``execute`` (where the move has access to both the pre- and
        # post-move network) and returned by :meth:`log_hastings_ratio`.
        # Defaults to ``0.0`` (symmetric proposal / no-op).
        self._log_hr: float = 0.0

    @abstractmethod
    def execute(self, model: Model) -> Model:
        """Apply the proposal to ``model.network`` in place.

        Args:
            model: Model whose ``network`` will be mutated.  The
                move uses ``model.rng`` (a seeded
                ``numpy.random.Generator``) for every random
                decision.

        Returns:
            The same ``model`` for chaining convenience.  The
            underlying network is mutated in place; no copy is
            returned.
        """
        pass

    @abstractmethod
    def undo(self, model: Model) -> None:
        """Undo the most recent call to :meth:`execute`.

        Args:
            model: The same model that was passed to
                :meth:`execute`.  After this call, the network
                is restored to its pre-execute state.
        """
        pass

    @abstractmethod
    def same_move(self, model: Model) -> None:
        """Replay the last proposal on an identical (cloned) model.

        Used by the parallel MCMC driver, which runs multiple
        chains with shared move sequences.  Concrete moves that
        use deep-copy ``undo_info`` typically implement this as a
        no-op, since replay is not meaningful without the exact
        random draws.

        Args:
            model: A topologically-equivalent clone of the model
                passed to :meth:`execute`.
        """
        pass
    
    def touched_nodes(self, net: Network) -> "set[Node] | None":
        """Network nodes whose incident branches or gammas this move modified.

        Consumed by scorers that maintain per-network-edge likelihood
        caches (see :class:`phynetpy._mcmc_gt._GTLikelihoodEngine`):
        any node returned here -- plus its ancestor path to the root
        -- will have its cached partials invalidated, everything else
        is reused across iterations.

        The default implementation returns ``None`` (fully dirty),
        which is the safe fallback for moves whose impact on the
        network structure is hard to localise.  Topology-preserving
        moves (``ChangeNodeHeight``, ``ChangeInheritanceProb``) should
        override and return the single node whose incident edges
        changed.

        Topology-changing moves (``SPR``, ``AddReticulation``,
        ``RemoveReticulation``, ``RelocateReticulation``,
        ``ChangeReticSource``, ``ChangeReticDest``,
        ``FlipReticulation``) generally keep the default ``None``:
        they replace edges entirely, which the engine's per-edge
        cache has to rebuild anyway.

        Args:
            net: Current network (i.e. the network after
                :meth:`execute`).  Overrides may use it to resolve
                stored labels to live :class:`Node` references.

        Returns:
            ``None`` for "fully dirty" (the engine rebuilds every
            cached partial), or a possibly-empty ``set[Node]`` of
            touched network nodes.
        """
        return None

    def _revert_if_exceeds_level(self, model: Model) -> bool:
        """Self-reject the proposal when it pushes the network past ``max_level``.

        Efficiency layer for the ``max_lvl`` search flag.  A move that can
        *raise* network level (add / relocate / endpoint reticulation moves)
        calls this right before finalising ``execute``.  If the resulting
        network's level exceeds the move's ``_max_level`` cap, the move's
        open transaction is rolled back and the proposal becomes a no-op
        (zero Hastings).

        This is purely an optimisation: it avoids wasting a likelihood
        evaluation on a proposal the accept-path level guard
        (:func:`phynetpy._search_flags.make_level_validator`) would reject
        anyway.  Moves that never set ``_max_level`` are unaffected.

        Args:
            model: Model whose ``network`` was just mutated by ``execute``.

        Returns:
            ``True`` when the proposal was reverted (caller should return
            immediately), ``False`` otherwise.
        """
        max_level = getattr(self, "_max_level", None)
        if max_level is None:
            return False
        net = model.network
        if net is None or _network_level(net) <= max_level:
            return False
        if net.in_transaction():
            net.rollback_transaction()
        self.undo_info = None
        self._log_hr = 0.0
        model.update_network()
        return True

    def log_hastings_ratio(self) -> float:
        r"""Log proposal-asymmetry correction ``log[ q(x|x') / q(x'|x) ]``.

        A proper MCMC sampler corrects for asymmetric proposal
        distributions via the Hastings ratio so the chain converges to
        the target distribution.  This method returns that ratio **in log
        space**, which is the convention used throughout PhyNetPy's
        samplers (see :mod:`phynetpy._network_moves` and
        :mod:`phynetpy._mcmc_seq`): the acceptance rule is

            log_alpha = (logpi' - logpi) + log_hastings_ratio (+ log|J|)

        so a strictly symmetric move returns ``0.0`` (not ``1.0``).  The
        previous interface returned a *linear* ratio and every move
        returned ``1.0``; consumed additively in log space that silently
        inflated acceptance by a factor of ``e`` and left genuinely
        asymmetric moves uncorrected.

        Three moves in this module are not symmetric and supply real
        corrections (delegated to :mod:`phynetpy._network_moves` where the
        math is shared with the sequence sampler):

        * :class:`SPR` weights candidate regraft edges by
          ``1 / d ** distance_decay``; the correction accounts for the
          forward / reverse regraft-weight asymmetry.
        * :class:`ChangeNodeHeight` draws a Gaussian whose sigma scales
          with the feasible half-range, which differs pre/post-move.
        * :class:`ChangeInheritanceProb` uses a Gaussian on ``gamma``;
          the correction is the truncated-normal normaliser ratio
          ``log[ Z(old) / Z(new) ]``.

        Dimension-changing moves (:class:`AddReticulation` /
        :class:`RemoveReticulation`) additionally fold the reversible-jump
        Jacobian into this return value, mirroring
        :func:`phynetpy._mcmc_seq.op_add_reticulation`.

        Note: the optimisation drivers (:class:`HillClimbing`,
        :class:`SimulatedAnnealing`) ignore this term -- they climb a
        score surface rather than sample a stationary distribution -- so
        for MPL search the exact value is immaterial.

        Returns:
            float: ``log[ q(x|x') / q(x'|x) ]``.  ``0.0`` for symmetric
            moves; a finite real (possibly including a log-Jacobian) for
            asymmetric / dimension-changing moves.  The value is computed
            in :meth:`execute` and cached on ``self._log_hr``.
        """
        return float(getattr(self, "_log_hr", 0.0))


####GRAPH MOVES####
r"""
ALL OF THE FOLLOWING NETWORK MOVES HAVE VARIABLE NAMES THAT ARE BASED OFF OF THIS BASIC NETWORK STRUCTURE:

                  a
                    \
                     \  
                      \
            x          z  
           / \        / \
          /   \      /   \
         /     \    /     \
        /        c         \
       /         |          \
      /          |           \
                 y            b

Every move below delegates its graph surgery and random draws to the
compiled core (``phynetpy._core.mv_*``), which mirrors the reference
algorithm step for step -- same candidate orderings, same sequence of
draws from ``model.rng`` -- so seeded searches are reproducible.
"""


class _TransactionalMove(Move):
    """Shared undo/replay for moves that edit inside a network transaction."""

    def undo(self, model: Model) -> None:
        """Roll back the transaction opened by ``execute``."""
        model.network.rollback_transaction()
        self.undo_info = None
        model.update_network()

    def same_move(self, model: Model) -> None:
        """No-op: replay is not supported for transactional moves."""
        pass


class AddReticulation(_TransactionalMove):
    """
    A move that adds a reticulation to a network.

    Host edges are split at a random point so their original branch
    length is preserved across the two halves.  The new reticulation
    edge is given a short length drawn from Exp(mean=0.1*min_host_len)
    and both parent edges of the new reticulation node receive
    gamma = 0.5.  The reversible-jump log-Hastings ratio is computed
    from the pre-move edge and reticulation counts.
    """

    def __init__(
        self,
        max_reticulations: int | None = None,
        max_level: int | None = None,
    ) -> None:
        """Create a new ``AddReticulation`` proposal.

        Args:
            max_reticulations: Optional hard cap on the number of
                reticulations.  If the current network already has
                ``max_reticulations`` reticulation nodes, the
                proposal silently returns without modifying the
                network.  ``None`` disables the cap.
            max_level: Optional cap on network level (``max_lvl``
                search flag).  If the addition would push the network
                level above this, the move self-reverts to a no-op.
                ``None`` disables the per-move check.
        """
        super().__init__()
        self._max_retics = max_reticulations
        self._max_level = max_level

    def execute(self, model: Model) -> Model:
        """Insert a new reticulation into ``model.network``."""
        self.undo_info = None
        self._log_hr, update = _core.mv_add_reticulation(
            model.network, model.rng, self._max_retics, self._max_level,
        )
        if update:
            model.update_network()
        return model


class RemoveReticulation(_TransactionalMove):
    """Remove a reticulation node from the network.

    Algorithm:
      1. Pick a random reticulation node c (in=2, out=1).
      2. Randomly choose one of its two parent edges to delete.
      3. Remove that parent edge.  c now has (in=1, out=1) -- suppress c.
      4. The source of the deleted edge may now be degree-2 -- suppress it.
    """

    def __init__(self) -> None:
        """Create a new ``RemoveReticulation`` proposal."""
        super().__init__()

    def execute(self, model: Model) -> Model:
        """Delete a random reticulation from ``model.network``."""
        self.undo_info = None
        self._log_hr, update = _core.mv_remove_reticulation(model.network, model.rng)
        if update:
            model.update_network()
        return model

    @staticmethod
    def _merged_length_if_deg2(net: Network, node: Node) -> float | None:
        """Summed length of a degree-2 node's two incident edges, else None."""
        if net.in_degree(node) != 1 or net.out_degree(node) != 1:
            return None
        in_e = list(net.in_edges(node))[0]
        out_e = list(net.out_edges(node))[0]
        return (in_e.get_length() or 0.0) + (out_e.get_length() or 0.0)


class FlipReticulation(_TransactionalMove):
    """Flip the direction of one reticulation edge.

    Picks a random reticulation edge ``z -> c`` and reverses it to
    ``c -> z``, provided the result is still a valid DAG.  After the
    flip, ``z`` becomes a reticulation (in-degree 2) and ``c`` may
    lose its reticulation status.  If the post-flip network is cyclic
    the move rolls itself back.
    """

    def __init__(self, max_level: int | None = None) -> None:
        """Create a new ``FlipReticulation`` proposal.

        Args:
            max_level: Optional cap on network level (``max_lvl`` search
                flag).  If the flip raises the level above this, the move
                self-reverts to a no-op.  ``None`` disables the check.
        """
        super().__init__()
        self._max_level = max_level

    def execute(self, model: Model) -> Model:
        """Propose a single reticulation-edge flip on ``model.network``."""
        self.undo_info = None
        self._log_hr, update = _core.mv_flip_reticulation(
            model.network, model.rng, self._max_level,
        )
        if update:
            model.update_network()
        return model


class SwitchParentage(_TransactionalMove):
    """
    PSPP (Parentage-Switching for Polyploid Phylogenetics) move for 
    Infer_MP_Allop. Alters the genetic parentage of a subnetwork while 
    maintaining the same ploidy values for each leaf.  When the
    reattachment cannot restore every leaf's ploidy the proposal reverts
    to a no-op rather than emit an invalid network.
    """
    def __init__(self, debug_id: int = 0) -> None:
        """Create a new ``SwitchParentage`` (PSPP) proposal.

        Args:
            debug_id: Deprecated and ignored; retained for constructor
                compatibility with existing callers.
        """
        super().__init__()
        self.valid_attachment_edges: list[Edge] = []

    def _random_choice(self, mylist: list, rng: np.random.Generator) -> Any:
        """Select a random element from a list, or None if empty."""
        if not mylist:
            return None
        return mylist[int(rng.integers(0, len(mylist)))]

    def execute(self, model: Model) -> Model:
        """Executes the PSPP (Switch-Parentage) move."""
        self.undo_info = None
        self.valid_attachment_edges = []
        _core.mv_switch_parentage(model.network, model.rng)
        model.update_network()
        self.same_move_info = None
        return model

    def same_move(self, model: Model) -> None:
        """Replays the same topology change on a different model instance."""
        if self.same_move_info is not None:
            model.network = _clone_net(self.same_move_info)
        model.update_network()


class SPR(_TransactionalMove):
    r"""Subtree Prune and Regraft on a phylogenetic network.

    Prunes a random eligible subtree edge ``u -> v`` (``u`` a plain
    internal node with in-degree 1 and out-degree 2, ``v`` not a
    reticulation), suppresses ``u``, then regrafts ``v`` by subdividing a
    uniformly chosen eligible edge ``a -> b`` at ``split ~ U(0, 1)`` and
    attaching ``w -> v``.  The exact log Hastings ratio is

    .. math::

        \log H = \log|P(N)| - \log|P(N')| + \log L_t - \log \ell_{e^*}

    with ``|P(N)|`` / ``|P(N')|`` the prunable-edge counts before / after,
    ``L_t`` the pre-split length of the subdivided target edge and
    ``\ell_{e^*}`` the length of the edge created by suppressing ``u``.

    ``distance_decay`` is retained only for backward-compatible
    construction and is ignored.
    """

    _DEFAULT_DISTANCE_DECAY = 2.0

    def __init__(self,
                 distance_decay: float | None = None,
                 debug_id: int = 0) -> None:
        """Create a new ``SPR`` proposal.

        Args:
            distance_decay: Deprecated and ignored; retained only for
                backward-compatible construction (see class docstring).
            debug_id: Unused; retained for constructor-signature
                compatibility with other moves.
        """
        super().__init__()
        self.undo_info = None
        self.same_move_info = None
        self._distance_decay = (
            self._DEFAULT_DISTANCE_DECAY
            if distance_decay is None
            else float(distance_decay)
        )

    @staticmethod
    def _prunable_edges(net: Network) -> list[Edge]:
        """Edges ``u -> v`` this move may prune (self-reversible subset)."""
        prunable: list[Edge] = []
        for e in net.E():
            u, v = e.src, e.dest
            if v.is_reticulation():
                continue
            if net.in_degree(u) != 1 or net.out_degree(u) != 2:
                continue
            prunable.append(e)
        return prunable

    def execute(self, model: Model) -> Model:
        """Propose a subtree-prune-and-regraft move on ``model.network``."""
        self.undo_info = None
        self._log_hr, update = _core.mv_spr(model.network, model.rng)
        if update:
            model.update_network()
        return model


################################
#### PARAMETER-LEVEL MOVES #####
################################


class ChangeNodeHeight(Move):
    """Slide an internal node up or down by adjusting incident branch lengths.

    A random internal (non-root, non-leaf) node is chosen and shifted by
    a truncated-Gaussian delta whose standard deviation is ``sigma_frac``
    times the feasible half-range.  Leaf branches are never modified (they
    are non-identifiable under the MSC).
    """

    _EPSILON = 1e-6
    _DEFAULT_SIGMA_FRAC = 0.4
    _TRUNCATE_RETRIES = 3

    def __init__(self, sigma_frac: float | None = None) -> None:
        """Create a new ``ChangeNodeHeight`` proposal.

        Args:
            sigma_frac: Fraction of the feasible half-range used as the
                standard deviation of the truncated-Gaussian slide.
                Defaults to :attr:`_DEFAULT_SIGMA_FRAC` when ``None``.
        """
        super().__init__()
        self._sigma_frac = (
            self._DEFAULT_SIGMA_FRAC if sigma_frac is None else float(sigma_frac)
        )

    @staticmethod
    def _is_leaf_edge(net: Network, edge: Edge) -> bool:
        """True when ``edge`` terminates at a leaf (tip) node."""
        return net.out_degree(edge.dest) == 0

    def execute(self, model: Model) -> Model:
        """Slide a random internal node's height by a truncated-Gaussian delta."""
        self._log_hr, touched, changes = _core.mv_change_node_height(
            model.network, model.rng, self._sigma_frac,
        )
        if touched is None:
            model.update_network()
            return model
        self.undo_info = changes
        self.same_move_info = changes
        self._touched_label = touched
        model.mark_touched(self.touched_nodes(model.network))
        return model

    def undo(self, model: Model) -> None:
        """Restore every edge length changed in ``execute`` to its old value."""
        net: Network = model.network
        if self.undo_info is not None:
            for src_lbl, dest_lbl, old_len, _new_len in self.undo_info:
                src = net.has_node_named(src_lbl)
                dest = net.has_node_named(dest_lbl)
                if src is not None and dest is not None:
                    net.get_edge(src, dest).set_length(old_len)
        model.mark_touched(self.touched_nodes(model.network))

    def same_move(self, model: Model) -> None:
        """Re-apply the same edge-length deltas to a cloned ``model``."""
        net: Network = model.network
        if self.same_move_info is not None:
            for src_lbl, dest_lbl, _old_len, new_len in self.same_move_info:
                src = net.has_node_named(src_lbl)
                dest = net.has_node_named(dest_lbl)
                if src is not None and dest is not None:
                    net.get_edge(src, dest).set_length(new_len)
        model.mark_touched(self.touched_nodes(model.network))

    def touched_nodes(self, net: Network) -> "set[Node] | None":
        """Return the single node whose height changed."""
        label = getattr(self, "_touched_label", None)
        if label is None:
            return None
        node = net.has_node_named(label)
        if node is None:
            return None
        return {node}


class ChangeInheritanceProb(Move):
    """Propose a new inheritance probability for a random reticulation node.

    The two parent edges of a reticulation always carry complementary
    gammas (gamma and 1-gamma).  The new value is drawn from a
    Gaussian centered on the current gamma, truncated to (epsilon,
    1-epsilon).
    """

    _SIGMA = 0.1
    _EPS = 0.01

    def __init__(self, sigma: float | None = None) -> None:
        """Create a new ``ChangeInheritanceProb`` proposal.

        Args:
            sigma: Standard deviation of the truncated-Gaussian random
                walk on gamma. Defaults to :attr:`_SIGMA` when ``None``.
        """
        super().__init__()
        if sigma is not None:
            self._sigma = sigma
        else:
            self._sigma = self._SIGMA

    def execute(self, model: Model) -> Model:
        """Perturb the gamma of a random reticulation's two parent edges."""
        self._log_hr, touched, undo, same = _core.mv_change_inheritance_prob(
            model.network, model.rng, self._sigma,
        )
        if touched is None:
            model.update_network()
            return model
        self.undo_info = undo
        self.same_move_info = same
        self._touched_label = touched
        model.mark_touched(self.touched_nodes(model.network))
        return model

    def undo(self, model: Model) -> None:
        """Restore both parent edges' gammas to their pre-``execute`` values."""
        net: Network = model.network
        if self.undo_info is not None:
            s1, d1, g1, s2, d2, g2 = self.undo_info
            n_s1, n_d1 = net.has_node_named(s1), net.has_node_named(d1)
            n_s2, n_d2 = net.has_node_named(s2), net.has_node_named(d2)
            if n_s1 is not None and n_d1 is not None:
                net.get_edge(n_s1, n_d1).set_gamma(g1)
            if n_s2 is not None and n_d2 is not None:
                net.get_edge(n_s2, n_d2).set_gamma(g2)
        model.mark_touched(self.touched_nodes(model.network))

    def same_move(self, model: Model) -> None:
        """Re-apply the same gamma change to a cloned ``model``."""
        net: Network = model.network
        if self.same_move_info is not None:
            s1, d1, g1, s2, d2, g2 = self.same_move_info
            n_s1, n_d1 = net.has_node_named(s1), net.has_node_named(d1)
            n_s2, n_d2 = net.has_node_named(s2), net.has_node_named(d2)
            if n_s1 is not None and n_d1 is not None:
                net.get_edge(n_s1, n_d1).set_gamma(g1)
            if n_s2 is not None and n_d2 is not None:
                net.get_edge(n_s2, n_d2).set_gamma(g2)
        model.mark_touched(self.touched_nodes(model.network))

    def touched_nodes(self, net: Network) -> "set[Node] | None":
        """Return the single reticulation whose inheritance prob changed."""
        label = getattr(self, "_touched_label", None)
        if label is None:
            return None
        node = net.has_node_named(label)
        if node is None:
            return None
        return {node}


class ChangeReticSource(_TransactionalMove):
    """Move the source (tail / parent) end of a reticulation edge.

    Picks a random reticulation edge ``z -> c``, detaches the source
    end, cleans up anything ``z`` leaves behind (deg-2 collapse or
    orphan-chain pruning), then reattaches ``c``'s parent from a
    freshly-inserted node split into a randomly chosen host edge.
    Cycle-safe: host edges downstream of ``c`` are filtered out.
    """

    def __init__(self, max_level: int | None = None) -> None:
        """Create a new ``ChangeReticSource`` proposal.

        Args:
            max_level: Optional cap on network level (``max_lvl`` search
                flag).  ``None`` disables the check.
        """
        super().__init__()
        self._max_level = max_level

    def execute(self, model: Model) -> Model:
        """Propose a source-relocation for one reticulation edge."""
        self.undo_info = None
        self._log_hr, update = _core.mv_change_retic_source(
            model.network, model.rng, self._max_level,
        )
        if update:
            model.update_network()
        return model


class RelocateReticulation(_TransactionalMove):
    """Atomically relocate a reticulation node end-to-end.

    Picks a random reticulation node, detaches both incoming edges
    and the single outgoing edge, cleans up any passthroughs or
    orphan chains left behind, and then reattaches a fresh
    reticulation on two newly split host edges.  Reports a
    log-Hastings ratio of ``0.0`` (an optimisation move).
    """

    def __init__(self, max_level: int | None = None) -> None:
        """Create a new ``RelocateReticulation`` proposal.

        Args:
            max_level: Optional cap on network level (``max_lvl`` search
                flag).  ``None`` disables the check.
        """
        super().__init__()
        self._max_level = max_level

    def execute(self, model: Model) -> Model:
        """Propose a whole-reticulation relocation."""
        self.undo_info = None
        self._log_hr, update = _core.mv_relocate_reticulation(
            model.network, model.rng, self._max_level,
        )
        if update:
            model.update_network()
        return model


class ChangeReticDest(_TransactionalMove):
    """Move the destination (head) end of a reticulation edge.

    Picks a random reticulation edge ``z -> c``, detaches the
    destination end, clears ``c``'s reticulation flag and collapses
    it if it became a deg-2 passthrough, then reattaches into a
    freshly-inserted reticulation node split onto a randomly chosen
    host edge.  Cycle-safe: edges upstream of ``z`` are filtered out.
    """

    def __init__(self, max_level: int | None = None) -> None:
        """Create a new ``ChangeReticDest`` proposal.

        Args:
            max_level: Optional cap on network level (``max_lvl`` search
                flag).  ``None`` disables the check.
        """
        super().__init__()
        self._max_level = max_level

    def execute(self, model: Model) -> Model:
        """Propose a destination-relocation for one reticulation edge."""
        self.undo_info = None
        self._log_hr, update = _core.mv_change_retic_dest(
            model.network, model.rng, self._max_level,
        )
        if update:
            model.update_network()
        return model
