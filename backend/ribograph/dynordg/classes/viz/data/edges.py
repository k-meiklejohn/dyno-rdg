from typing import Literal
from dataclasses import dataclass, field
from ...core import RiboNode
# ── Type aliases ─────────────────────────────────────────────────────────────

Pt    = tuple[float, float]
Edge  = tuple[RiboNode, RiboNode]


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 output  –  EdgeSpec
# ─────────────────────────────────────────────────────────────────────────────

EdgeType = Literal[
    'shift', '40s_retention', 'drop', 'initiation', 'load',
    '0', '1', '2', '3',         # horizontal / phase-carry edges
]

@dataclass
class EdgeSpec:
    """Everything known about an edge *before* any coordinates are assigned."""
    u:          RiboNode
    v:          RiboNode
    flux_start: float
    flux_end:   float
    etype:      EdgeType
    direction:  int | None   # +1 / -1 / 0 / None (bulk, resolved later)
    shift_n:    int  = 0     # codon displacement for 'shift' edges
    is_event:   bool = False  # True when u.phase != v.phase


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 output  –  NodeLayout
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EdgeSlot:
    """One entry in an ordered slot list for a node side."""
    edge:      Edge
    direction: int   # resolved direction at this node


@dataclass
class NodeLayout:
    """Ordered in/out edge slots for one node, ready for geometry."""
    node:        RiboNode
    in_slots:    list[EdgeSlot]  = field(default_factory=list)
    out_slots:   list[EdgeSlot]  = field(default_factory=list)
    drop_direction: int | None   = None   # set when node has a drop edge


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 output  –  EdgeGeom
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EdgeGeom:
    """
    All coordinate data for one edge, in *local node-relative* y space
    after Phase 3, and in *world* space after Phase 4.
    """
    edge: Edge

    # Primary rectangle corners
    in0:  Pt | None = None
    in1:  Pt | None = None
    out0: Pt | None = None
    out1: Pt | None = None

    # Curve taper quadrant hints
    in_quad:  int | None = None
    out_quad: int | None = None

    # Decay triangle
    decay0: Pt | None = None
    decay1: Pt | None = None
    decay2: Pt | None = None

    # Alignment anchors (used only during Phase 4, stripped after)
    in_bot:  Pt | None = None
    out_bot: Pt | None = None

    # Helper rectangles for stacked bulk offsets
    helper_rects: list[list[Pt]] = field(default_factory=list)

    # Carried from EdgeSpec for renderer use
    etype:     EdgeType  = '0'
    is_event:  bool      = False
    flux_end:  float     = 0.0