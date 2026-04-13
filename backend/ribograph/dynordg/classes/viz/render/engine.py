from ...core import RiboNode
from ...graph import RiboGraph
from ..data import LayoutResult, EdgeSpec, Edge, EdgeSlot, EdgeGeom, EdgeType, NodeLayout, Pt
from networkx import topological_sort
import math
from typing import Literal

_IN_EDGE_ORDER: dict[tuple, int] = {
    ('shift',         +1, -1): 0,
    ('initiation',    +1):     1,
    ('shift',         +1, +1): 2,
    ('load',          +1):     3,
    # 4 reserved for direction == 0
    ('load',          -1):     5,
    ('shift',         -1, +1): 6,
    ('40s_retention', -1):     7,
    ('shift',         -1, -1): 8,
}

_OUT_EDGE_ORDER: dict[tuple, int] = {
    ('shift',         -1, -1): 0,
    ('40s_retention', -1):     1,
    ('shift',         -1, +1): 2,
    ('drop',          -1):     3,
    # 4 reserved for direction == 0
    ('drop',          +1):     5,
    ('shift',         +1, +1): 6,
    ('initiation',    +1):     7,
    ('shift',         +1, -1): 8,
}


# ─────────────────────────────────────────────────────────────────────────────
# Edge-order lookup tables
# ─────────────────────────────────────────────────────────────────────────────




def _sort_key(spec: EdgeSpec, direction: int, order: dict) -> tuple[int, int]:
    if direction == 0:
        return (4, 0)
    if spec.etype == 'shift':
        priority = order.get(('shift', direction, spec.shift_n), 99)
    else:
        priority = order.get((spec.etype, direction), 99)
    assert priority != 99, \
        f'Unknown edge type/direction combo: {spec.etype!r}, {direction!r}'
    return (priority, spec.shift_n)


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate helpers
# ─────────────────────────────────────────────────────────────────────────────

def _shift_pt(pt: Pt, delta: float, axis: Literal['x', 'y']) -> Pt:
    x, y = pt
    return (x + delta, y) if axis == 'x' else (x, y + delta)


def _shift_geom(geom: EdgeGeom, delta: float, axis: Literal['x', 'y'],
                keys: list[str]) -> None:
    """Shift named point attributes of *geom* in-place."""
    for k in keys:
        pt = getattr(geom, k)
        if pt is not None:
            setattr(geom, k, _shift_pt(pt, delta, axis))
    geom.helper_rects = [
        [_shift_pt(pt, delta, axis) for pt in rect]
        for rect in geom.helper_rects
    ]

class LayoutEngine:
    """
    Stateless layout engine.  Feed it a graph, get back a LayoutResult.

    Each phase method is independently unit-testable and can be overridden
    in a subclass to change layout behaviour without touching other phases.
    """

    def __init__(self, log_scale: float = 1):
        self.log_scale = log_scale

    # ── Entry point ──────────────────────────────────────────────────────────

    def run(self, graph: RiboGraph) -> LayoutResult:
        node_x = self._node_x_positions(graph)

        specs     = self.classify_edges(graph)
        layouts   = self.order_nodes(graph, specs)
        geoms     = self.compute_geometries(graph, specs, layouts, node_x)
        result    = self.align_layout(graph, geoms, layouts, node_x)
        return result

    # ── Pre-pass: node x positions ───────────────────────────────────────────

    def _node_x_positions(self, graph: RiboGraph) -> dict[RiboNode, float]:
        raw = {n: n.position for n in graph.nodes}
        if self.log_scale <= 1:
            return raw

        xs      = sorted(set(raw.values()))
        gaps    = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
        log_g   = [math.log(g, self.log_scale) + 1 for g in gaps]
        log_xs  = [xs[0]]
        for g in log_g:
            log_xs.append(log_xs[-1] + g)
        log_map = dict(zip(xs, log_xs))
        return {n: log_map[raw[n]] for n in graph.nodes}

    # ── Phase 1: classify ────────────────────────────────────────────────────

    def classify_edges(self, graph: RiboGraph) -> dict[Edge, EdgeSpec]:
        """
        Assign etype, direction, and shift_n to every edge.
        Returns a plain dict — the graph is not mutated.
        """
        specs: dict[Edge, EdgeSpec] = {}

        for u, v, data in graph.edges(data=True):
            specs[(u, v)] = self._classify_one(u, v, data)

        # Resolve bulk (load/drop) directions now that all non-bulk are known
        self._resolve_bulk_directions(graph, specs)

        return specs

    def _classify_one(self, u: RiboNode, v: RiboNode,
                      data: dict) -> EdgeSpec:
        is_event = u.phase != v.phase

        if not is_event:
            return EdgeSpec(
                u=u, v=v,
                flux_start=data['flux_start'], flux_end=data['flux_end'],
                etype=str(u.phase), direction=0,
                is_event=False,
            )

        if u.phase == -1:
            etype = 'load'
        elif v.phase == -1:
            etype = 'drop'
        elif u.phase == 0 and v.phase > 0:
            etype = 'initiation'
        elif u.phase > 0 and v.phase == 0:
            etype = '40s_retention'
        else:
            etype = 'shift'

        shift_n   = (v.position - u.position) if etype == 'shift' else 0
        direction = (
            None if (u.phase == -1 or v.phase == -1)
            else int((v.phase - u.phase) / abs(v.phase - u.phase))
        )

        return EdgeSpec(
            u=u, v=v,
            flux_start=data['flux_start'], flux_end=data['flux_end'],
            etype=etype, direction=direction,
            shift_n=shift_n, is_event=True,
        )

    def _resolve_bulk_directions(self, graph: RiboGraph,
                                  specs: dict[Edge, EdgeSpec]) -> None:
        """Fill in direction=None on load/drop EdgeSpecs."""
        for node in graph.nodes:
            if node.phase == -1:
                continue
            bulk_edges = self._bulk_edges_at(node, graph)
            for bulk_node, direction in bulk_edges.items():
                # load: bulk → node
                key = (bulk_node, node)
                if key in specs and specs[key].direction is None:
                    total = sum(
                        s.direction or 0
                        for (u, v), s in specs.items()
                        if v == node and u.phase != -1
                    )
                    specs[key].direction = -1 if total >= 0 else 1

                # drop: node → bulk
                key = (node, bulk_node)
                if key in specs and specs[key].direction is None:
                    total_out = sum(
                        s.direction or 0
                        for (u, v), s in specs.items()
                        if u == node and v.phase != -1
                    )
                    total_in = sum(
                        s.direction or 0
                        for (u, v), s in specs.items()
                        if v == node and u.phase != -1
                    )
                    specs[key].direction = -1 if (total_out - total_in) > 0 else 1

    def _bulk_edges_at(self, node: RiboNode,
                       graph: RiboGraph) -> dict[RiboNode, int]:
        return {
            **{v: 1  for u, v in graph.edges if u == node and v.phase == -1},
            **{u: -1 for u, v in graph.edges if v == node and u.phase == -1},
        }

    # ── Phase 2: order ───────────────────────────────────────────────────────

    def order_nodes(self, graph: RiboGraph,
                    specs: dict[Edge, EdgeSpec]) -> dict[RiboNode, NodeLayout]:
        """
        Sort edges into in/out slots for each non-bulk node.
        Returns a NodeLayout per node — no coordinates involved.
        """
        layouts: dict[RiboNode, NodeLayout] = {}

        for node in graph.nodes:
            if node.phase == -1:
                continue

            nl = NodeLayout(node=node)

            # in-slots
            in_specs = [
                (specs[(u, v)], specs[(u, v)].direction)
                for u, v in graph.in_edges(node)
                if (u, v) in specs
            ]
            nl.in_slots = [
                EdgeSlot(edge=(s.u, s.v), direction=d)
                for s, d in sorted(
                    in_specs,
                    key=lambda sd: _sort_key(sd[0], sd[1], _IN_EDGE_ORDER))
            ]

            # out-slots
            out_specs = [
                (specs[(u, v)], specs[(u, v)].direction)
                for u, v in graph.out_edges(node)
                if (u, v) in specs
            ]
            nl.out_slots = [
                EdgeSlot(edge=(s.u, s.v), direction=d)
                for s, d in sorted(
                    out_specs,
                    key=lambda sd: _sort_key(sd[0], sd[1], _OUT_EDGE_ORDER))
            ]

            # record drop direction so geometry phase can use it
            for s, _ in out_specs:
                if s.etype == 'drop':
                    nl.drop_direction = s.direction

            layouts[node] = nl

        return layouts

    # ── Phase 3: local geometry ───────────────────────────────────────────────

    def compute_geometries(
        self,
        graph:   RiboGraph,
        specs:   dict[Edge, EdgeSpec],
        layouts: dict[RiboNode, NodeLayout],
        node_x:  dict[RiboNode, float],
    ) -> dict[Edge, EdgeGeom]:
        """
        Assign in0/in1/out0/out1 coordinates to every edge.
        Coordinates are *local-y* at this stage (x is already world x).
        """
        geoms: dict[Edge, EdgeGeom] = {
            (s.u, s.v): EdgeGeom(
                edge=(s.u, s.v),
                etype=s.etype,
                is_event=s.is_event,
                flux_end=s.flux_end,
            )
            for s in specs.values()
        }

        for node, nl in layouts.items():
            x = node_x[node]

            self._fill_side(
                node, nl, specs, geoms, x,
                slots=nl.in_slots, side='left', flux_key='flux_end',
            )
            self._fill_side(
                node, nl, specs, geoms, x,
                slots=nl.out_slots, side='right', flux_key='flux_start',
            )

        self._fill_bulk_geoms(graph, specs, geoms, node_x, bulk_length_factor=2)

        return geoms

    def _fill_side(
        self,
        node:     RiboNode,
        nl:       NodeLayout,
        specs:    dict[Edge, EdgeSpec],
        geoms:    dict[Edge, EdgeGeom],
        x:        float,
        slots:    list[EdgeSlot],
        side:     Literal['left', 'right'],
        flux_key: Literal['flux_start', 'flux_end'],
    ) -> None:
        sign     = -1 if side == 'left' else 1
        inout    = 'in' if side == 'left' else 'out'
        bottom_d = -1   if side == 'right' else 1
        top_d    = -bottom_d

        x_offset  = 0.0
        current_y = 0.0
        decay     = 0.0

        def flux(slot: EdgeSlot) -> float:
            s = specs[slot.edge]
            return s.flux_end if flux_key == 'flux_end' else s.flux_start

        def geom(slot: EdgeSlot) -> EdgeGeom:
            return geoms[slot.edge]

        def spec(slot: EdgeSlot) -> EdgeSpec:
            return specs[slot.edge]

        # ── bottom diagonals ─────────────────────────────────────────────────
        for slot in [s for s in slots if s.direction == bottom_d]:
            f = flux(slot)
            g = geom(slot)
            setattr(g, inout + '0', (x + x_offset,              current_y))
            setattr(g, inout + '1', (x + x_offset + sign * f,   current_y))
            setattr(g, inout + '_quad', 2 if side == 'left' else 1)
            if x_offset:
                self._add_helper_rect(g, x, current_y, x_offset, f)
            x_offset  += sign * f
            current_y += f

        # ── horizontal ───────────────────────────────────────────────────────
        h_slots = [s for s in slots if s.direction == 0]
        assert len(h_slots) <= 1

        for slot in h_slots:
            g  = geom(slot)
            s  = spec(slot)
            f  = s.flux_end if flux_key == 'flux_end' else s.flux_start
            decay = s.flux_start - s.flux_end
            setattr(g, inout + '_bot', (x, current_y))

            if side == 'left':
                if nl.drop_direction is not None:
                    if nl.drop_direction == 1:
                        g.in1, g.in0     = (x, current_y),         (x, current_y + f)
                        g.decay1, g.decay2 = (x, current_y + f),   (x, current_y + f + decay)
                    else:
                        g.in1, g.in0     = (x, current_y + decay), (x, current_y + decay + f)
                        g.decay1, g.decay2 = (x, current_y),       (x, current_y + decay)
                    current_y += f + decay
                    continue

                setattr(g, inout + '1', (x, current_y))
                setattr(g, inout + '0', (x, current_y + f))

            else:
                g.out0 = (x, current_y)
                g.out1 = (x, current_y + f)
                # decay anchor is set by the downstream node's drop_direction
                v_node = slot.edge[1]
                if v_node in {sl.edge[0] for sl in slots}:
                    pass   # resolved in a second pass if needed

            current_y += f + decay

        # ── top diagonals ────────────────────────────────────────────────────
        x_offset  = 0.0
        add        = decay if side == 'left' else 0.0
        current_y  = sum(flux(s) for s in slots) + add

        for slot in reversed([s for s in slots if s.direction == top_d]):
            f = flux(slot)
            g = geom(slot)
            setattr(g, inout + '0', (x + x_offset,             current_y))
            setattr(g, inout + '1', (x + x_offset + sign * f,  current_y))
            setattr(g, inout + '_quad', 3 if side == 'left' else 4)
            if x_offset:
                self._add_helper_rect(g, x, current_y, x_offset, f)
            x_offset  += sign * f
            current_y -= f

    def _fill_bulk_geoms(
        self,
        graph:              RiboGraph,
        specs:              dict[Edge, EdgeSpec],
        geoms:              dict[Edge, EdgeGeom],
        node_x:             dict[RiboNode, float],
        bulk_length_factor: float,
    ) -> None:
        DELTA_SIGN = {
            (-1, 'load'): +1, (-1, 'drop'): -1,
            (+1, 'load'): -1, (+1, 'drop'): +1,
        }
        for (u, v), s in specs.items():
            if u.phase != -1 and v.phase != -1:
                continue
            g      = geoms[(u, v)]
            length = max(s.flux_start * bulk_length_factor, 0.2)
            delta  = length * DELTA_SIGN[(s.direction, s.etype)]

            change_side = 'in' if v.phase == -1 else 'out'
            get_side    = 'out' if change_side == 'in' else 'in'

            for pos, opp in (('0', '1'), ('1', '0')):
                pt = getattr(g, f'{get_side}{pos}')
                if pt is not None:
                    setattr(g, f'{change_side}{opp}', _shift_pt(pt, delta, 'y'))

    @staticmethod
    def _add_helper_rect(g: EdgeGeom, x: float, y: float,
                         width: float, height: float) -> None:
        g.helper_rects.append([
            (x, y), (x + width, y), (x, y + height), (x + width, y + height)
        ])

    # ── Phase 4: global alignment ─────────────────────────────────────────────

    def align_layout(
        self,
        graph:   RiboGraph,
        geoms:   dict[Edge, EdgeGeom],
        layouts: dict[RiboNode, NodeLayout],
        node_x:  dict[RiboNode, float],
    ) -> LayoutResult:
        """
        Three sequential global alignment passes:
          4a. Align horizontal edges (make out_bot == in_bot)
          4b. Stack phases vertically with a buffer
          4c. Centre non-shift event edges horizontally
        Each pass calls _shift_node_geoms() — no direct dict mutation.
        """
        self._align_horizontal(graph, geoms, layouts)
        self._stack_phases(graph, geoms, layouts, buffer=1)
        self._centre_events(graph, geoms, layouts)
        return LayoutResult(geoms=geoms)

    # ── 4a: horizontal alignment ──────────────────────────────────────────────

    def _align_horizontal(
        self,
        graph:   RiboGraph,
        geoms:   dict[Edge, EdgeGeom],
        layouts: dict[RiboNode, NodeLayout],
    ) -> None:
        for node in topological_sort(graph):
            if node.phase == -1:
                continue
            for slot in layouts[node].out_slots:
                g = geoms[slot.edge]
                if g.is_event or g.out_bot is None or g.in_bot is None:
                    continue
                agreed = max(g.out_bot[1], g.in_bot[1])
                u, v   = slot.edge
                if agreed > g.out_bot[1]:
                    self._shift_node_geoms(u, agreed - g.out_bot[1], 'y',
                                           geoms, layouts, graph)
                if agreed > g.in_bot[1]:
                    self._shift_node_geoms(v, agreed - g.in_bot[1], 'y',
                                           geoms, layouts, graph)

    # ── 4b: phase stacking ────────────────────────────────────────────────────

    def _stack_phases(
        self,
        graph:   RiboGraph,
        geoms:   dict[Edge, EdgeGeom],
        layouts: dict[RiboNode, NodeLayout],
        buffer:  float,
    ) -> None:
        def phase_nodes(phase: int) -> list[RiboNode]:
            core   = [n for n in graph.nodes if n.phase == phase]
            adj_m1 = [
                n for n in graph.nodes if n.phase == -1
                if any(
                    (graph.has_edge(n, nb) and nb.phase == phase) or
                    (graph.has_edge(nb, n) and nb.phase == phase)
                    for nb in graph.neighbors(n)
                )
            ]
            return core + adj_m1

        def min_y(nodes: list[RiboNode]) -> float:
            pts = self._points_exclusive(nodes, geoms, graph)
            return min(p[1] for p in pts) if pts else 0.0

        def max_y(nodes: list[RiboNode]) -> float:
            pts = self._points_exclusive(nodes, geoms, graph)
            return max(p[1] for p in pts) if pts else 0.0

        core0 = [n for n in graph.nodes if n.phase == 0]
        p0    = phase_nodes(0)
        if p0:
            shift = -min_y(p0)
            for n in core0:
                self._shift_node_geoms(n, shift, 'y', geoms, layouts, graph)

        skip       = 0 if core0 else 1
        prev_nodes = p0

        for phase in range(1, 4):
            core_p = [n for n in graph.nodes if n.phase == phase]
            if not core_p:
                skip += 1
                continue
            p_nodes = phase_nodes(phase)
            shift   = (max_y(prev_nodes) + buffer + skip) - min_y(p_nodes)
            for n in core_p:
                self._shift_node_geoms(n, shift, 'y', geoms, layouts, graph)
            prev_nodes = p_nodes
            skip       = 0

    # ── 4c: event x-centring ─────────────────────────────────────────────────

    def _centre_events(
        self,
        graph:   RiboGraph,
        geoms:   dict[Edge, EdgeGeom],
        layouts: dict[RiboNode, NodeLayout],
    ) -> None:
        for (u, v), g in geoms.items():
            if u.phase == -1 or v.phase == -1:
                continue
            if not g.is_event or g.etype == 'shift':
                continue
            # Determine x-gap based on edge direction
            # direction +1 → in1/out0, direction -1 → in0/out1
            if g.in1 and g.out0:
                diff = g.in1[0] - g.out0[0]
            elif g.in0 and g.out1:
                diff = g.in0[0] - g.out1[0]
            else:
                continue
            if diff == 0:
                continue
            self._shift_node_geoms(u,  diff / 2, 'x', geoms, layouts, graph)
            self._shift_node_geoms(v, -diff / 2, 'x', geoms, layouts, graph)

    # ── Shared shift helper ───────────────────────────────────────────────────

    # Keys owned by each (is_event, side) combination
    _OWNED: dict[tuple[bool, str], list[str]] = {
        (False, 'out'): ['out0', 'out1', 'decay0', 'out_bot'],
        (False, 'in'):  ['in0',  'in1',  'decay1', 'decay2', 'in_bot'],
        (True,  'out'): ['out0', 'out1', 'decay0'],
        (True,  'in'):  ['in0',  'in1',  'decay1', 'decay2'],
    }

    def _shift_node_geoms(
        self,
        node:    RiboNode,
        delta:   float,
        axis:    Literal['x', 'y'],
        geoms:   dict[Edge, EdgeGeom],
        layouts: dict[RiboNode, NodeLayout],
        graph:   RiboGraph,
    ) -> None:
        """Shift all geom coordinates *owned* by node."""
        for edges, side in (
            (graph.out_edges(node), 'out'),
            (graph.in_edges(node),  'in'),
        ):
            for u, v in edges:
                g    = geoms.get((u, v))
                if g is None:
                    continue
                keys = self._OWNED[(g.is_event, side)]
                _shift_geom(g, delta, axis, keys)

        # Also shift bulk edge endpoints anchored at this node
        for u, v in graph.edges:
            if u == node and v.phase == -1:
                g = geoms.get((u, v))
                if g:
                    _shift_geom(g, delta, axis, ['in1', 'in0'])
            elif v == node and u.phase == -1:
                g = geoms.get((u, v))
                if g:
                    _shift_geom(g, delta, axis, ['out1', 'out0'])

    # ── Point-set helpers ─────────────────────────────────────────────────────

    _GEOM_POINT_KEYS = ('out0', 'in0', 'out1', 'in1', 'decay0', 'decay1', 'decay2')

    def _points_exclusive(
        self,
        nodes: list[RiboNode],
        geoms: dict[Edge, EdgeGeom],
        graph: RiboGraph,
    ) -> list[Pt]:
        node_set = set(nodes)
        return [
            pt
            for (u, v), g in geoms.items()
            if u in node_set and v in node_set
            for key in self._GEOM_POINT_KEYS
            if (pt := getattr(g, key)) is not None
        ]

