import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
from matplotlib.figure import Figure

from math import log
from networkx import topological_sort
from ..graph import RiboGraph
from ..simulation import RiboGraphFlux
from ..core import RiboNode


class RiboGraphVis(RiboGraph):
    """
    Holds all precomputed visualisation data for a RiboGraphFlux.
    Eager: all layout and render data is computed at construction time.
    """

    _IN_EDGE_ORDER = {
        ('shift',          +1, -1): 0,
        ('initiation',     +1):     1,
        ('shift',          +1, +1): 2,
        ('load',           +1):     3,
        # 4 reserved for direction == 0
        ('load',           -1):     5,
        ('shift',          -1, +1): 6,
        ('40s_retention',  -1):     7,
        ('shift',          -1, -1): 8,
    }

    _OUT_EDGE_ORDER = {
        ('shift',          -1, -1): 0,
        ('40s_retention',  -1):     1,
        ('shift',          -1, +1): 2,
        ('drop',           -1):     3,
        # 4 reserved for direction == 0
        ('drop',           +1):     5,
        ('shift',          +1, +1): 6,
        ('initiation',     +1):     7,
        ('shift',          +1, -1): 8,
    }

    EDGE_Y_KEYS = (
        'bot_right', 'top_right', 'bot_left', 'top_left',
        'decay_bot_right', 'decay_top_right'
    )

    # ── Construction ─────────────────────────────────────────────────────────

    def __init__(self, incoming_graph_data: RiboGraphFlux,
                fig_size=(12, 6), dpi=150, log_scale: float = 1, **attr):
        self.fig_size  = fig_size
        self.dpi       = dpi
        self.log_scale = log_scale

        super().__init__(**attr)
        
        # Add nodes explicitly first to ensure canonical instances are registered
        for node in incoming_graph_data.nodes:
            self.add_node(node)
        # Then add edges with their data
        for u, v, data in incoming_graph_data.edges(data=True):
            self.add_edge(u, v, **data)
        
    
        self.graph.update(incoming_graph_data.graph)
        self._prune_recycle_edges()
        self.compute_layout()

    # ── Top-level layout ─────────────────────────────────────────────────────

    def compute_layout(self):
        for node in self.nodes:
            self.nodes[node]['x']            = node.position
            self.nodes[node]['y']            = node.phase
            self.nodes[node]['helper_rects'] = []
            self.nodes[node]['base']         = 0
            self.nodes[node]['decay_side']   = None

        if self.log_scale > 1:
            self._log_scale_nodes_position()

        self._edge_type_direction()
        self._build_edge_order()
        self._node_layout()
        self._align_horizontal_edges()
        self._bulk_edge_source_position(2)
        self._adjust_phase_positions(1)
        self.fig = self._render()

    # ── Graph preparation ────────────────────────────────────────────────────

    def _prune_recycle_edges(self):
        edges_to_remove = [(u, v) for u, v in self.edges
                           if u.phase == -1 and v.phase == -1]
        self.remove_edges_from(edges_to_remove)
        nodes_to_remove = [n for n, d in self.degree() if d < 1]
        self.remove_nodes_from(nodes_to_remove)

    def _edge_type_direction(self):
        for u, v in self.edges:
            data  = self[u][v]
            event = u.phase != v.phase
            data['event'] = event

            if not event:
                data['type']      = str(u.phase)
                data['direction'] = 0
                continue

            # classify type
            if u.phase == -1:
                data['type'] = 'load'
            elif v.phase == -1:
                data['type'] = 'drop'
            elif u.phase == 0 and v.phase > 0:
                data['type'] = 'initiation'
            elif u.phase > 0 and v.phase == 0:
                data['type'] = '40s_retention'
            elif u.phase > 0 and v.phase > 0:
                data['type']    = 'shift'
                data['shift_n'] = v.position - u.position

            # bulk edges: direction set later in _assign_bulk_direction
            if u.phase == -1 or v.phase == -1:
                data['direction'] = None
            else:
                data['direction'] = int((v.phase - u.phase) / abs(v.phase - u.phase))

    def _log_scale_nodes_position(self):
        x_positions = sorted({self.nodes[node]['x'] for node in self.nodes})
        gaps        = [x_positions[i+1] - x_positions[i]
                       for i in range(len(x_positions) - 1)]
        log_gaps    = [log(g, self.log_scale) + 1 for g in gaps]

        log_x = [x_positions[0]]
        for gap in log_gaps:
            log_x.append(log_x[-1] + gap)

        log_map = dict(zip(x_positions, log_x))
        for node in self.nodes:
            self.nodes[node]['x'] = log_map[self.nodes[node]['x']]

    # ── Edge ordering ────────────────────────────────────────────────────────

    def _build_edge_order(self):
        for node in self.nodes:
            if node.phase == -1:
                continue
            self._assign_bulk_direction(node)
            self.nodes[node]['in_sorted_edges']  = self._sort_edges(
                self.in_edges(node,  data=True), self._IN_EDGE_ORDER)
            self.nodes[node]['out_sorted_edges'] = self._sort_edges(
                self.out_edges(node, data=True), self._OUT_EDGE_ORDER)

    def _assign_bulk_direction(self, node):
        """Set direction on load/drop edges attached to this node."""
        bulk_node = self._bulk_node_at(node.position)
        if bulk_node is None:
            return

        if (bulk_node, node) in self.edges:
            total = sum(
                (data.get('direction') or 0)
                for _, _, data in self.in_edges(node, data=True)
            )
            self[bulk_node][node]['direction'] = -1 if total >= 0 else 1
            self.nodes[node]['drop_direction'] = -1 if total >= 0 else 1

        if (node, bulk_node) in self.edges:
            total = sum(
                (data.get('direction') or 0)
                for _, _, data in self.out_edges(node, data=True)
            )
            self[node][bulk_node]['direction'] = -1 if total > 0 else 1
            self.nodes[node]['drop_direction'] = -1 if total > 0 else 1

    def _bulk_node_at(self, position):
        """Return the bulk RiboNode at the given position, or None."""
        for node in self.nodes:
            if node.phase == -1 and node.position == position:
                return node
        return None

    def _sort_edges(self, edges, edge_order):
        return [
            (u, v, data['direction'])
            for u, v, data in sorted(
                edges, key=lambda e: self._edge_sort_key(e, edge_order))
        ]

    def _edge_sort_key(self, edge, edge_order):
        u, v, data = edge
        etype     = data['type']
        direction = data['direction']
        shift_n   = data.get('shift_n', 0)

        if direction == 0:
            priority = 4
        elif etype == 'shift':
            priority = edge_order.get(('shift', direction, shift_n), 99)
        else:
            priority = edge_order.get((etype, direction), 99)

        assert priority != 99, \
            f"Unknown edge type/direction combo: {etype!r}, {direction!r}"
        return (priority, shift_n)

    # ── Node layout ──────────────────────────────────────────────────────────

    def _node_layout(self):
        for node in self.nodes:
            if node.phase == -1:
                continue
            self._layout_side(node, self.nodes[node]['in_sorted_edges'],
                               side='left',  flux_key='flux_end')
            self._layout_side(node, self.nodes[node]['out_sorted_edges'],
                               side='right', flux_key='flux_start')

    def _layout_side(self, node, edges, side, flux_key):
        sign  = -1 if side == 'left' else 1
        outer = side
        inner = 'left' if side == 'right' else 'right'
        x = self.nodes[node]['x']
        # ── bottom diagonals (direction == sign) ─────────────────────────
        x_offset  = 0
        current_y = 0
        for u, v, direction in edges:
            if direction != sign:
                continue
            flux = self[u][v][flux_key]
            self[u][v]['top_' + inner] = (x + x_offset,               current_y)
            self[u][v]['top_' + outer] = (x + x_offset + sign * flux,  current_y)
            if x_offset != 0:
                self.nodes[node]['helper_rects'].append(
                    ((x + x_offset, current_y),
                     (x, current_y + flux))
                )
            x_offset  += sign * flux
            current_y += flux

        # ── horizontal (direction == 0) ───────────────────────────────────
        for u, v, direction in edges:
            if direction != 0:
                continue
            flux       = self[u][v][flux_key]
            start_flux = self[u][v]['flux_start']
            end_flux   = self[u][v]['flux_end']
            decay      = start_flux - end_flux

            if side == 'right':
                self[u][v]['src_bot']  = current_y
                self[u][v]['bot_left'] = (x, current_y)
                self[u][v]['top_left'] = (x, current_y + flux)
            else:
                self[u][v]['tgt_bot'] = current_y
                if decay > 0:
                    if self.nodes[node]['drop_direction'] == 1:
                        self[u][v]['bot_right']      = (x, current_y)
                        self[u][v]['top_right']       = (x, current_y + flux)
                        self[u][v]['decay_bot_right'] = (x, current_y + flux)
                        self[u][v]['decay_top_right'] = (x, current_y + flux + decay)
                        u.decay_side = 'top'
                    else:
                        self[u][v]['bot_right']      = (x, current_y + decay)
                        self[u][v]['top_right']       = (x, current_y + decay + flux)
                        self[u][v]['decay_bot_right'] = (x, current_y)
                        self[u][v]['decay_top_right'] = (x, current_y + decay)
                        u.decay_side = 'bottom'
                else:
                    self[u][v]['bot_right'] = (x, current_y)
                    self[u][v]['top_right'] = (x, current_y + flux)

            current_y += flux + decay
            break

        # ── top diagonals (direction == -sign) ───────────────────────────
        x_offset  = 0
        current_y = sum(self[u][v][flux_key] for u, v, _ in edges)
        top = [(u, v) for u, v, d in edges if d == -sign]
        for u, v in reversed(top):
            flux = self[u][v][flux_key]
            self[u][v]['bot_' + inner] = (x + x_offset,               current_y)
            self[u][v]['bot_' + outer] = (x + x_offset + sign * flux,  current_y)
            if x_offset != 0:
                self.nodes[node]['helper_rects'].append(
                    ((x + x_offset + sign * flux, current_y),
                     (x, current_y))
                )
            x_offset  += sign * flux
            current_y -= flux

    # ── Alignment & shifting ─────────────────────────────────────────────────

    def _align_horizontal_edges(self):
        h_edges = [
            (u, v, data)
            for node in topological_sort(self)
            if node.phase != -1
            for u, v, data in self.out_edges(node, data=True)
            if data.get('direction') == 0
        ]
        for u, v, data in h_edges:
            print(f"  v={v}, id={id(v)}")
            for node in self.nodes:
                if node == v:
                    print(f"  matching node id={id(node)}, has out_sorted_edges={hasattr(node, 'out_sorted_edges')}")
        for u, v, data in h_edges:
            print(f"  h_edge {u} -> {v}, u.phase={u.phase}, v.phase={v.phase}")
            src_bot    = data['src_bot']
            tgt_bot    = data['tgt_bot']
            agreed_bot = max(src_bot, tgt_bot)

            if agreed_bot > src_bot:
                self._shift_node(u, agreed_bot - src_bot)
            if agreed_bot > tgt_bot:
                self._shift_node(v, agreed_bot - tgt_bot)

    def _shift_node(self, node, delta):
        """
        Shift all y-coordinates owned by this node.
        Ownership is split: each node owns the side it laid out.
          - out_sorted_edges: source owns left-side keys (bot_left, top_left, src_bot)
          - in_sorted_edges:  target owns right-side keys (bot_right, top_right,
                              decay_*, tgt_bot)
        """
        if node.phase == -1:
            return

        owned = {
            'out': ('bot_left',  'top_left'),
            'in':  ('bot_right', 'top_right', 'decay_bot_right',
                    'decay_top_right'),
        }

        for edges, keys in (
            (self.nodes[node]['out_sorted_edges'], owned['out']),
            (self.nodes[node]['in_sorted_edges'],  owned['in']),
        ):
            for u, v, _ in edges:
                data = self[u][v]
                for key in keys:
                    if key in data:
                        x, y = data[key]
                        data[key] = (x, y + delta)

        self.nodes[node]['helper_rects'] = [
            ((x0, y0 + delta), (x1, y1 + delta))
            for (x0, y0), (x1, y1) in self.nodes[node]['helper_rects']
        ]

    def _bulk_edge_source_position(self, bulk_length_factor):
        """Position the bulk (load/drop) edge rectangles."""
        for u, v, data in self.edges(data=True):
            if u.phase != -1 and v.phase != -1:
                continue
            flux   = data['flux_start']
            length = max(flux * bulk_length_factor, 0.2)
            
            change_side = 'top' if (
                (data['direction'] == 1 and v.phase == -1) or
                (data['direction'] == -1 and v.phase != -1)
            ) else 'bot'

            get_side = 'bot' if change_side == 'top' else 'top'

            # Direction determines whether we go up or down
            delta = -length if data['direction'] == 1 else length

            # Apply transformation consistently
            print(self[u][v])
            for pos in ('left', 'right'):
                x, y = data[f"{get_side}_{pos}"]
                data[f"{change_side}_{pos}"] = (x, y + delta)

    def _adjust_phase_positions(self, buffer: float):
        """Stack phases vertically with a buffer between them."""
        scanning_nodes = [n for n in self.nodes if n.phase == 0]
        min_y = min(self._phase_y_positions(0))
        for node in scanning_nodes:
            self._shift_node(node, -min_y)

        skip = 0
        for phase in range(1, 4):
            phase_nodes = [n for n in self.nodes if n.phase == phase]
            if not phase_nodes:
                skip += 1
                continue
            min_phase = min(self._phase_y_positions(phase))
            prev_max  = max(self._phase_y_positions(phase - 1))
            for node in phase_nodes:
                self._shift_node(node, (prev_max + buffer + skip) - min_phase)
            skip = 0

    def _phase_y_positions(self, phase) -> list[float]:
        phase_nodes = [n for n in self.nodes if n.phase == phase]
        return [
            data[key][1]
            for u, v, data in self.edges(phase_nodes, data=True)
            for key in self.EDGE_Y_KEYS
            if key in data
        ]

    # ── Rendering ────────────────────────────────────────────────────────────

    def _render(self) -> Figure:
        fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)
        ax.set_aspect('equal')
        ax.axis('off')

        for _, _, data in self.edges(data=True):
            if 'decay_start' in data:
                ax.add_patch(self._draw_patch(
                    [data['decay_start'],
                     data['decay_top_right'],
                     data['decay_bot_left']],
                    facecolor='steelblue'))
            ax.add_patch(self._draw_patch(
                [data['top_left'], data['top_right'],
                 data['bot_right'], data['bot_left']],
                facecolor='steelblue'))

        fig.tight_layout()
        return fig

    def _draw_patch(self, points: list, **kwargs):
        if len(points) < 3:
            raise ValueError('Points must contain at least 3 items')
        vertices = points + [points[0]]
        codes    = [Path.MOVETO] + [Path.LINETO] * (len(points) - 1) + [Path.CLOSEPOLY]
        return mpatches.PathPatch(Path(vertices, codes), **kwargs)

    # ── Output ───────────────────────────────────────────────────────────────

    def show(self):
        plt.figure(self.fig)
        plt.show()