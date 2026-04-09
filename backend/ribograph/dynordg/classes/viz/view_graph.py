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
        'out0', 'in0', 'out1', 'in1', "deacy0", "decay1", "decay2"
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
        self._bulk_edge_source_position(2)
        self._align_events()
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

        if (node, bulk_node) in self.edges:
            total = sum(
                (data.get('direction') or 0)
                for _, _, data in self.out_edges(node, data=True)
            )
            self[node][bulk_node]['direction'] = -1 if total > 0 else 1
            self.nodes[node]['drop_direction'] = self[node][bulk_node]['direction']

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
        _, _, data = edge
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

        x = self.nodes[node]['x']
        # ── bottom diagonals (direction == sign) ─────────────────────────
        x_offset  = 0
        current_y = 0

        bottom_d = -1 if side == 'right' else 1
        top_d    = bottom_d * -1
        inout = 'in' if side == 'left' else 'out'


        bottom = [(u, v) for u, v, d in edges if d == bottom_d]
        for u, v in bottom:
            print(u, v, current_y)
            flux = self[u][v][flux_key]
            self[u][v][inout + '0'] = (x + x_offset,                current_y)
            self[u][v][inout + '1'] = (x + x_offset + sign * flux,  current_y)
            if x_offset != 0:
                self.nodes[node]['helper_rects'].append(
                    ((x + x_offset, current_y),
                     (x, current_y + flux))
                )
            x_offset  += sign * flux
            current_y += flux

        horizontal = [(u, v) for u, v, d in edges if d == 0]
        assert len(horizontal) <= 1
    
        # ── horizontal (direction == 0) ───────────────────────────────────
        for u, v in horizontal:
            flux       = self[u][v][flux_key]
            start_flux = self[u][v]['flux_start']
            end_flux   = self[u][v]['flux_end']
            decay      = start_flux - end_flux
            self[u][v][inout + '_bot']  = current_y

            
            if side == 'left':
                if 'drop_direction' in self.nodes[node].keys():
                    if self.nodes[node]['drop_direction'] == 1:
                        self[u][v]['in1']      = (x, current_y)
                        self[u][v]['in0']       = (x, current_y + flux)
                        self[u][v]['decay1'] = (x, current_y + flux)
                        self[u][v]['decay2'] = (x, current_y + flux + decay)
                        continue
                    else:
                        self[u][v]['in1']      = (x, current_y + decay)
                        self[u][v]['in0']       = (x, current_y + decay + flux)
                        self[u][v]['decay1'] = (x, current_y)
                        self[u][v]['decay2'] = (x, current_y + decay)
                        continue
                    
                self[u][v][inout + '1'] = (x, current_y)
                self[u][v][inout + '0'] = (x, current_y + flux)
            
            else:   
                self[u][v]['out0'] = (x, current_y)
                self[u][v]['out1'] = (x, current_y + flux)
                if self.nodes[v]['drop_direction'] == -1:
                    self[u][v]['decay0'] =  self[u][v]['out0']
                else:
                    self[u][v]['decay0'] = self[u][v]['out1']



            current_y += flux + decay

        # ── top diagonals (direction == -sign) ───────────────────────────
        x_offset  = 0
        current_y = sum(self[u][v][flux_key] for u, v, _ in edges)
        top = [(u, v) for u, v, d in edges if d == top_d]
        for u, v in reversed(top):
            flux = self[u][v][flux_key]
            self[u][v][inout + '0'] = (x + x_offset,               current_y)
            self[u][v][inout + '1'] = (x + x_offset + sign * flux,  current_y)

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
            if not data.get('event')
        ]

        for u, v, data in h_edges:

            src_bot    = data['out_bot']
            tgt_bot    = data['in_bot']
            agreed_bot = max(src_bot, tgt_bot)

            if agreed_bot > src_bot:
                self._shift_node(u, agreed_bot - src_bot, 'y')
            if agreed_bot > tgt_bot:
                self._shift_node(v, agreed_bot - tgt_bot, 'y')

    def _shift_node(self, node, delta, xy):
        """
        Shift all edge y-coordinates owned by this node.
        """
        if node.phase == -1:
            return
        
        owned = {
            'out': ['out0', 'out1', 'decay0'],
            'in':  ['in0', 'in1', 'decay1', 'decay2']
        }


        for edges, side in (
            (self.out_edges(node), 'out'),
            (self.in_edges(node),  'in'),
        ):
            for u, v in edges:
                data = self[u][v]
                keys = owned[side]
                for key in keys:
                    if key in data:
                        x, y = data[key]
                        if xy == 'y':
                            data[key] = (x, y + delta)
                        elif xy =='x':
                            data[key] = (x + delta, y)


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
            
            change_side = 'in' if v.phase == -1 else 'out'
            get_side = 'in' if change_side == 'out' else 'out'

            # Direction determines whether we go up or down
            delta = length if data['direction'] == 1 else -length

            # Apply transformation consistently
            for pos in ('0', '1'):
                opp = '1' if pos =='0' else '0'
                x, y = data[f"{get_side}{pos}"]
                data[f"{change_side}{opp}"] = (x, y + delta)

    def _adjust_phase_positions(self, buffer: float):
        """Stack phases vertically with a buffer between them."""
        scanning_nodes = [n for n in self.nodes if n.phase == 0]
        min_y = min(pos[1] for pos in self._edge_position_from_nodes(scanning_nodes, exclusive=True))
        for node in scanning_nodes:
            self._shift_node(node, -min_y, 'y')

        skip = 0
        prev_nodes = scanning_nodes
        for phase in range(1, 4):
            phase_nodes = [n for n in self.nodes if n.phase == phase]
            if not phase_nodes:
                skip += 1
                continue
            min_phase = min(pos[1] for pos in self._edge_position_from_nodes(phase_nodes, exclusive=True))
            if not len(prev_nodes) == 0:
                prev_max  = max(pos[1] for pos in self._edge_position_from_nodes(prev_nodes, exclusive=True))
            for node in phase_nodes:
                self._shift_node(node, (prev_max + buffer + skip) - min_phase, 'y')
            prev_nodes = phase_nodes
            skip = 0
    
    def _align_events(self):
        for u, v, data in self.edges(data=True):
            if u.phase == -1 or v.phase == -1:
                continue
            if (not data['event']) or (data['type'] == 'shift'):
                continue
            difference = data['in1'][0] - data['out1'][0]
            if difference == 0:
                continue
            difference *= data['direction']
            self._shift_node(u, difference/2, 'x')
            self._shift_node(v, -difference/2, 'x')

    def _edge_position_from_nodes(self, nbunch, exclusive=False) -> list[tuple[float,float]]:
        if not isinstance(nbunch, RiboNode):
            for node in nbunch:
                if node not in self:
                    raise ValueError(f'Node: {node} not in Graph')
        else:
            if nbunch not in self:
                raise ValueError(f'Node: {nbunch} not in Graph')
        nodes = set(nbunch)
        if exclusive:
            return [
                data[key]
                for u, v, data in self.edges(nbunch, data=True)
                if u in nodes and v in nodes
                for key in self.EDGE_Y_KEYS
                if key in data
            ]
        else:
            return [
                data[key]
                for u, v, data in self.edges(nbunch, data=True)
                for key in self.EDGE_Y_KEYS
                if key in data
            ]

    @property
    def positions(self) -> list[list]:
        out = []
        for node in self.nodes:
            out.extend(self._edge_position_from_nodes(node))
        return out
    


    # ── Rendering ────────────────────────────────────────────────────────────

    def _render(self) -> Figure:
        fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)
        ax.set_aspect('equal')
        ax.axis('off')

        all_x = [pos[0] for pos in self.positions]
        all_y = [pos[1] for pos in self.positions]

        xmin, xmax = min(all_x), max(all_x)
        ymin, ymax = min(all_y), max(all_y)

        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
        for u, v, data in self.edges(data=True):
            if 'decay0' in data.keys():
                if data['decay1'] != data['decay2']:
                    ax.add_patch(self._draw_patch(
                        [data['decay0'],
                        data['decay1'],
                        data['decay2']],
                        facecolor='steelblue'))
            print(u,v, data)
            ax.add_patch(self._draw_patch(
                [data['in0'], data['in1'],
                 data['out0'], data['out1']],
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