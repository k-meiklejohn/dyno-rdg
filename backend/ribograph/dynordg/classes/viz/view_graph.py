import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from networkx import DiGraph
import math

from ..simulation import TransitionMap
from ..core import RiboNode
from ..graph import RiboGraph

from ..simulation import RiboGraphFlux

@dataclass
class CanvasPos(tuple):
    def __new__(cls, *args):
        
        if len(args) == 1 and isinstance(args[0], (tuple, CanvasPos, RiboNode)):
            coords = args[0]
        elif len(args) == 2:
            coords = args
        else:
            raise ValueError(f'CanvasPos requires 2 ints or a length-2 tuple, got: {args}')

        if len(coords) != 2:
            raise ValueError(f'Canvas tuple must be of length 2, got length: {len(coords)}')
        for coord in coords:
            if not isinstance(coord, int):
                raise ValueError(f"Canvas coordinates must be 'int', got {type(coord).__name__!r}")

        return super().__new__(cls, coords)

    def __init__(self, *args):
        super().__init__()
        self.x = self[0]
        self.y = self[1]

    def __repr__(self):
        return f"(X:{self.x}, Y:{self.y})"



@dataclass
class EdgeData:
    def __init__(self, source: RiboNode, target: RiboNode):
        self.source = source
        self.target = target
        self.vertical = None
        self.event = source.phase == target.phase
        self.source_x = source.position
        self.source_y = source.phase
        self.target_x = target.position
        self.target_y = target.phase

        if self.target_y == -1:
            self.target_y = None

        if self.source_y == -1:
            self.source_y = None

    @property
    def _edge_type(self):
        if self.event:
            if self.source.phase == -1:
                return 'load'
            elif self.target.phase == -1:
                return 'drop'
            elif self.source.phase == 0 and self.target.phase > 0:
                return 'initiation'
            elif self.source.phase > 0 and self.target.phase == 0:
                return '40s_retention'
            elif self.source.phase > 0 and self.target.phase > 0:
                return 'frameshift'
        else:
            return str(self.source.phase)

        




class RiboGraphVis(RiboGraph):
    """
    Holds all precomputed visualisation data for a RiboGraphFlux.
    Eager: all layout and render data is computed at construction time.
    """

    _IN_EDGE_ORDER = {
        ('shift', +1, -1): 0,   # shift-1 up
        ('initiation', +1): 1,
        ('shift', +1, +1): 2,   # shift+1 up
        ('drop', +1):       3,
        # 4 reserved for direction == 0
        ('drop', -1):       5,
        ('shift', -1, +1):  6,  # shift+1 down
        ('shift', -1, -1):  7,  # shift-1 down
    }

    _OUT_EDGE_ORDER = {
        ('shift', -1, -1): 0,
        ('40S_retention', -1): 1,
        ('shift', -1, +1): 2,
        ('drop', -1):       3,
        # 4 reserved for direction == 0
        ('drop', +1):       5,
        ('shift', +1, +1):  6,
        ('shift', +1, -1):  7,
    }

    def __init__(self, incoming_graph_data: RiboGraphFlux, fig_size = (12,6), dpi = 150, log_scale : float = 1, **attr):
        self.fig_size = fig_size
        self.dpi = dpi
        self.log_scale = log_scale
        
        if isinstance(incoming_graph_data, RiboGraph):
            # Skip NetworkX conversion — initialise empty then steal the internals
            super().__init__(None, **attr)
            self._node = incoming_graph_data._node
            self._adj  = incoming_graph_data._adj
            self._pred = incoming_graph_data._pred
            self._succ = incoming_graph_data._succ
            self.graph.update(incoming_graph_data.graph)
        else:
            super().__init__(incoming_graph_data, **attr)
        self.compute_layout()


    def compute_layout(self):
        for node in self.nodes:
            node.x = node.position
            node.base = 0

        if self.log_scale > 1:
            self._log_scale_nodes_position()
        
        self._edge_type_direction()
        self._build_edge_order()
        self._node_layout()
        


    def _edge_type_direction(self):
        for u, v in self.edges:

            self[u][v]['event'] = u.phase != v.phase
            if self[u][v]['event']:
                if u.phase == -1 and v.phase > -1:
                    self[u][v]['type'] =  'load'
                elif v.phase == -1:
                    self[u][v]['type'] =  'drop'
                elif u.phase == 0 and v.phase > 0:
                    self[u][v]['type'] =  'initiation'
                elif u.phase > 0 and v.phase == 0:
                    self[u][v]['type'] =  '40s_retention'
                elif u.phase > 0 and v.phase > 0:
                    self[u][v]['type'] = 'shift'
                    self[u][v]['shift_n'] = v.position - u.position
            else:
                self[u][v]['type'] =  str(u.phase)
            
            if not self[u][v]['event']:
                self[u][v]['direction'] = 0

            elif u.phase == -1 or v.phase == -1:
                self[u][v]['direction'] = None


            else:
                self[u][v]['direction'] = int((v.phase - u.phase) / abs(v.phase - u.phase))
                print(u, v, self[u][v]['direction'])


    def _log_scale_nodes_position(self):
        """
        Change x position of all nodes in graph by scaling the distances between them by a logarithmic factor
        """
        
        x_pos =[]
        for node in self.nodes:
            x_pos.append(node.x)
        x_pos = sorted(list(set(x_pos)))

        
        gaps = [x_pos[i+1] - x_pos[i] for i in range(len(x_pos)-1)]

        log_gaps = [math.log(g, self.log_scale) + 1 for g in gaps]

        log_x_pos = [x_pos[0]]
        for gap in log_gaps:
            log_x_pos.append(log_x_pos[-1] + gap)

        log_map = dict(zip(x_pos, log_x_pos))

        for node in self.nodes:
            node.x = log_map[node.x]

    def _build_edge_order(self):
        for node in self.nodes:
            self._assign_bulk_direction(node)
            node.in_sorted_edges = self._sort_edges(
                self.in_edges(node, data=True), self._IN_EDGE_ORDER)
            node.out_sorted_edges = self._sort_edges(
                self.out_edges(node, data=True), self._OUT_EDGE_ORDER)

    def _assign_bulk_direction(self, node):

        if ((node.position, -1), node) in self.edges:
            print(node)
            total = sum((data.get('direction') or 0) for _, _, data in self.in_edges(node, data=True))
            self[(node.position, -1)][node]['direction'] = -1 if total >= 0 else 1
            node.drop_direction = -1 if total >= 0 else 1

        if (node, (node.position, -1)) in self.edges:
            total = sum((data.get('direction') or 0) for _, _, data in self.out_edges(node, data=True))
            self[node][(node.position, -1)]['direction'] = -1 if total > 0 else 1
            node.drop_direction = -1 if total > 0 else 1

    def _sort_edges(self, edges, edge_order):
        return [
            (u, v, data['direction'])
            for u, v, data in sorted(edges, key=lambda e: self._edge_sort_key(e, edge_order))
        ]

    def _edge_sort_key(self, edge, EDGE_ORDER):
        u, v, data = edge
        etype = data['type']
        direction = data['direction']
        shift_n = data.get('shift_n', 0)

        if direction == 0:
            priority = 4
        elif etype == 'shift':
            priority = EDGE_ORDER.get(('shift', direction, shift_n), 99)
        else:
            priority = EDGE_ORDER.get((etype, direction), 99)

        return (priority, shift_n)
    
    def _node_layout(self):
        for node in self.nodes:
            node.helper_rects = []

            # Incoming (left, end_flux)
            self._layout_side(
                node,
                node.in_sorted_edges,
                side="left",
                flux_key="end_flux"
            )

            # Outgoing (right, start_flux)
            self._layout_side(
                node,
                node.out_sorted_edges,
                side="right",
                flux_key="start_flux"
            )

    def _layout_side(self, node, edges, side, flux_key):
        x_offset = 0
        current_y = 0

        sign = -1 if side == "left" else 1

        # -------- TOP + MIDDLE --------
        for u, v, direction in edges:
            flux = self[u][v][flux_key]

            if direction == 1:
                self[u][v]['top_right'] = (node.x + x_offset, current_y)
                self[u][v]['top_left']  = (node.x + x_offset + sign * flux, current_y)

                if x_offset != 0:
                    node.helper_rects.append(
                        ((node.x + x_offset, current_y),
                        (node.x, current_y + flux))
                    )

                x_offset += sign * flux
                current_y += flux

            elif direction == 0:
                start_flux = self[u][v]['start_flux']
                end_flux   = self[u][v]['end_flux']
                decay      = start_flux - end_flux

                if node.drop_direction == 1:


                self[u][v]['bot_right'] = (node.x + x_offset, current_y)
                self[u][v]['top_right'] = (node.x + x_offset, current_y + flux)
                break
            else:
                break

        # -------- BOTTOM --------
        x_offset = 0
        current_y = sum(self[u][v][flux_key] for u, v, _ in edges)

        for u, v, direction in reversed(edges):
            flux = self[u][v][flux_key]

            if direction == -1:
                self[u][v]['bot_right'] = (node.x + x_offset, current_y)
                self[u][v]['bot_left']  = (node.x + x_offset + sign * flux, current_y)

                if x_offset != 0:
                    node.helper_rects.append(
                        ((node.x + x_offset + sign * flux, current_y),
                        (node.x, current_y))
                    )

                x_offset += sign * flux
                current_y -= flux
            else:
                break
                    


    # def _render(self) -> Figure:
    #     """Draw everything onto a Figure using the precomputed layout."""
    #     fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
    #     ax.set_aspect("equal")
    #     ax.axis("off")

    #     # Your custom drawing logic goes here
    #     # e.g. ax.add_patch(...), ax.annotate(...), bezier curves, etc.

    #     fig.tight_layout()
    #     return fig

    # # --- Output targets ---

    # def show(self):
    #     """Interactive display."""
    #     plt.figure(self.fig)
    #     plt.show()

    # def save(self, path: str, **kwargs):
    #     """Save to file. kwargs passed to savefig (dpi, bbox_inches, etc.)"""
    #     self.fig.savefig(path, **kwargs)

    # def get_figure(self) -> Figure:
    #     """Return the Figure object directly (e.g. for notebook embedding)."""
    #     return self.fig

    # def redraw(self, **kwargs) -> "RiboGraphVis":
    #     """Recompute layout and re-render with updated params, returns new instance."""
    #     return RiboGraphVis(self.fluxgraph, **{
    #         "figsize": self.figsize,
    #         "dpi": self.dpi,
    #         **kwargs
    #     })