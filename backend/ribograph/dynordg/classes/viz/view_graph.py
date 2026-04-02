import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from networkx import DiGraph
import math

from backend.ribograph.dynordg.classes.simulation.transitionmap import TransitionMap
from ..core import RiboNode
from ..graph import RiboGraph

if TYPE_CHECKING:
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

    def __init__(self, incoming_graph_data: RiboGraphFlux, fig_size = (12,6), dpi = 150,  **attr):
        self.fig_size = fig_size
        self.dpi = dpi
        
        super().__init__(incoming_graph_data, **attr)
        for node in self.nodes:
            node.x = node.position

    
    def _log_scale_nodes_position(self, factor:float):
        x_pos =[]
        for node in self.nodes:
            x_pos.append(node.x)
        x_pos = set(x_pos)

        
        gaps = [x_pos[i+1] - x_pos[i] for i in range(len(x_pos)-1)]
        log_gaps = [math.log(g, factor) + 1 for g in gaps]

        if log_reduction > 1: #SCALE GRAPH BASED ON LOG VALUES OF DISTANCES
            # --- Compute unique sorted x positions ---
            x_pos = sorted(set(list(out['source_x']) + [out['target_x'].iloc[-1]]))

            # --- Compute gaps and log-scale ---
            gaps = [x_pos[i+1] - x_pos[i] for i in range(len(x_pos)-1)]
            log_gaps = [math.log(g, log_reduction) + 1 for g in gaps]

            # --- Map original x to log-scaled x ---
            log_x_pos = [x_pos[0]]
            for gap in log_gaps:
                log_x_pos.append(log_x_pos[-1] + gap)

            log_map = dict(zip(x_pos, log_x_pos))

            out['source_x'] = out['source_x'].map(log_map)
            out['target_x'] = out['target_x'].map(log_map)

    def _compute_layout(self):
        """Extract and precompute all geometry from the fluxgraph."""

        for node in self.fluxgraph.nodes:
            pass

    def _scale_graph(self):
        
    def


    def _render(self) -> Figure:
        """Draw everything onto a Figure using the precomputed layout."""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.set_aspect("equal")
        ax.axis("off")

        # Your custom drawing logic goes here
        # e.g. ax.add_patch(...), ax.annotate(...), bezier curves, etc.

        fig.tight_layout()
        return fig

    # --- Output targets ---

    def show(self):
        """Interactive display."""
        plt.figure(self.fig)
        plt.show()

    def save(self, path: str, **kwargs):
        """Save to file. kwargs passed to savefig (dpi, bbox_inches, etc.)"""
        self.fig.savefig(path, **kwargs)

    def get_figure(self) -> Figure:
        """Return the Figure object directly (e.g. for notebook embedding)."""
        return self.fig

    def redraw(self, **kwargs) -> "RiboGraphVis":
        """Recompute layout and re-render with updated params, returns new instance."""
        return RiboGraphVis(self.fluxgraph, **{
            "figsize": self.figsize,
            "dpi": self.dpi,
            **kwargs
        })