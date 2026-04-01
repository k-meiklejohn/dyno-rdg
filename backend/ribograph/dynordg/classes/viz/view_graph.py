import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from ..core import RiboNode

if TYPE_CHECKING:
    from ..simulation import RiboGraphFlux

@dataclass
class CanvasPos(tuple):
    def __new__(cls, *args):
        
        if len(args) == 1 and isinstance(args[0], (tuple, CanvasPos)):
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

        




class RiboGraphVis:
    """
    Holds all precomputed visualisation data for a RiboGraphFlux.
    Eager: all layout and render data is computed at construction time.
    """

    def __init__(self, fluxgraph: "RiboGraphFlux", figsize=(12, 8), dpi=150):
        self.fluxgraph = fluxgraph
        self.figsize = figsize
        self.dpi = dpi

        # Computed eagerly at construction
        self.layout: SankeyLayout = self._compute_layout()
        self.fig: Figure = self._render()

    def _compute_layout(self) -> SankeyLayout:
        """Extract and precompute all geometry from the fluxgraph."""

        for node in self.fluxgraph.nodes:


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