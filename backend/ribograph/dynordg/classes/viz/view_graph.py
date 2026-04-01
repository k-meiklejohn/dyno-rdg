import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .fluxgraph import RiboGraphFlux

@dataclass
class SankeyLayout:
    """Precomputed geometry for a single node or edge — whatever your renderer needs."""
    node_positions: dict   # node_id -> (x, y)
    edge_flows: dict       # (src, dst) -> flux value
    edge_paths: dict       # (src, dst) -> precomputed path/bezier data
    node_sizes: dict       # node_id -> size scalar


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
        # Your custom layout logic goes here
        ...

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