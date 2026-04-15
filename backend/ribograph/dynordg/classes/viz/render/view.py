"""
ribo_graph_vis.py  –  Refactored visualisation layer for RiboGraphFlux.

Architecture
------------
RiboGraphVis          – layout engine  (unchanged public API)
  └─ RiboRenderer     – stateless factory that turns layout data → patches
       └─ EdgePainter – per-edge patch builder (rect / taper / decay)

Changing the look of any edge type now only requires editing EdgePainter or
the COLOR_DICT / STYLE_DICT on RiboRenderer.  The layout maths is untouched.
"""

from __future__ import annotations
from dataclasses import dataclass
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.path import Path
from ..data import EdgeGeom, LayoutResult

# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EdgeStyle:
    """Visual properties for a single rendered edge segment."""
    facecolor: str
    edgecolor: str
    alpha: float = 1.0
    linewidth: float = 0.5
    zorder: int = 2


@dataclass
class RenderPrimitive:
    """A single matplotlib patch to be added to the axes."""
    patch: mpatches.Patch
    zorder: int = 2


# ─────────────────────────────────────────────────────────────────────────────
# EdgePainter  –  builds patches for one edge
# ─────────────────────────────────────────────────────────────────────────────

class EdgePainter:
    """
    Converts the pre-computed layout data for a single edge into a list of
    RenderPrimitive objects.

    This is the *only* class you need to touch to change how edges look.
    """

    TAPER_DEPTH_SCALE: float = 0.4

    def __init__(self, geom: EdgeGeom, style: EdgeStyle):
        self.geom  = geom
        self.u, self.v = geom.edge
        self.style = style

    # ── Public entry point ───────────────────────────────────────────────────

    def primitives(self) -> list[RenderPrimitive]:
        """Return all patches needed to draw this edge."""
        out: list[RenderPrimitive] = []

        out.extend(self._body_primitives())
        out.extend(self._helper_rect_primitives())
        out.extend(self._decay_primitives())
        out.extend(self._taper_primitives())

        return out

    # ── Body rect ────────────────────────────────────────────────────────────

    def _body_primitives(self) -> list[RenderPrimitive]:
        g = self.geom
        corners = [g.in0, g.in1, g.out0, g.out1]
        patch = self._rect_patch(corners)
        return [RenderPrimitive(patch, zorder=self.style.zorder)]

    # ── Helper rects (stacked bulk offsets) ──────────────────────────────────

    def _helper_rect_primitives(self) -> list[RenderPrimitive]:
        return [
            RenderPrimitive(self._rect_patch(r), zorder=self.style.zorder)
            for r in self.geom.helper_rects
        ]

    # ── Decay wedge ──────────────────────────────────────────────────────────

    def _decay_primitives(self) -> list[RenderPrimitive]:
        g = self.geom

        if g.decay0 is None or g.decay1 is None or g.decay2 is None:

            return []

        if g.decay1 == g.decay2:
            return []
        patch = self._rect_patch(
            [g.decay0, g.decay1, g.decay2],
            facecolor='purple',
            edgecolor='purple',
        )
        return [RenderPrimitive(patch, zorder=self.style.zorder - 1)]

    # ── Curved tapers (event edges only) ─────────────────────────────────────

    def _taper_primitives(self) -> list[RenderPrimitive]:
        g = self.geom

        if not g.is_event:
            return []

        out: list[RenderPrimitive] = []
        flux = g.flux_end

        if self.v.phase != -1 and g.in0 is not None and g.in_quad is not None:
            patch = self._taper_patch(
                centre=g.in0,
                flux=flux,
                quad=g.in_quad,
                inout='in',
            )
            out.append(RenderPrimitive(patch, zorder=self.style.zorder + 1))

        if self.u.phase != -1 and g.out0 is not None and g.out_quad is not None:
            patch = self._taper_patch(
                centre=g.out0,
                flux=flux,
                quad=g.out_quad,
                inout='out',
            )
            out.append(RenderPrimitive(patch, zorder=self.style.zorder + 1))

        return out

    # ── Patch constructors ───────────────────────────────────────────────────

    def _patch_kwargs(self, **overrides) -> dict:
        kw = dict(
            facecolor=self.style.facecolor,
            edgecolor=self.style.edgecolor,
            alpha=self.style.alpha,
            linewidth=0,
        )
        kw.update(overrides)
        return kw

    def _rect_patch(self, points: list[tuple], **overrides) -> mpatches.PathPatch:
        pts = [p for p in points if p is not None]
        if len(pts) < 3:
            raise ValueError('rect_patch needs ≥ 3 valid points')

        verts = pts + [pts[0]]
        codes = [Path.MOVETO] + [Path.LINETO] * (len(pts) - 1) + [Path.CLOSEPOLY]

        return mpatches.PathPatch(Path(verts, codes), **self._patch_kwargs(**overrides))

    def _taper_patch(
        self,
        centre: tuple[float, float],
        flux: float,
        quad: int,
        inout: str,
    ) -> mpatches.PathPatch:
        """Curved tapered cap for event-edge endpoints."""
        x, y   = centre
        x_sign = -1 if quad in (1, 4) else 1
        y_sign =  1 if quad in (1, 2) else -1

        if inout == 'out':
            verts = [
                (x, y),
                (x,                  y + y_sign * flux),
                (x - x_sign * flux,  y + y_sign * flux),
                (x - x_sign * flux,  y),
            ]
        else:  # 'in'
            verts = [
                (x,                  y),
                (x,                  y + y_sign * flux),
                (x - x_sign * flux,  y + y_sign * flux),
                (x - x_sign * flux,  y),
            ]

        codes = [
            Path.MOVETO, Path.LINETO,
            Path.CURVE3, Path.CURVE3,
        ]
        return mpatches.PathPatch(Path(verts, codes), **self._patch_kwargs())


# ─────────────────────────────────────────────────────────────────────────────
# RiboRenderer  –  assembles all patches for a fully-laid-out graph
# ─────────────────────────────────────────────────────────────────────────────

class RiboRenderer:
    """
    Stateless renderer.  Call ``render(graph)`` to get a Figure.

    Customise visuals by subclassing and overriding COLOR_DICT / STYLE_DICT,
    or by replacing ``edge_style()`` entirely.
    """

    # ── Colour palette ───────────────────────────────────────────────────────

    COLOR_DICT: dict[str, str] = {
        'shift':           'red',
        '40s_retention':   'orange',
        'drop':            'purple',
        'initiation':      'green',
        'load':            'purple',
        'decay':           'purple',
        '0':               'gray',
        '1':               'lightblue',
        '2':               'steelblue',
        '3':               'darkblue',
    }

    # ── Per-type style overrides (merged on top of the default) ──────────────

    STYLE_OVERRIDES: dict[str, dict] = {
        # e.g. 'shift': {'alpha': 0.8, 'linewidth': 1.0},
    }

    # ── Construction / entry point ───────────────────────────────────────────

    def __init__(self, fig_size: tuple = (12, 6), dpi: int = 150):
        self.fig_size = fig_size
        self.dpi      = dpi

    def render(self, graph: 'LayoutResult') -> Figure:
        fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)
        ax.set_aspect('equal')
        ax.axis('off')

        self._set_axis_limits(ax, graph)

        primitives = self._collect_primitives(graph)
        for prim in sorted(primitives, key=lambda p: p.zorder):
            prim.patch.set_transform(ax.transData)
            ax.add_patch(prim.patch)

        fig.tight_layout()
        return fig

    # ── Primitive collection ─────────────────────────────────────────────────

    def _collect_primitives(self, layout: LayoutResult) -> list[RenderPrimitive]:
        out: list[RenderPrimitive] = []

        for geom in layout.geoms.values():
            style   = self.edge_style(geom)
            painter = EdgePainter(geom, style)
            out.extend(painter.primitives())

        return out
    
    # ── Style resolution ─────────────────────────────────────────────────────

    def edge_style(self, geom: EdgeGeom) -> EdgeStyle:
        etype     = geom.etype
        color     = self.COLOR_DICT.get(etype, 'black')
        overrides = self.STYLE_OVERRIDES.get(etype, {})

        return EdgeStyle(
            facecolor=color,
            edgecolor=color,
            **overrides,
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _set_axis_limits(ax, graph: 'LayoutResult') -> None:
        positions = graph.all_points
        if not positions:
            return
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        ax.set(xlim=(min(xs), max(xs)), ylim=(min(ys), max(ys)))

