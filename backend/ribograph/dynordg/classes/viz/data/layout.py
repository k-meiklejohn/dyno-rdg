from dataclasses import dataclass
from .edges import Edge, EdgeGeom, Pt, EdgeSpec

@dataclass
class LayoutResult:
    """
    The complete, renderer-ready description of the figure.
    This is the *only* object the renderer needs to read.
    """
    geoms: dict[Edge, EdgeGeom]

    @property
    def all_points(self) -> list[Pt]:
        pts: list[Pt] = []
        for g in self.geoms.values():
            for attr in ('in0', 'in1', 'out0', 'out1',
                         'decay0', 'decay1', 'decay2'):
                pt = getattr(g, attr)
                if pt is not None:
                    pts.append(pt)
            for rect in g.helper_rects:
                pts.extend(rect)
        return pts

