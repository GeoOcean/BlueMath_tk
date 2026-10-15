"""
Point labels that do not overlap each other.
"""

from __future__ import annotations

from collections.abc import Sequence

from matplotlib.axes import Axes

#: Candidate offsets (points, ha, va): 8 compass directions per ring, near to far.
_OFFSETS: list[tuple[tuple[float, float], str, str]] = [
    ((radius * ux / ((ux**2 + uy**2) ** 0.5), radius * uy / ((ux**2 + uy**2) ** 0.5)), ha, va)
    for radius in (6, 11, 16, 22, 28, 34, 42, 55)
    for ux, uy, ha, va in (
        (1, 1, "left", "bottom"),
        (1, -1, "left", "top"),
        (-1, 1, "right", "bottom"),
        (-1, -1, "right", "top"),
        (0, 1, "center", "bottom"),
        (0, -1, "center", "top"),
        (1, 0, "left", "center"),
        (-1, 0, "right", "center"),
    )
]


def _overlap(a: Sequence[float], b: Sequence[float], margin: float) -> bool:
    return not (
        a[2] + margin < b[0] or b[2] + margin < a[0] or a[3] + margin < b[1] or b[3] + margin < a[1]
    )


def place_labels(
    ax: Axes,
    x: Sequence[float],
    y: Sequence[float],
    texts: Sequence[str],
    *,
    leader_colors: Sequence[str] | None = None,
    **text_kwargs,
) -> None:
    """
    Annotate points, each at the nearest offset that clears the labels already placed.

    The axes limits must be final (the label sizes are measured on the
    rendered figure). A label pushed far from its point gets a leader line.

    Parameters
    ----------
    ax : Axes
        Axes with its final limits.
    x, y : sequence of float
        Points in data coordinates.
    texts : sequence of str
        One label per point.
    leader_colors : sequence of str, optional
        Leader-line colour per point. Default is grey.
    **text_kwargs
        Passed to ``ax.annotate`` (``fontsize``, ``path_effects``, ...).
    """

    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    inv = ax.transData.inverted()
    pad = abs(inv.transform((3.5 * ax.figure.dpi / 72.0, 0))[0] - inv.transform((0, 0))[0])
    placed: list[tuple[float, float, float, float]] = []
    for i, (px, py, text) in enumerate(zip(x, y, texts)):
        for offset, ha, va in _OFFSETS:
            probe = ax.annotate(text, (px, py), xytext=offset, textcoords="offset points",
                                ha=ha, va=va, **text_kwargs)
            bbox = probe.get_window_extent(renderer)
            probe.remove()
            (x0, y0), (x1, y1) = inv.transform((bbox.x0, bbox.y0)), inv.transform((bbox.x1, bbox.y1))
            box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            if not any(_overlap(box, other, pad) for other in placed):
                break
        placed.append(box)
        leader = (offset[0] ** 2 + offset[1] ** 2) ** 0.5 > 9
        color = leader_colors[i] if leader_colors is not None else "0.4"
        ax.annotate(
            text, (px, py), xytext=offset, textcoords="offset points", ha=ha, va=va,
            arrowprops=dict(arrowstyle="-", color=color, lw=0.7, alpha=0.75, shrinkA=0, shrinkB=3)
            if leader else None,
            **text_kwargs,
        )


__all__ = ["place_labels"]
