"""Geometry for laying one cell per channel out at its real position.

A per-channel value heatmap and a per-channel trace grid draw completely
different things into their cells, but agree entirely on where the cells go:
one square per channel at its ``(x, y)``, sized per channel, with a sensible
layout when the positions are missing or degenerate. That agreement is what
lives here.

* :func:`tiled_grid_positions` — a square-ish fallback layout for sources that
  carry no positions at all.
* :func:`tile_by_group` — fan out groups of channels that share a coordinate
  range, so one group does not render on top of another.
* :func:`infer_pitch` — the spacing between adjacent channels, which is what a
  cell defaults to when no size is given.
* :func:`resolve_cell_geometry` — ``(positions, sizes)`` into per-channel
  rectangles ``(x0, y0, side)`` and centers, honouring ``invert_y`` and the
  degenerate "every channel at one point" case.
* :func:`build_quad_mesh_arrays` — tessellate those rects into a quad mesh, for
  a renderer that fills cells rather than drawing into them.

Positions are plain arrays. Deriving them from a particular data model's
channel metadata is that model's business, not this module's.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "build_quad_mesh_arrays",
    "infer_pitch",
    "resolve_cell_geometry",
    "tile_by_group",
    "tiled_grid_positions",
]

# Gap between adjacent tiled blocks, in channel pitches.
_GROUP_GUTTER_PITCHES = 2.0

# Quad corner offsets (unit square, lower-left origin) and the two triangles
# that tessellate it. Multiplied by each channel's side length and added to its
# lower-left corner to build a mesh.
_CORNERS = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
_QUAD_TRIS = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)


def tiled_grid_positions(n_ch: int) -> np.ndarray:
    """Fallback layout for sources without electrode positions (e.g. NWB
    playback of plain ``TimeSeries`` streams).

    Packs channels into a roughly-square ``ceil(sqrt(n_ch)) × ceil(n_ch/side)``
    grid so they're all visible as individual tiles rather than stacking on top
    of each other at the origin.
    """
    if n_ch <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    side = int(np.ceil(np.sqrt(n_ch)))
    idx = np.arange(n_ch)
    positions = np.column_stack([idx % side, idx // side]).astype(np.float32)
    return positions


def tile_by_group(positions: np.ndarray, group_ids: np.ndarray) -> np.ndarray:
    """Shift each group's block along x so groups render side by side.

    Devices that are physically separate often report the same coordinate
    range -- two arrays each numbering their electrodes from the same origin --
    so their channels land on identical ``(x, y)`` and one block hides another.
    Each group keeps its internal layout and is placed to the right of the
    previous one, with a :data:`_GROUP_GUTTER_PITCHES`-pitch gutter between
    blocks.

    A single distinct group is returned unchanged, so the layout is only
    rewritten when there is genuinely more than one block to separate.
    """
    positions = np.asarray(positions, dtype=np.float32)
    hs = np.asarray(group_ids).reshape(-1)
    if hs.shape[0] != positions.shape[0]:
        return positions
    groups = np.unique(hs)
    if groups.size <= 1:
        return positions

    xs = positions[:, 0].astype(np.float64)
    ys = positions[:, 1].astype(np.float64)
    pitch = infer_pitch(xs, ys)
    gutter = _GROUP_GUTTER_PITCHES * pitch

    tiled = positions.copy()
    cursor = 0.0
    for g in groups:
        mask = hs == g
        gx = xs[mask]
        tiled[mask, 0] = (gx - gx.min() + cursor).astype(np.float32)
        # Block width includes the last column's own cell footprint (+pitch).
        cursor += (gx.max() - gx.min() + pitch) + gutter
    logger.info("Tiled %d channel-group blocks side by side to avoid overlap.", groups.size)
    return tiled


def infer_pitch(xs: np.ndarray, ys: np.ndarray) -> float:
    """Smallest positive spacing between distinct x or y coordinates.

    The channel pitch -- for an electrode array, its inter-electrode spacing.
    Defaults to ``1`` when the geometry is degenerate, giving unit-square
    tiling.
    """
    steps: list[float] = []
    for vals in (np.unique(xs), np.unique(ys)):
        if vals.size > 1:
            steps.append(float(np.min(np.diff(vals))))
    positive = [s for s in steps if s > 0]
    return min(positive) if positive else 1.0


def resolve_cell_geometry(
    positions: np.ndarray,
    sizes: np.ndarray | None,
    n_ch: int,
    invert_y: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel display rects ``(x0, y0, side)`` and centers ``(cx, cy)``.

    ``(x0, y0)`` is each cell's lower-left corner. When ``invert_y`` is set the
    layout is flipped about the x-axis so low-``y`` electrodes render at the top
    while each cell stays axis-aligned with its corner anchoring intact.

    Falls back to a unit-spaced sequential layout when every channel shares one
    position, which is what an unmapped source looks like: no geometry loaded,
    so every channel defaulted to the same point.
    """
    xs = positions[:, 0].astype(np.float64)
    ys = positions[:, 1].astype(np.float64)

    if n_ch and np.allclose(xs, xs[0]) and np.allclose(ys, ys[0]):
        # Degenerate (all channels at one point) — lay out a unit-spaced
        # square so the cells don't stack on top of each other.
        logger.info("All channel positions identical; using sequential unit layout for %d channels.", n_ch)
        side = int(np.ceil(np.sqrt(n_ch)))
        xs = (np.arange(n_ch) % side).astype(np.float64)
        ys = (np.arange(n_ch) // side).astype(np.float64)
        cell_sizes = np.ones(n_ch, dtype=np.float64)
    else:
        pitch = infer_pitch(xs, ys)
        if sizes is None:
            cell_sizes = np.full(n_ch, pitch, dtype=np.float64)
        else:
            cell_sizes = np.asarray(sizes, dtype=np.float64).reshape(-1).copy()
            # Zero / missing electrode size → fall back to the pitch.
            cell_sizes[~(cell_sizes > 0)] = pitch

    y0 = -(ys + cell_sizes) if invert_y else ys
    rects = np.column_stack([xs, y0, cell_sizes])
    centers = np.column_stack([xs + cell_sizes / 2.0, y0 + cell_sizes / 2.0])
    return rects, centers


def build_quad_mesh_arrays(rects: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vertex positions ``(n*4, 3)`` and triangle indices ``(n*2, 3)`` for the
    per-channel quads described by ``rects`` (``(n, 3)`` of ``(x0, y0, side)``)."""
    n = rects.shape[0]
    # corner_xy[i, k] = lower-left[i] + corner[k] * side[i]
    corner_xy = rects[:, None, :2] + _CORNERS[None, :, :] * rects[:, None, 2:3]
    positions = np.zeros((n * 4, 3), dtype=np.float32)
    positions[:, :2] = corner_xy.reshape(-1, 2)
    base = (np.arange(n) * 4)[:, None, None]
    indices = (base + _QUAD_TRIS[None, :, :]).reshape(-1, 3).astype(np.uint32)
    return positions, indices
