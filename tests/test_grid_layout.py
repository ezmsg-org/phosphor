"""Cell geometry for per-channel grids.

These are the calculations that decide where a channel's cell lands, and they
are all in the business of degrading gracefully: real electrode geometry is
often missing, partial, or reported by two devices that each numbered their
electrodes from the same origin. Getting that wrong does not raise -- it draws
every channel on top of every other, or silently off screen, which is why the
fallbacks are pinned here rather than left to the widgets.
"""

import numpy as np
import pytest

from phosphor.grid_layout import (
    build_quad_mesh_arrays,
    infer_pitch,
    resolve_cell_geometry,
    tile_by_group,
    tiled_grid_positions,
)

# ---- fallback layout --------------------------------------------------------


def test_tiled_positions_are_square_ish_and_unique():
    pos = tiled_grid_positions(10)
    assert pos.shape == (10, 2)
    assert len({tuple(p) for p in pos}) == 10, "channels must not share a tile"
    # ceil(sqrt(10)) == 4 columns, so x stays under 4 and y under 3.
    assert pos[:, 0].max() < 4 and pos[:, 1].max() < 3


def test_tiled_positions_handles_no_channels():
    assert tiled_grid_positions(0).shape == (0, 2)


# ---- pitch ------------------------------------------------------------------


def test_pitch_is_the_smallest_positive_step():
    xs = np.array([0.0, 400.0, 800.0])
    ys = np.array([0.0, 400.0, 800.0])
    assert infer_pitch(xs, ys) == pytest.approx(400.0)


def test_pitch_of_degenerate_geometry_is_one():
    """One column of channels has no x spacing to measure; a zero pitch would
    collapse every cell to nothing."""
    xs = np.zeros(4)
    ys = np.array([0.0, 2.0, 4.0, 6.0])
    assert infer_pitch(xs, ys) == pytest.approx(2.0)
    assert infer_pitch(np.zeros(4), np.zeros(4)) == 1.0


# ---- grouping ---------------------------------------------------------------


def test_groups_sharing_a_coordinate_range_are_fanned_out():
    """Two devices that each number electrodes from their own origin would
    otherwise render exactly on top of one another."""
    block = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    positions = np.vstack([block, block])
    groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    tiled = tile_by_group(positions, groups)

    first, second = tiled[:4, 0], tiled[4:, 0]
    assert second.min() > first.max(), "the second block must start clear of the first"
    # y is untouched: only the blocks' horizontal placement changes.
    np.testing.assert_array_equal(tiled[:, 1], positions[:, 1])
    # Each block keeps its own internal spacing.
    assert np.ptp(first) == pytest.approx(np.ptp(block[:, 0]))
    assert np.ptp(second) == pytest.approx(np.ptp(block[:, 0]))


def test_a_single_group_is_left_alone():
    """The common case -- one device, or an all-zero group field because no
    geometry was loaded -- must not have its layout rewritten."""
    positions = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    np.testing.assert_array_equal(tile_by_group(positions, np.zeros(2)), positions)


def test_mismatched_group_ids_are_ignored_rather_than_raising():
    """A stale group array is a bad layout, not a crash: the plot is a
    diagnostic tool and is most wanted when the metadata is wrong."""
    positions = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    np.testing.assert_array_equal(tile_by_group(positions, np.array([0, 1, 2])), positions)


# ---- cell geometry ----------------------------------------------------------


def test_cells_default_to_the_pitch_so_they_tile():
    positions = np.array([[0.0, 0.0], [400.0, 0.0], [0.0, 400.0], [400.0, 400.0]])
    rects, centers = resolve_cell_geometry(positions, None, 4, invert_y=False)

    np.testing.assert_allclose(rects[:, 2], 400.0)
    np.testing.assert_allclose(centers[0], [200.0, 200.0])


def test_a_channel_smaller_than_the_pitch_keeps_its_size():
    """An electrode reported as smaller than the array pitch should draw
    smaller, with a gap -- not fill its cell."""
    positions = np.array([[0.0, 0.0], [400.0, 0.0]])
    rects, _ = resolve_cell_geometry(positions, np.array([100.0, 400.0]), 2, invert_y=False)
    assert rects[0, 2] == pytest.approx(100.0)
    assert rects[1, 2] == pytest.approx(400.0)


def test_a_missing_size_falls_back_to_the_pitch():
    """Zero is what an unpopulated size field holds, and a zero-side cell is
    invisible."""
    positions = np.array([[0.0, 0.0], [400.0, 0.0]])
    rects, _ = resolve_cell_geometry(positions, np.array([0.0, 400.0]), 2, invert_y=False)
    assert rects[0, 2] == pytest.approx(400.0)


def test_invert_y_flips_the_layout_without_reshaping_cells():
    positions = np.array([[0.0, 0.0], [0.0, 400.0]])
    upright, up_centers = resolve_cell_geometry(positions, None, 2, invert_y=False)
    flipped, flip_centers = resolve_cell_geometry(positions, None, 2, invert_y=True)

    # Row order reverses on screen...
    assert up_centers[0, 1] < up_centers[1, 1]
    assert flip_centers[0, 1] > flip_centers[1, 1]
    # ...while cells stay the same size and keep their x.
    np.testing.assert_allclose(flipped[:, 2], upright[:, 2])
    np.testing.assert_allclose(flipped[:, 0], upright[:, 0])


def test_every_channel_at_one_point_lays_out_sequentially():
    """What an unmapped source looks like: no geometry, so every channel
    defaulted to the same coordinate. Drawn verbatim it is one cell deep."""
    positions = np.zeros((9, 2))
    rects, centers = resolve_cell_geometry(positions, None, 9, invert_y=False)

    assert len({tuple(c) for c in centers}) == 9, "cells must not stack"
    np.testing.assert_allclose(rects[:, 2], 1.0)


# ---- mesh -------------------------------------------------------------------


def test_quad_mesh_covers_each_cell_with_two_triangles():
    rects = np.array([[0.0, 0.0, 2.0], [10.0, 0.0, 4.0]])
    positions, indices = build_quad_mesh_arrays(rects)

    assert positions.shape == (8, 3)
    assert indices.shape == (4, 3)
    # First quad spans its own rect, corner to corner.
    np.testing.assert_allclose(positions[:4, 0].min(), 0.0)
    np.testing.assert_allclose(positions[:4, 0].max(), 2.0)
    # Second quad is offset and larger, and indexes its own vertices only.
    np.testing.assert_allclose(positions[4:, 0].min(), 10.0)
    np.testing.assert_allclose(positions[4:, 0].max(), 14.0)
    assert indices[2:].min() >= 4


def test_quad_mesh_of_no_cells_is_empty_not_malformed():
    positions, indices = build_quad_mesh_arrays(np.zeros((0, 3)))
    assert positions.shape == (0, 3)
    assert indices.shape == (0, 3)
