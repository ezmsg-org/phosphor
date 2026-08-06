"""GPU-accelerated 2D grid of per-channel waveform plots.

A reusable companion to :mod:`channel_grid`. Where the channel grid colors one
square per channel by a scalar value, this widget draws a *mini line-plot* per
channel — a rolling history of fixed-length waveforms plus their running mean —
at each channel's real ``(x, y)`` electrode position.

It is deliberately app-agnostic so two viewers can share it:

* **evoked potentials** — each marker-locked epoch window is one waveform per
  channel (``push_epoch``).
* **spike waveforms** (future) — each detected spike snippet is a waveform; the
  synchronized ``push_epoch`` model can be extended to per-channel pushes later.

Layout, cell geometry, and the camera-anchoring background mesh are shared with
the channel grid via :mod:`grid_layout`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import fastplotlib as fpl
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .decimate import minmax_decimate, minmax_decimate_x, plan_minmax_decimation
from .grid_layout import build_quad_mesh_arrays, resolve_cell_geometry
from .trace_grid_buffer import TraceGridBuffer

logger = logging.getLogger(__name__)

__all__ = ["TraceGridConfig", "TraceGridWidget"]

RGBA = tuple[float, float, float, float]

# World-space viewport rect shared for pan/zoom sync: (xmin, xmax, ymin, ymax).
Viewport = tuple[float, float, float, float]


def _viewport_differs(a: Viewport, b: Viewport, *, rtol: float = 1e-4) -> bool:
    """True if two world rects differ by more than a small relative tolerance.

    Guards the per-frame camera poll against float jitter that would otherwise
    emit a viewport_changed every frame even when the user isn't interacting.
    """
    span = max(abs(b[1] - b[0]), abs(b[3] - b[2]), 1e-9)
    return any(abs(x - y) > rtol * span for x, y in zip(a, b))


@dataclass
class TraceGridConfig:
    positions: np.ndarray
    """``(n_ch, 2)`` array of electrode ``(x, y)`` lower-left corners."""

    n_samples: int
    """Number of samples per waveform (epoch window length)."""

    sizes: np.ndarray | None = None
    """``(n_ch,)`` per-channel cell side length; ``None`` → inferred pitch."""

    channel_labels: list[str] | None = None
    history: int = 10
    """How many recent waveforms to retain and overlay per channel."""

    show_individual: bool = True
    show_mean: bool = True

    show_error: bool = False
    """Draw a +/- one standard deviation band around the mean.

    Needs ``track_statistics``; meaningless without a mean to bound."""

    track_statistics: bool = True
    """Accumulate the running mean and standard deviation over every waveform.

    Off for a stack with no meaningful average -- action potentials from
    possibly-different units -- where it costs two float64 arrays and a pass
    over every arrival to compute a number nobody draws."""

    age_fade: bool = True
    """Fade older waveforms.

    Costs a rewrite of the whole colour buffer on every arrival, because every
    retained waveform's age changes when one arrives. Worth it at evoked rates
    and not at spike rates, where a flat colour keeps a push to one slot's
    worth of vertices."""

    autoscale: bool = True
    """Auto-fit the shared amplitude range to the retained waveforms."""

    y_min: float = -100.0
    y_max: float = 100.0
    """Fixed amplitude range used when ``autoscale`` is False."""

    cell_pad_frac: float = 0.08
    """Fraction of each cell reserved as inner padding around the waveform."""

    individual_color: RGBA = (0.45, 0.62, 0.95, 1.0)
    mean_color: RGBA = (1.0, 0.82, 0.25, 1.0)
    error_color: RGBA = (1.0, 0.82, 0.25, 0.45)
    cell_color: RGBA = (0.12, 0.12, 0.15, 1.0)
    individual_thickness: float = 1.0
    mean_thickness: float = 2.5
    error_thickness: float = 1.0

    invert_y: bool = False
    """Flip so low-``y`` electrodes render at the top (Blackrock ``.cmp``
    row-0-at-bottom convention). Matches :class:`ChannelGridWidget`'s default
    of ``False`` for impedance; callers map as they prefer."""

    x_unit: str = "s"
    x_extent: tuple[float, float] | None = None
    """Optional ``(t0, t1)`` of the waveform window, used only for the tooltip."""

    display_max_points: int | None = None
    """Max vertices drawn per waveform. ``None`` → derive from the monitor
    (~2 samples per horizontal pixel of a single cell, which is all a min/max
    envelope can show). Waveforms are min/max-decimated to this budget for
    rendering only; the stored history, running mean, and autoscale stay at full
    resolution. See :mod:`phosphor.decimate`."""


class TraceGridWidget(QtWidgets.QWidget):
    """Embeddable grid: one mini waveform-plot per channel at its electrode
    position, overlaying a rolling history of waveforms and their mean."""

    # Emitted when the user pans/zooms the camera, carrying the new visible
    # world rect (xmin, xmax, ymin, ymax). Lets a companion view (e.g. the SNR
    # heatmap) follow the same electrode region. Not emitted for programmatic
    # :meth:`set_viewport` changes, so linked views don't echo each other.
    viewport_changed = QtCore.Signal(object)

    def __init__(self, config: TraceGridConfig, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._config = config
        self._n_ch = config.positions.shape[0]
        self._n_samples = int(config.n_samples)
        self._show_individual = bool(config.show_individual)
        self._show_mean = bool(config.show_mean)
        self._show_error = bool(config.show_error) and bool(config.track_statistics)
        self._autoscale = bool(config.autoscale)
        self._y_min = float(config.y_min)
        self._y_max = float(config.y_max)

        # Per-channel display rects (x0, y0, side) and centers, shared layout.
        self._rects, self._centers = resolve_cell_geometry(config.positions, config.sizes, self._n_ch, config.invert_y)
        # Precompute the per-channel x coordinate of each sample inside its cell.
        self._x_line = self._compute_x_line()  # (n_ch, n_samples)

        # Display-only min/max envelope decimation: shrink the per-frame vertex
        # buffer (and its GPU mirror) without touching the stored/analyzed data.
        budget = self._resolve_display_max_points()
        self._dec_plan = plan_minmax_decimation(self._n_samples, budget)
        # Precompute the decimated x line once; it never changes.
        self._x_line_dec = minmax_decimate_x(self._x_line, self._dec_plan)
        # Always log the decision (even when inactive) so it's clear whether
        # decimation ran — a silent no-op is indistinguishable from "not wired".
        if self._dec_plan.active:
            logger.info(
                "TraceGrid: min/max-decimating %d samples → %d display points per waveform (%.1fx, budget=%d).",
                self._n_samples,
                self._dec_plan.n_out,
                self._n_samples / self._dec_plan.n_out,
                budget,
            )
        else:
            logger.info(
                "TraceGrid: no decimation — %d samples already within display budget=%d.",
                self._n_samples,
                budget,
            )

        # Retention and running statistics live in the buffer; this widget only
        # decides how to draw them.
        self._buffer = TraceGridBuffer(
            self._n_ch,
            self._n_samples,
            config.history,
            track_statistics=config.track_statistics,
        )
        self._graphics_version = -1  # buffer version the current graphics were built for
        self._dirty = True  # flag "the data changed, the screen hasn't caught up yet"
        self._needs_autoscale_camera = True

        # Pan/zoom-sync state. ``_last_viewport`` is the camera rect at the last
        # poll; ``_suppress_emit`` absorbs the frames right after a programmatic
        # set_viewport (the aspect-preserving camera settles over a frame or two)
        # so a linked view's change isn't echoed back as a user gesture.
        self._last_viewport: Viewport | None = None
        self._suppress_emit = False

        qt_layout = QtWidgets.QVBoxLayout(self)
        qt_layout.setContentsMargins(0, 0, 0, 0)

        self._figure = fpl.Figure()
        self._subplot = self._figure[0, 0]
        self._fpl_widget = self._figure.show()
        qt_layout.addWidget(self._fpl_widget)

        # Faint background quad per cell — frames each channel and anchors the
        # camera (the waveform lines may be all-NaN at first, giving auto_scale
        # nothing to fit). Reuses the channel-grid quad tessellation.
        bg_positions, bg_indices = build_quad_mesh_arrays(self._rects)
        bg_colors = np.tile(np.asarray(config.cell_color, dtype=np.float32), (bg_positions.shape[0], 1))
        self._bg_mesh = self._subplot.add_mesh(bg_positions, bg_indices, mode="basic", colors=bg_colors)

        self._indiv_ml = None
        self._mean_ml = None
        self._error_ml = None

        # NOTE: assigning None does *not* disable interaction — fastplotlib's
        # setter substitutes a default PanZoomController, so the grid keeps
        # mouse/trackpad pan + zoom (with maintain_aspect, cells stay square).
        # We observe that camera below to broadcast viewport_changed.
        self._subplot.controller = None
        self._subplot.axes.visible = False
        self._subplot.title.visible = False
        try:
            self._subplot.tooltip.enabled = False
        except AttributeError:
            pass

        self._tooltip = QtWidgets.QLabel(self._fpl_widget)
        self._tooltip.setStyleSheet(
            "background: rgba(25,25,30,220); color: #e8e8e8;"
            " padding: 8px 14px; font-size: 16pt;"
            " font-family: 'Menlo','Consolas','DejaVu Sans Mono',monospace;"
            " border: 1px solid rgba(120,120,120,160);"
        )
        self._tooltip.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._tooltip.hide()

        self._figure.add_animations(lambda: self._animation_callback())
        renderer = self._subplot.renderer
        renderer.add_event_handler(self._on_pointer_move, "pointer_move")
        renderer.add_event_handler(self._on_pointer_leave, "pointer_leave")
        renderer.add_event_handler(self._on_double_click, "double_click")

    # ---- Public API ----------------------------------------------------

    def push_epoch(self, data: np.ndarray) -> None:
        """Append one waveform per channel. *data* shape ``(n_samples, n_ch)``.

        Rolls the per-channel history so the new waveform becomes the most
        recent; the oldest is dropped once ``history`` is reached.
        """
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] != self._n_samples or arr.shape[1] != self._n_ch:
            logger.warning(
                "push_epoch shape %s != (n_samples=%d, n_ch=%d); ignoring.",
                arr.shape,
                self._n_samples,
                self._n_ch,
            )
            return
        self._buffer.push(arr.T)  # (n_ch, n_samples)
        self._dirty = True

    def set_history(self, history: int) -> None:
        """Change how many waveforms are overlaid.

        Retained waveforms survive where they still fit, and the running mean
        is untouched: how many are drawn has nothing to do with how many are
        averaged.
        """
        self._buffer.set_history(history)
        self._dirty = True

    def set_show_error(self, show: bool) -> None:
        self._show_error = bool(show) and self._buffer.track_statistics
        self._dirty = True

    def set_show_individual(self, show: bool) -> None:
        self._show_individual = bool(show)
        self._dirty = True

    def set_show_mean(self, show: bool) -> None:
        self._show_mean = bool(show)
        self._dirty = True

    def set_autoscale(self, enabled: bool) -> None:
        self._autoscale = bool(enabled)
        self._dirty = True

    def set_y_range(self, y_min: float, y_max: float) -> None:
        if y_min > y_max:
            y_min, y_max = y_max, y_min
        self._autoscale = False
        self._y_min = float(y_min)
        self._y_max = float(y_max)
        self._dirty = True

    def clear(self) -> None:
        """Drop all retained waveforms (e.g. before swapping acquisition source).

        Repaints immediately (do what _animation_callback does):
        fastplotlib renders on-demand, so just flagging
        ``_dirty`` would leave the old traces on screen until the next epoch adds
        graphics and triggers a draw. We remove the line graphics now and request
        a frame so the cleared grid is visible the instant the event is switched.
        """
        self._buffer.clear()
        self._dirty = False
        self._rebuild_lines()  # nothing retained → deletes the lines, recreates none
        self._request_draw()

    def _request_draw(self) -> None:
        """Ask the canvas to present a fresh frame now (on-demand renderer)."""
        figure = getattr(self, "_figure", None)
        canvas = getattr(figure, "canvas", None) if figure is not None else None
        request = getattr(canvas, "request_draw", None)
        if callable(request):
            request()

    def close_figure(self) -> None:
        """Tear down the fastplotlib figure deterministically so rendercanvas
        releases its Qt callbacks before the C++ widget is destroyed."""
        figure = getattr(self, "_figure", None)
        if figure is None:
            return
        try:
            figure.close()
        except Exception:
            logger.exception("fastplotlib figure close raised; ignoring")
        self._figure = None

    @property
    def history(self) -> int:
        return self._buffer.history

    @property
    def show_individual(self) -> bool:
        return self._show_individual

    @property
    def show_mean(self) -> bool:
        return self._show_mean

    @property
    def autoscale(self) -> bool:
        return self._autoscale

    @property
    def y_range(self) -> tuple[float, float]:
        return self._y_min, self._y_max

    # ---- Geometry ------------------------------------------------------

    def _resolve_display_max_points(self) -> int:
        """Vertex budget per waveform for min/max-envelope rendering.

        Honours an explicit ``config.display_max_points``; otherwise derives it
        from the monitor — ~2 samples per horizontal pixel of one cell is all an
        envelope can resolve. Assumes the grid may span the full screen width
        (over-provisions slightly, ignoring side docks — acceptable, and a
        future resize-aware variant can recompute from the live canvas). Falls
        back to a fixed budget when no screen can be queried.

        **Multi-monitor:** we size for the widest to be safe.
        """
        configured = self._config.display_max_points
        if configured is not None:
            return max(2, int(configured))

        fallback = 1500
        n_columns = max(1, int(np.unique(self._rects[:, 0]).size))
        try:
            screens = QtGui.QGuiApplication.screens()
            phys_w = max(
                (float(s.geometry().width()) * float(s.devicePixelRatio()) for s in screens),
                default=0.0,
            )
        except Exception:
            logger.debug("TraceGrid: could not query screens for decimation target.", exc_info=True)
            return fallback
        if phys_w <= 0:
            return fallback
        per_cell_px = phys_w / n_columns
        return max(2, int(2.0 * per_cell_px))

    def _compute_x_line(self) -> np.ndarray:
        x0 = self._rects[:, 0]
        side = self._rects[:, 2]
        pad = self._config.cell_pad_frac
        inner = side * (1.0 - 2.0 * pad)
        if self._n_samples > 1:
            frac = np.arange(self._n_samples, dtype=np.float64) / (self._n_samples - 1)
        else:
            frac = np.zeros(self._n_samples, dtype=np.float64)
        # (n_ch, n_samples)
        return (x0 + pad * side)[:, None] + frac[None, :] * inner[:, None]

    def _map_y(self, values: np.ndarray) -> np.ndarray:
        """Map amplitude *values* (..., n_ch, n_samples) into per-cell y."""
        y0 = self._rects[:, 1]
        side = self._rects[:, 2]
        pad = self._config.cell_pad_frac
        inner = side * (1.0 - 2.0 * pad)
        span = max(self._y_max - self._y_min, 1e-12)
        frac = (values - self._y_min) / span
        # Allow a little overflow past the cell, but not into neighbours.
        frac = np.clip(frac, -0.1, 1.1)
        return (y0 + pad * side)[..., :, None] + frac * inner[..., :, None]

    # ---- Rendering -----------------------------------------------------

    def _animation_callback(self) -> None:
        if self._dirty:
            self._dirty = False
            if self._autoscale:
                self._recompute_autoscale()
            self._refresh_lines()

        if self._needs_autoscale_camera:
            self._needs_autoscale_camera = False
            self._subplot.auto_scale(maintain_aspect=True, zoom=0.95)

        self._poll_viewport()

    # ---- Pan/zoom viewport sync ----------------------------------------

    def _read_viewport(self) -> Viewport:
        """Current visible world rect from the camera state."""
        state = self._subplot.camera.get_state()
        zoom = state["zoom"] or 1.0
        w = state["width"] / zoom
        h = state["height"] / zoom
        cx, cy = float(state["position"][0]), float(state["position"][1])
        return (cx - w / 2.0, cx + w / 2.0, cy - h / 2.0, cy + h / 2.0)

    def _poll_viewport(self) -> None:
        """Emit viewport_changed when the camera moved from user interaction.

        Runs every animation frame. A change originating from set_viewport is
        absorbed (``_suppress_emit``) so linked views don't ping-pong; once the
        rect stops changing we clear the flag and resume emitting user gestures.
        """
        rect = self._read_viewport()
        if self._last_viewport is not None and not _viewport_differs(rect, self._last_viewport):
            self._suppress_emit = False
            return
        self._last_viewport = rect
        if self._suppress_emit:
            return
        self.viewport_changed.emit(rect)

    def current_viewport(self) -> Viewport:
        """The visible world rect (xmin, xmax, ymin, ymax); for initial sync."""
        return self._read_viewport()

    def set_viewport(self, rect: Viewport) -> None:
        """Frame *rect* (world coords), matching a linked view's pan/zoom.

        maintain_aspect is preserved, so the region is centered and fit to this
        widget's shape (it may reveal a little extra along the wider axis rather
        than distorting cells). Does not re-emit viewport_changed.
        """
        xmin, xmax, ymin, ymax = rect
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        cam = self._subplot.camera
        state = cam.get_state()
        state["position"] = (cx, cy, float(state["position"][2]))
        state["width"] = max(xmax - xmin, 1e-9)
        state["height"] = max(ymax - ymin, 1e-9)
        state["zoom"] = 1.0
        cam.set_state(state)
        self._suppress_emit = True
        self._needs_autoscale_camera = False  # user/linked view now owns the camera
        self._request_draw()

    def reset_view(self) -> None:
        """Refit the camera to all electrodes (double-click). The resulting
        viewport is broadcast via the normal poll, so a linked view follows."""
        self._needs_autoscale_camera = True
        self._suppress_emit = False  # a reset is a user action; always broadcast
        self._request_draw()

    def _on_double_click(self, _event) -> None:
        self.reset_view()

    def _recompute_autoscale(self) -> None:
        valid = self._buffer.recent()
        if valid.size == 0 or not np.any(np.isfinite(valid)):
            return
        lo = float(np.nanmin(valid))
        hi = float(np.nanmax(valid))
        if not np.isfinite(lo) or not np.isfinite(hi):
            return
        if hi <= lo:
            hi = lo + 1.0
        margin = 0.05 * (hi - lo)
        self._y_min = lo - margin
        self._y_max = hi + margin

    # ---- Drawing -------------------------------------------------------
    #
    # Two paths. A waveform arriving changes numbers, not shapes, so its slot is
    # written into the graphics that already exist. Only a change of shape --
    # history, or a toggle that adds or removes lines -- recreates them. Doing
    # the recreate on every arrival is what made this unusable at spike rates:
    # it tears down and re-uploads every retained waveform to draw one new one.

    def _slot_rows(self, slot: int) -> slice:
        """Rows of the individual-waveform graphic belonging to one ring slot.

        Lines are laid out slot-major, so a slot's channels are contiguous and
        one arrival writes one block.
        """
        return slice(slot * self._n_ch, (slot + 1) * self._n_ch)

    def _slot_alpha(self) -> np.ndarray:
        """Per-slot opacity, brightest for the newest, zero for never-written.

        Keyed on age rather than position because nothing is reordered when a
        waveform arrives -- see :attr:`TraceGridBuffer.ages`.
        """
        buf = self._buffer
        ages = buf.ages
        if not self._config.age_fade:
            alpha = np.full(buf.history, 0.9, dtype=np.float32)
        else:
            ramp = np.linspace(0.9, 0.15, buf.history, dtype=np.float32)
            alpha = ramp[ages]
        return np.where(ages < buf.n_retained, alpha, 0.0).astype(np.float32)

    def _individual_positions(self) -> tuple[np.ndarray, int]:
        """Vertex positions for every slot, ``(history * n_ch, m, 3)``."""
        buf = self._buffer
        # Decimate raw amplitudes (peak-preserving) before the cell mapping;
        # x is the matching precomputed envelope. Full-res data is untouched.
        traces = minmax_decimate(buf.traces, self._dec_plan)
        m = traces.shape[-1]
        pos = np.empty((buf.history, self._n_ch, m, 3), dtype=np.float32)
        pos[..., 0] = np.broadcast_to(self._x_line_dec, (buf.history, self._n_ch, m))
        pos[..., 1] = self._map_y(traces)
        pos[..., 2] = 0.0
        return pos.reshape(buf.history * self._n_ch, m, 3), m

    def _individual_colors(self) -> np.ndarray:
        """One RGBA per line, ``(history * n_ch, 4)``, faded by slot age.

        What ``add_multi_line`` takes. Writing into an existing graphic needs
        the per-vertex form instead -- see :meth:`_expand_colors`.
        """
        buf = self._buffer
        rgba = np.tile(np.asarray(self._config.individual_color, dtype=np.float32), (buf.history, self._n_ch, 1))
        rgba[..., 3] = self._slot_alpha()[:, None]
        return rgba.reshape(-1, 4)

    @staticmethod
    def _expand_colors(graphic, per_line: np.ndarray) -> np.ndarray:
        """Per-line colours as the flat per-vertex buffer a graphic holds.

        fastplotlib takes one colour per line when a graphic is built and then
        stores one per vertex, so an in-place recolour has to repeat each line's
        colour across its own run. The stride comes from the buffer rather than
        the point count because it also covers whatever separator vertices the
        renderer inserted between lines.
        """
        total = graphic.colors.value.shape[0]
        stride = total // max(per_line.shape[0], 1)
        return np.repeat(per_line, stride, axis=0)[:total]

    def _summary_positions(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Mean and +/- one standard deviation, decimated for display.

        Both are computed at full resolution and reduced only for drawing, so
        what is averaged is never what happened to fit on screen.
        """
        stats = self._buffer.statistics()
        if stats is None:
            return None, None
        mean, std = stats
        mean_pos = self._curve_positions(minmax_decimate(mean, self._dec_plan))
        if not self._show_error or std is None or not np.any(np.isfinite(std)):
            return mean_pos, None
        band = np.concatenate([mean - std, mean + std], axis=0)
        return mean_pos, self._curve_positions(minmax_decimate(band, self._dec_plan))

    def _curve_positions(self, curve: np.ndarray) -> np.ndarray:
        """One line per row of *curve*, laid into the cells."""
        n_lines, m = curve.shape[0], curve.shape[-1]
        pos = np.empty((n_lines, m, 3), dtype=np.float32)
        # The band is two curves per channel, so x tiles rather than broadcasts.
        reps = n_lines // self._n_ch
        pos[..., 0] = np.tile(self._x_line_dec, (reps, 1))
        pos[..., 1] = self._map_y(curve)
        pos[..., 2] = 0.0
        return pos

    def _graphics_shape(self) -> tuple:
        """Which graphics should exist and how big. A change means recreate.

        The flags say *should this be drawn at all*, not just whether the user
        asked for it, because the widget is built before its first waveform
        arrives. A key that described only sizes and toggles would settle on
        "nothing to draw" against the empty buffer and never change when data
        turned up -- every arrival would take the in-place path, find no
        graphics to write into, and draw nothing.
        """
        buf = self._buffer
        has_data = buf.n_retained > 0
        return (
            buf.history,
            self._n_ch,
            self._dec_plan.n_out,
            self._show_individual and has_data,
            self._show_mean and has_data and buf.track_statistics,
            # A spread needs a second waveform, so the band appears later than
            # the mean does.
            self._show_error and buf.track_statistics and buf.n_seen > 1,
        )

    def _refresh_lines(self) -> None:
        """Draw the current buffer, recreating graphics only if the shape moved."""
        if self._graphics_version != self._graphics_shape():
            self._rebuild_lines()
            return
        self._update_lines()

    def _update_lines(self) -> None:
        """Write new numbers into the graphics that already exist."""
        buf = self._buffer
        if self._indiv_ml is not None and buf.n_retained:
            pos, _ = self._individual_positions()
            self._indiv_ml.data[:] = pos
            if self._config.age_fade:
                self._indiv_ml.colors[:] = self._expand_colors(self._indiv_ml, self._individual_colors())
        mean_pos, band_pos = self._summary_positions()
        if self._mean_ml is not None and mean_pos is not None:
            self._mean_ml.data[:] = mean_pos
        if self._error_ml is not None and band_pos is not None:
            self._error_ml.data[:] = band_pos

    def _rebuild_lines(self) -> None:
        """Recreate the waveform, mean, and error graphics from scratch.

        Only for a change of shape: the number of lines, their length, or which
        of them exist at all.
        """
        for name in ("_indiv_ml", "_mean_ml", "_error_ml"):
            graphic = getattr(self, name, None)
            if graphic is not None:
                self._subplot.delete_graphic(graphic)
                setattr(self, name, None)

        buf = self._buffer
        self._graphics_version = self._graphics_shape()
        if buf.n_retained == 0:
            return

        if self._show_individual:
            pos, _ = self._individual_positions()
            self._indiv_ml = self._subplot.add_multi_line(
                pos,
                colors=self._individual_colors(),
                thickness=self._config.individual_thickness,
            )
            self._indiv_ml.visible = self._show_individual

        mean_pos, band_pos = self._summary_positions()
        # Drawn before the mean so the band sits behind the line it bounds.
        if band_pos is not None:
            self._error_ml = self._subplot.add_multi_line(
                band_pos,
                colors=np.tile(np.asarray(self._config.error_color, dtype=np.float32), (band_pos.shape[0], 1)),
                thickness=self._config.error_thickness,
            )
            self._error_ml.visible = self._show_error
        if self._show_mean and mean_pos is not None:
            self._mean_ml = self._subplot.add_multi_line(
                mean_pos,
                colors=np.tile(np.asarray(self._config.mean_color, dtype=np.float32), (self._n_ch, 1)),
                thickness=self._config.mean_thickness,
            )
            self._mean_ml.visible = self._show_mean

    # ---- Tooltip / hit-testing -----------------------------------------

    def _channel_at(self, wx: float, wy: float) -> int:
        x0 = self._rects[:, 0]
        y0 = self._rects[:, 1]
        side = self._rects[:, 2]
        inside = (wx >= x0) & (wx <= x0 + side) & (wy >= y0) & (wy <= y0 + side)
        hits = np.flatnonzero(inside)
        return int(hits[0]) if hits.size else -1

    def _on_pointer_move(self, event) -> None:
        world = self._subplot.map_screen_to_world(event)
        if world is None:
            self._tooltip.hide()
            return
        ch_idx = self._channel_at(float(world[0]), float(world[1]))
        if ch_idx < 0:
            self._tooltip.hide()
            return
        labels = self._config.channel_labels
        label = labels[ch_idx] if labels and ch_idx < len(labels) else f"ch{ch_idx}"
        text = f"{label}\n{self._buffer.n_retained} shown / {self._buffer.n_seen} total"
        self._tooltip.setText(text)
        self._tooltip.adjustSize()
        x = int(getattr(event, "x", 0)) + 14
        y = int(getattr(event, "y", 0)) + 14
        max_x = self._fpl_widget.width() - self._tooltip.width() - 4
        max_y = self._fpl_widget.height() - self._tooltip.height() - 4
        self._tooltip.move(min(x, max_x), min(y, max_y))
        self._tooltip.show()

    def _on_pointer_leave(self, _event) -> None:
        self._tooltip.hide()
