"""GPU-accelerated real-time scatter/heatmap widget."""

from __future__ import annotations

from dataclasses import dataclass

import fastplotlib as fpl
import numpy as np
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QLabel, QToolTip, QVBoxLayout, QWidget

from .scatter_buffer import ScatterBuffer

__all__ = ["ScatterConfig", "ScatterWidget"]


@dataclass
class ScatterConfig:
    positions: np.ndarray  # (n_channels, 2) or (n_channels, 3)
    cmap: str = "viridis"
    modulate_color: bool = True
    modulate_size: bool = False
    marker_size: float = 10.0  # base marker size (screen pixels)
    size_range: tuple[float, float] = (4.0, 20.0)
    vmin: float | None = None
    vmax: float | None = None
    channel_labels: list[str] | None = None


class ScatterWidget(QWidget):
    """Embeddable QWidget that renders a GPU-accelerated scatter heatmap.

    Each channel has a fixed 2D position; incoming scalar data modulates the
    color and/or size of the marker at that position.

    Usage::

        widget = ScatterWidget(ScatterConfig(
            positions=electrode_positions,  # (n_ch, 2)
            cmap="viridis",
            modulate_color=True,
        ))
        widget.show()
        widget.push_data(values)  # (n_channels,)
    """

    def __init__(self, config: ScatterConfig, parent: QWidget | None = None):
        super().__init__(parent)
        self._config = config
        n_channels = config.positions.shape[0]

        # Use only xy for rendering
        self._positions_2d = config.positions[:, :2].astype(np.float32)

        # Buffer
        self._buffer = ScatterBuffer(
            n_channels=n_channels,
            vmin=config.vmin,
            vmax=config.vmax,
        )

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # fastplotlib figure
        self._figure = fpl.Figure()
        self._subplot = self._figure[0, 0]
        self._fpl_widget = self._figure.show()
        layout.addWidget(self._fpl_widget)

        # Value label overlay
        self._value_label = QLabel(self._fpl_widget)
        self._value_label.setStyleSheet(
            "background: rgba(25,25,30,200); color: #b4b4b4;"
            " padding: 2px 6px; font-size: 9pt;"
            " font-family: 'Menlo', 'Consolas', 'DejaVu Sans Mono', monospace;"
        )
        self._value_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._value_label.hide()

        # Graphics
        self._scatter = None
        self._setup_graphics()

        # Disable pan/zoom, axes, and default title
        self._subplot.controller = None
        self._subplot.axes.visible = False
        self._subplot.title.visible = False

        # Event handlers
        renderer = self._subplot.renderer
        renderer.add_event_handler(self._on_pointer_move_event, "pointer_move")

        # Animation callback
        self._figure.add_animations(lambda: self._animation_callback())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push_data(self, data: np.ndarray, timestamps=None) -> None:
        """Push scalar values. Shape ``(n_channels,)`` or ``(n_samples, n_channels)``."""
        self._buffer.push_data(data)

    # ------------------------------------------------------------------
    # Graphics setup
    # ------------------------------------------------------------------

    def _setup_graphics(self) -> None:
        positions = self._positions_2d
        n = positions.shape[0]

        # Build (n, 3) positions for fpl (z=0)
        pts = np.zeros((n, 3), dtype=np.float32)
        pts[:, :2] = positions

        self._scatter = self._subplot.add_scatter(
            pts,
            cmap=self._config.cmap,
            cmap_transform=np.zeros(n, dtype=np.float32),
            sizes=self._config.marker_size,
            size_space="screen",
        )

    # ------------------------------------------------------------------
    # Animation callback
    # ------------------------------------------------------------------

    def _animation_callback(self) -> None:
        values = self._buffer.consume()
        if values is None:
            return

        vmin, vmax = self._buffer.vmin, self._buffer.vmax
        span = max(vmax - vmin, 1e-12)
        norm = np.nan_to_num(np.clip((values - vmin) / span, 0.0, 1.0), nan=0.0).astype(np.float32)

        if self._config.modulate_color:
            self._scatter.cmap.transform = norm
        if self._config.modulate_size:
            lo, hi = self._config.size_range
            self._scatter.sizes = lo + norm * (hi - lo)

        self._subplot.auto_scale(maintain_aspect=True, zoom=0.9)

    # ------------------------------------------------------------------
    # Tooltip on hover
    # ------------------------------------------------------------------

    def _on_pointer_move_event(self, event) -> None:
        world = self._subplot.map_screen_to_world(event)
        if world is None:
            self._value_label.hide()
            return

        wx, wy = float(world[0]), float(world[1])
        dists = (self._positions_2d[:, 0] - wx) ** 2 + (self._positions_2d[:, 1] - wy) ** 2
        idx = int(np.argmin(dists))

        labels = self._config.channel_labels
        label = labels[idx] if labels and idx < len(labels) else f"Ch {idx}"
        value = self._buffer._current[idx]
        self._value_label.setText(f"{label}: {value:.4g}")
        self._value_label.adjustSize()
        self._value_label.show()

        QToolTip.showText(
            self._fpl_widget.mapToGlobal(QPoint(int(event.x), int(event.y))),
            f"{label}: {value:.4g}",
            self._fpl_widget,
        )
