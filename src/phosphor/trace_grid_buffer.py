"""Retention and statistics for a grid of per-channel waveforms.

The CPU side of :class:`~phosphor.trace_grid.TraceGridWidget`, split out for the
same reason :class:`~phosphor.sweep_buffer.SweepBuffer` is: it is pure numpy, so
it can be exercised without a GPU, and that is where the behaviour that is easy
to get wrong actually lives.

Two things it does that a naive buffer does not:

**Writes into a ring.** A new waveform overwrites one slot. The alternative --
rolling the whole buffer so the newest is always at index 0 -- copies every
retained sample on every arrival, which is affordable for an evoked potential
arriving once a second and not for spikes arriving hundreds of times a second.
Nothing is reordered, so a caller draws the slots where they lie and asks
:attr:`ages` which is newest.

**Counts every waveform, not just the retained ones.** ``history`` is how many
to *draw*; the running mean and standard deviation come from every waveform ever
pushed. An evoked response is normally the average of hundreds of sweeps while
only a handful are worth overlaying, and tying the two together would force a
caller to retain hundreds of traces to average hundreds of traces.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["TraceGridBuffer"]


class TraceGridBuffer:
    """Ring of the last ``history`` waveforms per channel, plus running stats.

    :param n_channels: Channels drawn, one cell each.
    :param n_samples: Samples per waveform.
    :param history: How many recent waveforms to retain for drawing.
    :param track_statistics: Accumulate the running mean and standard
        deviation. Off for data with no meaningful average -- a stack of
        action potentials from possibly-different units -- where it would cost
        two float64 arrays and a pass over every arrival to compute a number
        nobody displays.
    """

    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        history: int = 10,
        *,
        track_statistics: bool = True,
    ) -> None:
        self.n_channels = int(n_channels)
        self.n_samples = int(n_samples)
        self._history = max(1, int(history))
        self.track_statistics = bool(track_statistics)
        # Bumped when the drawn shape changes, so a renderer knows its graphics
        # are the wrong size. Data changes bump `updates` instead, which is the
        # cheap path: same shape, new numbers.
        self.version = 0
        self.updates = 0
        self._allocate()

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------

    def _allocate(self) -> None:
        self.traces = np.full((self._history, self.n_channels, self.n_samples), np.nan, dtype=np.float32)
        self._cursor = 0
        self._n_retained = 0
        self._reset_statistics()

    def _reset_statistics(self) -> None:
        # float64 because these accumulate without bound: a session's worth of
        # sums in float32 loses the low bits of every later arrival.
        shape = (self.n_channels, self.n_samples)
        self._count = np.zeros(shape, dtype=np.float64) if self.track_statistics else None
        self._sum = np.zeros(shape, dtype=np.float64) if self.track_statistics else None
        self._sumsq = np.zeros(shape, dtype=np.float64) if self.track_statistics else None
        self.n_seen = 0

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def push(self, trace: np.ndarray) -> None:
        """Retain one waveform per channel, shape ``(n_channels, n_samples)``.

        Channels with nothing to contribute pass NaN, which is retained as a gap
        and left out of the statistics rather than counted as a zero.
        """
        arr = np.asarray(trace, dtype=np.float32)
        if arr.shape != (self.n_channels, self.n_samples):
            raise ValueError(f"push expects {(self.n_channels, self.n_samples)}, got {arr.shape}")

        self.traces[self._cursor] = arr
        self._cursor = (self._cursor + 1) % self._history
        self._n_retained = min(self._n_retained + 1, self._history)

        if self.track_statistics:
            finite = np.isfinite(arr)
            contribution = np.where(finite, arr, 0.0).astype(np.float64)
            self._count += finite
            self._sum += contribution
            self._sumsq += contribution * contribution
        self.n_seen += 1
        self.updates += 1

    def clear(self) -> None:
        """Drop every retained waveform and forget the running statistics.

        Both together: a caller clears because what came before no longer
        describes what comes next -- a new source, changed conditioning -- and
        a mean carried across that boundary would average two different things.
        """
        self.traces[:] = np.nan
        self._cursor = 0
        self._n_retained = 0
        self._reset_statistics()
        self.updates += 1

    def set_history(self, history: int) -> None:
        """Change how many waveforms are retained for drawing.

        Keeps as many of the newest as still fit, so growing the history does
        not blank what is on screen and shrinking it drops the oldest. The
        running statistics are untouched: how many are drawn has nothing to do
        with how many are averaged.
        """
        history = max(1, int(history))
        if history == self._history:
            return
        keep = min(self._n_retained, history)
        newest = self.recent(keep)
        self._history = history
        self.traces = np.full((history, self.n_channels, self.n_samples), np.nan, dtype=np.float32)
        if keep:
            # Refill oldest-first so the cursor lands past the newest.
            self.traces[:keep] = newest[::-1]
        self._cursor = keep % history
        self._n_retained = keep
        self.version += 1

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    @property
    def history(self) -> int:
        return self._history

    @property
    def n_retained(self) -> int:
        """How many slots hold a waveform. Below ``history`` until it fills."""
        return self._n_retained

    @property
    def ages(self) -> np.ndarray:
        """Age of each slot, ``0`` for the newest, one entry per slot.

        Nothing is reordered on write, so this is how a renderer knows which
        slot is which -- to fade older waveforms, or to skip empty ones. An age
        below :attr:`n_retained` means the slot holds a waveform; at or above it
        the slot has never been written.
        """
        return (self._cursor - 1 - np.arange(self._history)) % self._history

    def recent(self, n: int | None = None) -> np.ndarray:
        """The ``n`` newest waveforms, newest first, as a copy.

        Convenience for callers that want them in order and can afford the
        copy. A renderer should prefer :attr:`traces` with :attr:`ages`, which
        costs nothing.
        """
        n = self._n_retained if n is None else min(int(n), self._n_retained)
        if n <= 0:
            return np.empty((0, self.n_channels, self.n_samples), dtype=np.float32)
        idx = (self._cursor - 1 - np.arange(n)) % self._history
        return self.traces[idx]

    def statistics(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Running ``(mean, std)`` over every waveform pushed, or ``None``.

        ``None`` when statistics are switched off or nothing has been pushed.
        Elements no waveform contributed to are NaN rather than zero: no data is
        not the same as a flat line, and drawing it as one invents a signal.
        """
        if not self.track_statistics or self.n_seen == 0:
            return None
        with np.errstate(invalid="ignore", divide="ignore"):
            n = self._count
            mean = np.where(n > 0, self._sum / np.maximum(n, 1), np.nan)
            # Var = E[x^2] - E[x]^2, clamped: with every sample equal the two
            # terms cancel to a small negative, and a negative variance would
            # make the square root NaN and take the band off screen.
            var = np.maximum(self._sumsq / np.maximum(n, 1) - mean * mean, 0.0)
            std = np.where(n > 1, np.sqrt(var), np.nan)
        return mean.astype(np.float32), std.astype(np.float32)
