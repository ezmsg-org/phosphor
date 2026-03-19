"""Default constants, colors, and configuration values."""

DEFAULT_DISPLAY_DUR = 2.0
DEFAULT_N_COLUMNS = 2000
DEFAULT_N_VISIBLE = 64
DEFAULT_MAX_FPS = 60

CURSOR_GAP_COLUMNS = 5
BG_COLOR = (0.10, 0.10, 0.12, 1.0)
CURSOR_COLOR = (0.25, 0.25, 0.28, 0.85)

DEFAULT_MAX_EVENTS = 500  # max stored events (deque capacity)
EVENT_POOL_SIZE = 64  # max simultaneously rendered event ticks
EVENT_COLOR = (1.0, 1.0, 1.0, 1.0)  # default white
EVENT_THICKNESS = 2.0

# 10-color repeating palette (RGBA, bright on dark background)
CHANNEL_COLORS = [
    (1.0, 0.40, 0.40, 1.0),  # red
    (0.40, 1.0, 0.40, 1.0),  # green
    (0.45, 0.70, 1.0, 1.0),  # blue
    (1.0, 1.0, 0.40, 1.0),  # yellow
    (0.40, 1.0, 1.0, 1.0),  # cyan
    (1.0, 0.40, 1.0, 1.0),  # magenta
    (1.0, 0.70, 0.30, 1.0),  # orange
    (0.90, 0.90, 0.90, 1.0),  # white
    (1.0, 0.60, 0.80, 1.0),  # pink
    (0.60, 1.0, 0.40, 1.0),  # lime
]
