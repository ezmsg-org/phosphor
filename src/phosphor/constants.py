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

# 10-color repeating palette (RGBA, bright on dark background).
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

# Lower-saturation alternative palette: less eye fatigue when many channels
# are visible at once. Used as the default for new SweepWidget instances.
SOFT_CHANNEL_COLORS = [
    (0.85, 0.55, 0.55, 1.0),  # muted red
    (0.55, 0.80, 0.60, 1.0),  # muted green
    (0.55, 0.70, 0.90, 1.0),  # muted blue
    (0.85, 0.80, 0.50, 1.0),  # muted yellow
    (0.55, 0.85, 0.85, 1.0),  # muted cyan
    (0.80, 0.60, 0.85, 1.0),  # muted magenta
    (0.85, 0.70, 0.50, 1.0),  # muted orange
    (0.78, 0.78, 0.78, 1.0),  # light grey
    (0.85, 0.65, 0.75, 1.0),  # muted pink
    (0.65, 0.80, 0.55, 1.0),  # muted lime
]

DEFAULT_LINE_THICKNESS = 0.8
