# phosphor

GPU-accelerated real-time sweep renderer for multichannel timeseries data. Built on [WebGPU](https://github.com/pygfx/wgpu-py) and Qt, phosphor renders thousands of channels at high sample rates with minimal CPU overhead.

Designed for neuroscience and real-time signal monitoring -- push `(n_samples, n_channels)` numpy arrays and phosphor handles downsampling, autoscaling, and rendering.

## Installation

```bash
pip install phosphor
```

## Quick Start

```python
import numpy as np
from PySide6.QtWidgets import QApplication
from phosphor import SweepConfig, SweepWidget

app = QApplication([])

widget = SweepWidget(SweepConfig(
    n_channels=128,
    srate=30000.0,
    display_dur=2.0,
    n_visible=64,
))
widget.show()

# Push data from any source -- shape: (n_samples, n_channels), float32
widget.push_data(np.random.randn(500, 128).astype(np.float32))

app.exec()
```

### Embedding in an Existing Qt Application

`SweepWidget` is a standard `QWidget` that can be added to any layout:

```python
from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QWidget
from phosphor import SweepConfig, SweepWidget

class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.sweep = SweepWidget(SweepConfig(n_channels=64, srate=1000.0))
        self.setCentralWidget(self.sweep)

    def on_new_data(self, data):
        self.sweep.push_data(data)
```

### Runtime Configuration

Update parameters without recreating the widget:

```python
from phosphor import SweepConfig

widget.update_config(SweepConfig(
    n_channels=256,
    srate=30000.0,
    display_dur=4.0,
    n_visible=128,
))
```

## Built-in Demo

```bash
python -m phosphor
python -m phosphor --channels 256 --srate 30000 --visible 64 --dur 2.0
```

## Keyboard Controls

| Key                     | Action                                    |
|-------------------------|-------------------------------------------|
| `Up` / `Down`           | Scroll channels by 1                      |
| `Page Up` / `Page Down` | Scroll channels by one page               |
| `[` / `]`               | Halve / double visible channel count      |
| `-` / `=`               | Y-axis zoom out / in (disables autoscale) |
| `A`                     | Toggle autoscale                          |
| `,` / `.`               | Halve / double display duration           |

## Development

We use [`uv`](https://docs.astral.sh/uv/) for development.

1. Fork and clone the repository
2. `uv sync` to create a virtual environment and install dependencies
3. `uv run pre-commit install` to set up linting and formatting hooks
4. `uv run pytest tests` to run the test suite
5. Submit a PR against the `dev` branch
