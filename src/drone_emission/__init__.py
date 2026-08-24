"""Drone-emission reproducibility package."""

from __future__ import annotations

import sys
from pathlib import Path

# The research modules predate packaging and still use sibling imports such as
# ``from utils import ...``. Keep that internal compatibility private while the
# public interface is the package CLI.
_PACKAGE_DIR = Path(__file__).resolve().parent
if str(_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_DIR))

__all__ = ["__version__"]
__version__ = "0.1.0"
