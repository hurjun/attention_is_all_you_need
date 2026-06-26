"""Pytest configuration: ensure the repository root is importable.

Having a ``conftest.py`` at the repository root causes pytest to prepend this
directory to ``sys.path`` so ``import transformer`` / ``import tasks`` work without
installing the package.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
