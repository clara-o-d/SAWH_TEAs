#!/usr/bin/env python3
"""
Convenience wrapper: generate Wilson Figure 2 data (Table S3 / Note S1), then plot.

For faster iteration, run the steps separately:
  python wilson-et-al._re-creation/scripts/figure2_generate.py
  python wilson-et-al._re-creation/scripts/figure2_plot.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from figure2_generate import main as generate_main
from figure2_plot import main as plot_main


def main():
    generate_main()
    plot_main()


if __name__ == "__main__":
    main()
