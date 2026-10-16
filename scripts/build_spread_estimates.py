#!/usr/bin/env python
"""Regenerate the per-symbol spread estimate table from the Dukascopy corpus.

Reads every ``<DUKASCOPY_ROOT>/<SYMBOL>_tick/<SYMBOL>_GMT+0_US-DST_spread_m5.npz``
(the per-bar M5 spread series used by the AA variable-spread model,
``INFRA-039``/``INFRA-041``) and prints a Python dict block of the **median
``open_spread`` per symbol, in pips** for pasting into
``seasonal_analysis/spread_estimates.py``.

The corpus is large and only available on the authoring host, so the resulting
small table is committed and the runtime never touches the corpus.

Usage::

    python scripts/build_spread_estimates.py
    python scripts/build_spread_estimates.py --root /path/to/dukascopy
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Dict

import numpy as np


def _pip(symbol: str) -> float:
    return 0.01 if symbol.upper().endswith("JPY") else 0.0001


def build_table(root: str) -> Dict[str, float]:
    table: Dict[str, float] = {}
    pattern = os.path.join(root, "*_tick", "*_spread_m5.npz")
    for path in sorted(glob.glob(pattern)):
        symbol = os.path.basename(path).split("_", 1)[0].upper()
        data = np.load(path)
        spread = np.asarray(data["open_spread"], dtype="float64")
        spread = spread[np.isfinite(spread)]
        if spread.size == 0:
            continue
        table[symbol] = round(float(np.median(spread)) / _pip(symbol), 4)
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=os.getenv("DUKASCOPY_ROOT"),
        help="Dukascopy corpus root (defaults to $DUKASCOPY_ROOT).",
    )
    args = parser.parse_args()
    if not args.root or not os.path.isdir(args.root):
        raise SystemExit(
            "Dukascopy root not found; pass --root or set DUKASCOPY_ROOT."
        )

    table = build_table(args.root)
    if not table:
        raise SystemExit(f"No *_spread_m5.npz artifacts under {args.root!r}.")

    print("SPREAD_ESTIMATES_PIPS: Dict[str, float] = {")
    for symbol in sorted(table):
        print(f'    "{symbol}": {table[symbol]},')
    print("}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
