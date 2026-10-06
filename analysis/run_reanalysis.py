"""One command for the ARR re-analysis:  .venv/bin/python analysis/run_reanalysis.py

Runs every step in order; outputs go to analysis/out/. Intermediate parses and
simulations are cached in analysis/out/cache/ (delete it to recompute from scratch).
"""

import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = [
    "p1_inventory.py",       # Part 1: inventory + Table 5 reproduction
    "p2_1_executive.py",     # 2.1 executive decomposition, probe-free objective
    "p2_4_attention.py",     # 2.4 attention baselines
    "p2_8_statistics.py",    # 2.8 statistics
]

if __name__ == "__main__":
    sys.path.insert(0, HERE)
    os.chdir(HERE)
    for step in STEPS:
        print(f"\n######## {step} ########", flush=True)
        runpy.run_path(os.path.join(HERE, step), run_name="__main__")
