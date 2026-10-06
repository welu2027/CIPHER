"""One command for the ARR re-analysis:  .venv/bin/python analysis/run_reanalysis.py

Runs every step under scorer v1 (as submitted) and v2 (label-normalised), each in
a fresh process; outputs go to analysis/out/v1/ and analysis/out/v2/. Parses and
simulations are cached in analysis/out/cache/ (delete it to recompute from scratch).

    .venv/bin/python analysis/run_reanalysis.py            # both versions
    .venv/bin/python analysis/run_reanalysis.py v2         # one version
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = [
    "p1_inventory.py",       # Part 1: inventory + Table 5 reproduction (+ v2 label audit)
    "p2_1_executive.py",     # 2.1 executive decomposition, probe-free objective
    "p2_2_contingency.py",   # 2.2 contingency vs a fixed reference
    "p2_3_adversarial.py",   # 2.3 alternative adversarial worlds
    "p2_4_attention.py",     # 2.4 attention baselines
    "p2_8_statistics.py",    # 2.8 statistics
]

if __name__ == "__main__":
    versions = sys.argv[1:] or ["v1", "v2"]
    for v in versions:
        for step in STEPS:
            if not os.path.exists(os.path.join(HERE, step)):
                print(f"(skipping {step}: not present)")
                continue
            print(f"\n######## [{v}] {step} ########", flush=True)
            subprocess.run([sys.executable, os.path.join(HERE, step)], cwd=HERE, check=True,
                           env={**os.environ, "CIPHER_SCORER": v})
