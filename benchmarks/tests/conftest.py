import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for sub in ("benchmarks/tools", "scripts"):
    sys.path.insert(0, str(ROOT / sub))
