"""Lets the scripts import the `advanced_rag` package from ./src without installing anything."""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Windows terminals sometimes can't print characters like "–" or "≥": don't crash on them.
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(errors="replace")
    except Exception:
        pass
