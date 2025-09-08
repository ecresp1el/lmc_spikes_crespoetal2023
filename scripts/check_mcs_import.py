#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

print("Python:", sys.executable)
print("sys.path[0]:", sys.path[0])
print("McsPyDataTools spec:", importlib.util.find_spec("McsPyDataTools"))
print("mcspydatatools spec:", importlib.util.find_spec("mcspydatatools"))

for name in ("McsPyDataTools", "mcspydatatools"):
    try:
        m = __import__(name)
        print(f"Imported {name} from:", getattr(m, "__file__", None))
        print(f"{name} attrs:", [a for a in dir(m) if a.lower().startswith("mcs") or a.lower().startswith("mcsr")][:10])
    except Exception as e:
        print(f"Import {name} failed:", e)

