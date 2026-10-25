"""Swarm Flight Demo CLI entrypoint.

Thin wrapper around demo_gui providing standard CLI flight demonstration
for all four controllers: naive, soft_avoid, feedforward, ua_adc.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demo_gui import main

if __name__ == "__main__":
    main()
