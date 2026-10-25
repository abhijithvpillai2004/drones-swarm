"""Phase 6 Robustness Study CLI Entrypoint.

Supports running the complete robustness and model mismatch study across all
controllers: naive, soft_avoid, feedforward, ua_adc.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.robustness_sweep import execute_robustness_study, main

if __name__ == "__main__":
    main()
