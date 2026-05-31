#!/usr/bin/env python3
"""TOFU entrypoint for the shared margin-unlearning trainer.

The historical TOFU launchers call this file.  The implementation now lives in
``scripts/run_margin_unlearning.py`` so TOFU uses the same training, probing,
seeding, checkpointing, and skip-final-eval behavior as the MUSE runs.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_margin_unlearning import main  # noqa: E402


if __name__ == "__main__":
    if "--corpus" not in sys.argv:
        sys.argv.extend(["--corpus", "tofu"])
    main()
