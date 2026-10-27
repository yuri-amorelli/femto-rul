"""Thin wrapper the API uses around the serving bundle (src/serving.py).

The bundle folder comes from FEMTO_BUNDLE_DIR, defaulting to
artifacts/serving/femto. Nothing here needs PyTorch or scikit-learn.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional

from src.serving import ServingBundle, default_bundle_dir, load_bundle

MAX_HISTORY = int(os.environ.get("FEMTO_MAX_HISTORY", "50000"))


class RULService:
    def __init__(self, bundle_dir: Optional[Path] = None) -> None:
        self.bundle_dir = Path(bundle_dir or os.environ.get("FEMTO_BUNDLE_DIR") or default_bundle_dir())
        self.bundle: Optional[ServingBundle] = None

    def load(self) -> None:
        self.bundle = load_bundle(self.bundle_dir)

    def warm_up(self) -> None:
        """Compute the demo replays in the background, so the first visitor does not wait."""
        def run():
            for name in self.bundle.demo_bearings:
                self.bundle.replay(name)
        threading.Thread(target=run, daemon=True, name="replay-warm-up").start()

    @property
    def loaded(self) -> bool:
        return self.bundle is not None

    @property
    def model_version(self) -> str:
        return self.bundle.version

    def step(self, **kwargs) -> dict:
        return self.bundle.step(max_history=MAX_HISTORY, **kwargs)
