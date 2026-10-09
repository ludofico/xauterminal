"""TimesFM 3.0 on CPU (Intel mac): XAUUSD M15 close -> median + 10/90% band. Tracks its own hit-rate."""
import os
from functools import lru_cache

import numpy as np

HORIZON = 32  # M15 bars = 8h
hits: list[int] = []
pending: list[dict] = []


@lru_cache(maxsize=1)
def _model():
    from timesfm3 import ModelConfig, TimesFM3Evaluator

    return TimesFM3Evaluator(ModelConfig(checkpoint_path=os.getenv("TIMESFM_CKPT", "google/timesfm-3.0-pytorch"),
                                         per_core_batch_size=1, device="cpu"))


def forecast(close: np.ndarray, last_ts: str) -> dict:
    out = next(iter(_model().predict_batch([close[-512:].astype(np.float32)], horizon=HORIZON,
                                            return_quantiles=True, use_symmetric_averaging=False)))
    q = np.asarray(out.quantiles)  # (h, 9): deciles 0.1..0.9
    med = np.asarray(out.forecast)
    pending.append({"ts": last_ts, "base": float(close[-1]), "dir": int(np.sign(med[-1] - close[-1]))})
    width = float((q[-1, 8] - q[-1, 0]) / close[-1])
    move = float(med[-1] / close[-1] - 1)
    return {"base": float(close[-1]), "median": med.round(2).tolist(), "q10": q[:, 0].round(2).tolist(), "q90": q[:, 8].round(2).tolist(),
            "move_pct": round(100 * move, 3), "band_pct": round(100 * width, 3),
            "bias": float(np.clip(move / max(width, 1e-6) * 2, -1, 1)), "hit_rate": hit_rate()}


def settle(close_by_ts) -> None:
    """Score forecasts whose horizon has passed. close_by_ts(ts, bars_after) -> price or None."""
    for p in list(pending):
        px = close_by_ts(p["ts"], HORIZON)
        if px is not None:
            hits.append(int(np.sign(px - p["base"]) == p["dir"]))
            pending.remove(p)


def hit_rate() -> float | None:
    return round(float(np.mean(hits[-100:])), 3) if hits else None
