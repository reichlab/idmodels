"""
Conditional-climatology baseline for the seasonal peak.

Timing: the probability that the peak has already occurred is the (smoothed) empirical frequency with which the
running maximum at season week t, attained `wks_since_max` weeks earlier, turned out to be the season peak in
historical training seasons. The remaining probability is spread over later in-window weeks in proportion to the
climatological distribution of the peak week.

Size: empirical quantiles of z among historical rows with a similar season week (within size_week_halfwidth) and the
same weeks-since-maximum bin.
"""

import numpy as np
import pandas as pd

from idmodels.config import PeakBaselineModelConfig
from idmodels.peak.base import PeakModel

WSM_BINS = 5  # weeks since the running max: 0, 1, 2, 3, 4+


def _wsm_bin(wks_since_max) -> np.ndarray:
    return np.clip(np.asarray(wks_since_max, dtype=float), 0, WSM_BINS - 1).astype(int)


class PeakBaselineModel(PeakModel):
    def __init__(self, model_config: PeakBaselineModelConfig):
        super().__init__(model_config)
        self.model_config: PeakBaselineModelConfig = model_config

    def _fit(self, rows: pd.DataFrame) -> None:
        cfg = self.model_config
        rows = rows.assign(wsm_bin=_wsm_bin(rows["wks_since_max"]), peaked=(rows["k"] <= 0).astype(float))
        self._rows = rows

        # P(already peaked | t, wsm bin), with a Jeffreys prior; zero before the window opens
        grp = rows.groupby(["season_week", "wsm_bin"])["peaked"].agg(["sum", "size"])
        self._p_peaked = ((grp["sum"] + 0.5) / (grp["size"] + 1.0)).to_dict()

        # empirical z quantiles by (t, wsm bin), pooling rows within +/- size_week_halfwidth weeks
        self._z_quantiles = {}
        for t in range(int(rows["season_week"].min()), cfg.window_end_week + 1):
            near = rows.loc[(rows["season_week"] - t).abs() <= cfg.size_week_halfwidth]
            for b in range(WSM_BINS):
                zb = near.loc[near["wsm_bin"] == b, "z"].to_numpy()
                if len(zb) < 20:
                    zb = near["z"].to_numpy()
                self._z_quantiles[(t, b)] = np.quantile(zb, self.size_levels)

    def _lookup_t(self, t: int) -> int:
        lo = int(self._rows["season_week"].min())
        return int(np.clip(t, lo, self.model_config.window_end_week))

    def _predict(self, feats: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.model_config
        t = feats["season_week"].to_numpy().astype(int)
        b = _wsm_bin(feats["wks_since_max"].fillna(0))

        p0 = np.array(
            [
                0.0 if ti < cfg.window_start_week else self._p_peaked.get((self._lookup_t(ti), bi), 0.5)
                for ti, bi in zip(t, b)
            ]
        )
        future = self._spread_climatology(t, first_k=1)
        no_future = future.sum(axis=1) == 0
        p0 = np.where(no_future & (t >= cfg.window_start_week), 1.0, p0)
        timing = future * (1.0 - p0)[:, None]
        timing[:, 0] = p0

        size = np.stack([self._z_quantiles[(self._lookup_t(ti), bi)] for ti, bi in zip(t, b)])
        return timing, size
