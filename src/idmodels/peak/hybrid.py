"""
Hybrid of the gradient-boosted and hierarchical peak models: the hierarchical model (idmodels.peak.hier) with the
gradient-boosted model's predictions (idmodels.peak.gbqr) as offsets.

GBQR maps the state of a season to timing and size forecasts flexibly, but has no notion of a season or location
effect. The hierarchical model has those, including the update of the current season's effect from what all
locations have shown so far, but its linear predictors are less sharp. In the hybrid, for each row,
    o0 = logit(GBQR probability that the peak has already occurred),
    oh_m = logit(GBQR's implied hazard at offset m, i.e. P(peak at t + m) / P(peak at t + m or later | not yet)),
    oz = sqrt(GBQR's median of z given that the peak is still to come),
enter the already-peaked logit, the hazard logits and the mean of sqrt(z) with coefficients c_off ~ N(1, sd) that
are estimated, so the model learns how far to trust GBQR. By default the hierarchical model's own linear feature
terms are dropped; its splines, source effects, location and season effects and the current-season update remain.

Offsets for the training rows are GBQR's out-of-bag predictions (from the bags that did not see the row's season),
so they are as accurate as GBQR's real forecasts, not optimistically in-sample.
"""

import numpy as np
import pandas as pd

from idmodels.config import PeakGBQRModelConfig, PeakHybridModelConfig
from idmodels.peak.gbqr import PeakGBQRModel
from idmodels.peak.hier import PeakHierModel

_P_CLIP = 1e-4


class PeakHybridModel(PeakHierModel):
    _uses_offsets = True

    def __init__(self, model_config: PeakHybridModelConfig):
        super().__init__(model_config)
        self.model_config: PeakHybridModelConfig = model_config

    def _gbqr_config(self) -> PeakGBQRModelConfig:
        cfg = self.model_config
        return PeakGBQRModelConfig(
            model_name=f"{cfg.model_name}_internal_gbqr",
            window_start_week=cfg.window_start_week,
            window_end_week=cfg.window_end_week,
            replay_start_week=cfg.replay_start_week,
            min_window_obs=cfg.min_window_obs,
            timing_smoothing_sd=cfg.timing_smoothing_sd,
            sync_burden_features=cfg.sync_burden_features,
            num_bags=cfg.gbqr_num_bags,
            bag_frac_samples=cfg.gbqr_bag_frac_samples,
            max_k=cfg.gbqr_max_k,
            n_estimators=cfg.gbqr_n_estimators,
            learning_rate=cfg.gbqr_learning_rate,
            min_child_samples=cfg.gbqr_min_child_samples,
            progress_bar=cfg.progress_bar,
        )

    def _fit(self, rows: pd.DataFrame) -> None:
        # GBQR on all training rows (every origin), then out-of-bag offsets for them
        self.gbqr_ = PeakGBQRModel(self._gbqr_config())
        g = self.gbqr_
        g.hist_, g.clim_, g.kmax, g.size_levels = self.hist_, self.clim_, self.kmax, self.size_levels
        g._fit(rows)
        timing, size = g.oob_predict(rows)
        o0, oh, oz = self._offsets_from(timing, size, rows["season_week"].to_numpy().astype(int))
        rows = rows.assign(_o0=o0, _oz=oz)
        rows = pd.concat([rows, pd.DataFrame(oh, columns=[f"_oh{m}" for m in range(oh.shape[1])])], axis=1)
        super()._fit(rows)

    def _training_offsets(self, rows: pd.DataFrame):
        oh = rows[[f"_oh{m}" for m in range(self.M_)]].to_numpy()
        return rows["_o0"].to_numpy(), oh, rows["_oz"].to_numpy()

    def _offsets(self, feats: pd.DataFrame):
        timing, size = self.gbqr_._predict(feats)
        return self._offsets_from(timing, size, feats["season_week"].to_numpy().astype(int))

    def _offsets_from(self, timing: np.ndarray, size: np.ndarray, t: np.ndarray):
        """Offsets from GBQR timing probabilities (n, kmax + 1) and z quantiles (n, levels); see module docstring."""
        cfg = self.model_config
        w0, w1 = cfg.window_start_week, cfg.window_end_week
        m = np.arange(1, self.kmax + 1)
        valid = (t[:, None] + m[None, :] >= w0) & (t[:, None] + m[None, :] <= w1)
        p0 = np.where(t >= w0, timing[:, 0], 0.0)
        o0 = np.log(np.clip(p0, _P_CLIP, 1 - _P_CLIP)) - np.log1p(-np.clip(p0, _P_CLIP, 1 - _P_CLIP))

        # hazards implied by GBQR's class probabilities over the valid future offsets
        pm = np.where(valid, timing[:, 1:], 0.0)
        pm = pm / np.maximum(pm.sum(axis=1, keepdims=True), 1e-12)
        surv = np.cumsum(pm[:, ::-1], axis=1)[:, ::-1]  # P(peak at offset >= m | not yet)
        h = np.clip(pm / np.maximum(surv, 1e-12), _P_CLIP, 1 - _P_CLIP)
        oh = np.where(valid, np.log(h) - np.log1p(-h), 0.0)

        # median of z given z > 0: the quantile of z at level p0 + (1 - p0) / 2
        lev = p0 + 0.5 * (1.0 - p0)
        zmed = np.array([np.interp(a, self.size_levels, np.sort(q)) for a, q in zip(lev, size)])
        oz = np.sqrt(np.maximum(zmed, 0.0)) if self._sqrt else np.log(np.maximum(zmed, cfg.z_floor))
        return o0, oh, oz

    def _design(self, feats: pd.DataFrame) -> pd.DataFrame:
        if self.model_config.hybrid_keep_features:
            return super()._design(feats)
        # a single constant column (dropped by whitening): the hierarchical model's own feature terms are unused
        return pd.DataFrame({"none": np.zeros(len(feats))}, index=feats.index)
