"""
Kernel conditional density estimation (KCDE) / analog model for the seasonal peak (cf. Ray, Sakrejda, Lauer,
Johansson & Reich 2017, Statistics in Medicine).

Each historical season-replay row i has a state vector x_i (standardized features) and outcomes (c_i, z_i). For a
query state x, row i receives weight
    w_i(x) = lambda^{1[source_i != source of the query]} * prod_j exp(-0.5 * ((x_ij - x_j) / h_j)^2),
and the predictive distribution of (c, z) is the weighted empirical distribution of the analog outcomes, mixed with
the conditional-climatology baseline with weight `baseline_mix`. Bandwidths h_j and the source discount lambda are
chosen to maximize the leave-one-season-out log score of the timing class on a subsample of training rows.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from idmodels.config import PeakBaselineModelConfig, PeakKCDEModelConfig
from idmodels.peak.base import PeakModel
from idmodels.peak.baseline import PeakBaselineModel

KCDE_FEATURES = ["season_week", "rel_max", "wks_since_max", "g1", "g3", "rm3", "nat_rel_max"]
_CHUNK = 256
_FINE_LEVELS = (np.arange(99) + 1) / 100


def _prepare(feats: pd.DataFrame) -> np.ndarray:
    x = feats[KCDE_FEATURES].astype(float).copy()
    x["wks_since_max"] = x["wks_since_max"].clip(upper=10)
    x["nat_rel_max"] = x["nat_rel_max"].fillna(x["rel_max"])
    return x.to_numpy()


class PeakKCDEModel(PeakModel):
    def __init__(self, model_config: PeakKCDEModelConfig):
        super().__init__(model_config)
        self.model_config: PeakKCDEModelConfig = model_config

    # ---------------------------------------------------------------------------------------------------------------
    # kernel weights

    def _log_weights(
        self, xq: np.ndarray, block: np.ndarray, log_h: np.ndarray, log_lambda: float, q_src: np.ndarray
    ) -> np.ndarray:
        """
        (n_query, len(block)) log kernel weights against training rows `block`, max-normalized per query row.
        Analogs from a different source than the query (q_src: source codes) are down-weighted by lambda.
        """
        h = np.exp(log_h)
        xb = self._x[block]
        lw = np.zeros((xq.shape[0], len(block)), dtype=np.float32)
        for j in range(xq.shape[1]):
            lw -= (0.5 * ((xq[:, [j]] - xb[None, :, j]) / h[j]) ** 2).astype(np.float32)
        lw += np.float32(log_lambda) * (q_src[:, None] != self._src[None, block]).astype(np.float32)
        return lw - lw.max(axis=1, keepdims=True)

    def _block(self, t: int) -> np.ndarray:
        """Training rows whose season week is within sw_radius of t (analogs are only drawn from these)."""
        r = self.model_config.sw_radius
        lo = min(max(t, self._t_min), self._t_max)
        return np.flatnonzero(np.abs(self._t - lo) <= r)

    def _allowed_classes(self, t: np.ndarray) -> np.ndarray:
        cfg = self.model_config
        classes = np.arange(self.kmax + 1)
        wk = t[:, None] + classes[None, :]
        allowed = (wk >= cfg.window_start_week) & (wk <= cfg.window_end_week)
        allowed[:, 0] = t >= cfg.window_start_week
        return allowed

    # ---------------------------------------------------------------------------------------------------------------
    # fitting

    def _fit(self, rows: pd.DataFrame) -> None:
        cfg = self.model_config
        rows = rows.dropna(subset=[c for c in KCDE_FEATURES if c != "nat_rel_max"]).reset_index(drop=True)
        x_raw = _prepare(rows)
        self._mu = x_raw.mean(axis=0)
        self._sd = x_raw.std(axis=0) + 1e-8
        self._x = ((x_raw - self._mu) / self._sd).astype(np.float64)
        self._src = rows["src_code"].to_numpy().astype(int)
        self._season = rows["season"].to_numpy()
        self._t = rows["season_week"].to_numpy().astype(int)
        self._t_min, self._t_max = int(self._t.min()), int(self._t.max())
        self._cls = np.clip(rows["timing_class"].to_numpy(), 0, self.kmax).astype(int)
        self._onehot = np.zeros((len(rows), self.kmax + 1))
        self._onehot[np.arange(len(rows)), self._cls] = 1.0
        self._z = rows["z"].to_numpy()

        # baseline used as the mixture component
        self._baseline = PeakBaselineModel(
            PeakBaselineModelConfig(
                model_name="kcde_internal_baseline",
                window_start_week=cfg.window_start_week,
                window_end_week=cfg.window_end_week,
                replay_start_week=cfg.replay_start_week,
                min_window_obs=cfg.min_window_obs,
            )
        )
        self._baseline.hist_, self._baseline.clim_ = self.hist_, self.clim_
        self._baseline.kmax, self._baseline.size_levels = self.kmax, self.size_levels
        self._baseline._fit(rows)

        self._tune(rows)

    def _tune(self, rows: pd.DataFrame) -> None:
        cfg = self.model_config
        rng = np.random.default_rng(20170101)
        nhsn = np.flatnonzero(rows["source"] == "nhsn")
        other = np.flatnonzero(rows["source"] != "nhsn")
        n_half = cfg.num_tuning_rows // 2
        q_idx = np.concatenate(
            [
                rng.choice(nhsn, size=min(n_half, len(nhsn)), replace=False),
                rng.choice(other, size=min(cfg.num_tuning_rows - min(n_half, len(nhsn)), len(other)), replace=False),
            ]
        )
        xq = self._x[q_idx]
        t_q = rows["season_week"].to_numpy().astype(int)[q_idx]
        allowed = self._allowed_classes(t_q)
        base_timing, _ = self._baseline._predict(rows.iloc[q_idx])
        base_p = self._masked_prob(base_timing, allowed)
        true_cls = self._cls[q_idx]
        alpha = cfg.baseline_mix
        groups = []
        for t in np.unique(t_q):
            qi = np.flatnonzero(t_q == t)
            block = self._block(int(t))
            same_season = self._season[q_idx[qi]][:, None] == self._season[None, block]
            groups.append((qi, block, same_season, self._src[q_idx[qi]]))

        def objective(theta):
            log_h, log_lambda = theta[:-1], min(theta[-1], 0.0)
            total = 0.0
            for qi, block, same_season, q_src in groups:
                w = np.exp(self._log_weights(xq[qi], block, log_h, log_lambda, q_src))
                w[same_season] = 0.0  # leave one season out
                p = self._masked_prob(w @ self._onehot[block], allowed[qi])
                p = (1 - alpha) * p + alpha * base_p[qi]
                total += np.sum(np.log(p[np.arange(len(qi)), true_cls[qi]] + 1e-12))
            return -total / len(q_idx)

        theta0 = np.concatenate([np.log(np.full(len(KCDE_FEATURES), 0.5)), [np.log(0.5)]])
        # unit steps on the log scale for the initial simplex; the default (5% of theta0) barely moves on this surface
        simplex = np.vstack([theta0, theta0 + np.eye(len(theta0))])
        res = minimize(
            objective,
            theta0,
            method="Nelder-Mead",
            options={"maxiter": cfg.max_tuning_iter, "xatol": 1e-2, "fatol": 1e-4, "initial_simplex": simplex},
        )
        self.log_h_, self.log_lambda_ = res.x[:-1], min(res.x[-1], 0.0)
        self.tuning_score_ = -res.fun
        self.bandwidths_ = dict(zip(KCDE_FEATURES, np.exp(self.log_h_) * self._sd))

    @staticmethod
    def _masked_prob(p: np.ndarray, allowed: np.ndarray) -> np.ndarray:
        p = np.where(allowed, p, 0.0)
        tot = p.sum(axis=1, keepdims=True)
        n_allowed = np.maximum(allowed.sum(axis=1, keepdims=True), 1)
        return np.where(tot > 0, p / np.where(tot > 0, tot, 1.0), allowed / n_allowed)

    # ---------------------------------------------------------------------------------------------------------------
    # prediction

    def _predict(self, feats: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        alpha = self.model_config.baseline_mix
        xq = (_prepare(feats) - self._mu) / self._sd
        xq = np.nan_to_num(xq, nan=0.0)
        base_timing, base_size = self._baseline._predict(feats)
        q_src = feats["src_code"].to_numpy().astype(int)

        timing = np.zeros((len(xq), self.kmax + 1))
        size = np.zeros((len(xq), len(self.size_levels)))
        t_all = feats["season_week"].to_numpy().astype(int)
        for t in np.unique(t_all):
            block = self._block(int(t))
            z_order = np.argsort(self._z[block])
            z_sorted = self._z[block][z_order]
            onehot = self._onehot[block]
            rows_t = np.flatnonzero(t_all == t)
            for s in range(0, len(rows_t), _CHUNK):
                sl = rows_t[s : s + _CHUNK]
                w = np.exp(self._log_weights(xq[sl], block, self.log_h_, self.log_lambda_, q_src[sl])).astype(float)
                w = w / w.sum(axis=1, keepdims=True)
                timing[sl] = (1 - alpha) * (w @ onehot) + alpha * base_timing[sl]

                # analog quantiles of z on a fine grid, then mix with the baseline quantiles as discrete distributions
                cw = np.cumsum(w[:, z_order], axis=1)
                analog_q = np.stack(
                    [z_sorted[np.minimum(np.searchsorted(row, _FINE_LEVELS), len(row) - 1)] for row in cw]
                )
                pts = np.concatenate([analog_q, base_size[sl]], axis=1)
                wts = np.concatenate(
                    [
                        np.full(analog_q.shape, (1 - alpha) / analog_q.shape[1]),
                        np.full(base_size[sl].shape, alpha / base_size.shape[1]),
                    ],
                    axis=1,
                )
                order = np.argsort(pts, axis=1)
                pts = np.take_along_axis(pts, order, axis=1)
                cum = np.cumsum(np.take_along_axis(wts, order, axis=1), axis=1)
                size[sl] = np.stack(
                    [p[np.minimum(np.searchsorted(c, self.size_levels), len(c) - 1)] for p, c in zip(pts, cum)]
                )
        return timing, size
