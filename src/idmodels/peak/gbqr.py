"""
Gradient-boosted direct models for the seasonal peak.

Size: for each quantile level tau, a LightGBM regressor with the pinball (quantile) loss predicts the tau-quantile of
z = log(peak + eps) - log(M_t + eps) from the state features.

Timing: a LightGBM multiclass classifier predicts the class of k = peak_week - t, with classes
{0: already peaked, 1, ..., max_k, max_k + 1: more than max_k weeks ahead}. The last class is spread over the
corresponding weeks in proportion to the climatological peak-week distribution.

Optionally (`size_offset=True`), each quantile regression is boosted from an offset (LightGBM init_score) equal to the
conditional-climatology baseline's quantile of z for that row, so the trees learn a correction to the baseline.
Without the offset, boosting starts from the unconditional quantile of z, which for upper levels is dominated by very
early-season rows (tiny running maxima, z of 7 or more); with a modest number of trees the predictions then never
come down to z = 0 in post-peak states, giving implausibly wide upper tails late in the season. The offset fixes the
tails but scored slightly worse overall in hindcasts, so it is off by default.

Both parts are bagged over training seasons: each bag is fit to a random subset of seasons, quantile predictions are
combined by the median across bags and class probabilities by the mean.
"""

import zlib

import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm import tqdm

from idmodels.config import PeakBaselineModelConfig, PeakGBQRModelConfig
from idmodels.peak.base import PeakModel
from idmodels.peak.baseline import PeakBaselineModel
from idmodels.peak.extra_features import FEATURE_GROUPS
from idmodels.peak.series import SYNC_BURDEN_FEATURES

GBQR_FEATURES = [
    "season_week",
    "rel_max",
    "wks_since_max",
    "g1",
    "g2",
    "g3",
    "rm3",
    "cum_rel",
    "hist_rel",
    "nat_rel_max",
    "nat_wks_since_max",
    "nat_g3",
    "src_code",
]


class PeakGBQRModel(PeakModel):
    def __init__(self, model_config: PeakGBQRModelConfig):
        super().__init__(model_config)
        self.model_config: PeakGBQRModelConfig = model_config

    @property
    def _features(self) -> list[str]:
        return GBQR_FEATURES + (SYNC_BURDEN_FEATURES if self.model_config.sync_burden_features else [])

    def _group_features(self, groups: list[str] | None) -> list[str]:
        """Feature columns for a list of feature groups (see PeakGBQRModelConfig.size_feature_groups)."""
        if groups is None:
            return self._features
        cols: list[str] = []
        for g in groups:
            cols += GBQR_FEATURES if g == "base" else SYNC_BURDEN_FEATURES if g == "sb" else FEATURE_GROUPS[g]
        return list(dict.fromkeys(cols))

    @property
    def _size_features(self) -> list[str]:
        return self._group_features(self.model_config.size_feature_groups)

    @property
    def _timing_features(self) -> list[str]:
        return self._group_features(self.model_config.timing_feature_groups)

    @staticmethod
    def _design(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        x = frame[cols].astype(float)
        if "src_code" in x:
            x["src_code"] = x["src_code"].astype(int)
        return x

    @staticmethod
    def _categorical(cols: list[str]) -> list[str]:
        return ["src_code"] if "src_code" in cols else []

    def _lgb_params(self, seed: int) -> dict:
        cfg = self.model_config
        return dict(
            verbosity=-1,
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            min_child_samples=cfg.min_child_samples,
            random_state=seed,
        )

    def _fit(self, rows: pd.DataFrame) -> None:
        cfg = self.model_config
        rng = np.random.default_rng(zlib.crc32(self._fitted_key_seed().encode()))
        x_size = self._design(rows, self._size_features)
        x_time = self._design(rows, self._timing_features)
        y_size = rows["z"].to_numpy()
        y_class = np.minimum(rows["timing_class"].to_numpy(), cfg.max_k + 1)
        seasons = rows["season"].unique()

        # conditional-climatology baseline supplies the (optional) offset for each quantile regression
        self._baseline = PeakBaselineModel(
            PeakBaselineModelConfig(
                model_name="gbqr_internal_baseline",
                window_start_week=cfg.window_start_week,
                window_end_week=cfg.window_end_week,
                replay_start_week=cfg.replay_start_week,
                min_window_obs=cfg.min_window_obs,
                timing_smoothing_sd=cfg.timing_smoothing_sd,
            )
        )
        self._baseline.hist_, self._baseline.clim_ = self.hist_, self.clim_
        self._baseline.kmax, self._baseline.size_levels = self.kmax, self.size_levels
        self._baseline._fit(rows)
        _, offset = self._baseline._predict(rows)  # (n, levels)
        if not cfg.size_offset:
            offset = np.zeros_like(offset)

        self._size_models: list[list[lgb.LGBMRegressor]] = []
        self._timing_models: list[lgb.LGBMClassifier] = []
        self.bag_seasons_: list[set] = []
        for _ in tqdm(range(cfg.num_bags), "peak_gbqr bag", disable=not cfg.progress_bar):
            bag_seasons = rng.choice(seasons, size=max(1, int(len(seasons) * cfg.bag_frac_samples)), replace=False)
            self.bag_seasons_.append(set(bag_seasons))
            in_bag = rows["season"].isin(bag_seasons).to_numpy()
            seeds = rng.integers(1e8, size=len(self.size_levels) + 1)
            bag_models = []
            for q_ind, q_level in enumerate(self.size_levels):
                m = lgb.LGBMRegressor(objective="quantile", alpha=q_level, **self._lgb_params(int(seeds[q_ind])))
                # no init_score at all when the offset is off: an explicit init_score (even zeros) disables LightGBM's
                # default of boosting from the average
                init = offset[in_bag, q_ind] if cfg.size_offset else None
                m.fit(
                    x_size.loc[in_bag],
                    y_size[in_bag],
                    init_score=init,
                    categorical_feature=self._categorical(self._size_features),
                )
                bag_models.append(m)
            self._size_models.append(bag_models)
            clf = lgb.LGBMClassifier(objective="multiclass", **self._lgb_params(int(seeds[-1])))
            clf.fit(x_time.loc[in_bag], y_class[in_bag], categorical_feature=self._categorical(self._timing_features))
            self._timing_models.append(clf)

    def _fitted_key_seed(self) -> str:
        # deterministic seed per training season set
        return "|".join(sorted(self.hist_["season"].unique()))

    def oob_predict(self, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Out-of-bag predictions for training rows: each row is predicted only by the bags whose training seasons exclude
        the row's season (all bags if there are none), so the predictions are honest out-of-sample forecasts.
        """
        use = np.array([[s not in bag for bag in self.bag_seasons_] for s in rows["season"]])
        use[~use.any(axis=1)] = True
        return self._predict(rows, bag_mask=use)

    def _predict(self, feats: pd.DataFrame, bag_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """bag_mask (n, num_bags): which bags to combine for each row (default all)."""
        cfg = self.model_config
        x_size = self._design(feats, self._size_features)
        x_time = self._design(feats, self._timing_features)
        x = x_size
        n_bags = len(self._size_models)
        if bag_mask is None:
            bag_mask = np.ones((len(x), n_bags), dtype=bool)

        _, offset = self._baseline._predict(feats)
        if not cfg.size_offset:
            offset = np.zeros_like(offset)
        # with the offset, predictions are corrections to it (predict() does not add init_score back)
        size_by_bag = np.stack(
            [np.column_stack([m.predict(x) for m in bag_models]) + offset for bag_models in self._size_models]
        )  # (bags, n, levels)
        size_by_bag = np.where(bag_mask.T[:, :, None], size_by_bag, np.nan)
        size = np.sort(np.nanmedian(size_by_bag, axis=0), axis=1)

        n_cls = cfg.max_k + 2
        probs = np.zeros((len(x), n_cls))
        for b, clf in enumerate(self._timing_models):
            p = np.zeros((len(x), n_cls))
            p[:, clf.classes_.astype(int)] = clf.predict_proba(x_time)
            probs += p * bag_mask[:, [b]]
        probs /= bag_mask.sum(axis=1, keepdims=True)

        t = feats["season_week"].to_numpy().astype(int)
        timing = np.zeros((len(x), self.kmax + 1))
        timing[:, : cfg.max_k + 1] = probs[:, : cfg.max_k + 1]
        tail = self._spread_climatology(t, first_k=cfg.max_k + 1)
        timing += probs[:, [cfg.max_k + 1]] * tail
        return timing, size
