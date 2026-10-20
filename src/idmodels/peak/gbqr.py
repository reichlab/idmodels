"""
Gradient-boosted direct models for the seasonal peak.

Size: for each quantile level tau, a LightGBM regressor with the pinball (quantile) loss predicts the tau-quantile of
z = log(peak + eps) - log(M_t + eps) from the state features.

Timing: a LightGBM multiclass classifier predicts the class of k = peak_week - t, with classes
{0: already peaked, 1, ..., max_k, max_k + 1: more than max_k weeks ahead}. The last class is spread over the
corresponding weeks in proportion to the climatological peak-week distribution.

Both parts are bagged over training seasons: each bag is fit to a random subset of seasons, quantile predictions are
combined by the median across bags and class probabilities by the mean.
"""

import zlib

import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm import tqdm

from idmodels.config import PeakGBQRModelConfig
from idmodels.peak.base import PeakModel
from idmodels.peak.extra_features import FEATURE_GROUPS
from idmodels.peak.series import CORE_FEATURES, SYNC_BURDEN_FEATURES


class PeakGBQRModel(PeakModel):
    """Gradient-boosted quantile regression (peak size) and multiclass classification (peak timing) peak model."""

    def __init__(self, model_config: PeakGBQRModelConfig):
        # Narrow self.model_config from PeakModelConfig to PeakGBQRModelConfig so that
        # type checkers resolve PeakGBQRModelConfig-specific attributes in this class.
        super().__init__(model_config)
        self.model_config: PeakGBQRModelConfig = model_config

    @staticmethod
    def _group_features(groups: list[str]) -> list[str]:
        """Feature columns for a list of feature groups (see PeakGBQRModelConfig.size_feature_groups)."""
        cols: list[str] = []
        for g in groups:
            cols += CORE_FEATURES if g == "core" else SYNC_BURDEN_FEATURES if g == "sb" else FEATURE_GROUPS[g]
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
        # deterministic given the set of training seasons
        rng = np.random.default_rng(zlib.crc32("|".join(sorted(self.hist_["season"].unique())).encode()))
        x_size = self._design(rows, self._size_features)
        x_time = self._design(rows, self._timing_features)
        y_size = rows["z"].to_numpy()
        y_class = np.minimum(rows["timing_class"].to_numpy(), cfg.max_k + 1)
        seasons = rows["season"].unique()

        self._size_models: list[list[lgb.LGBMRegressor]] = []
        self._timing_models: list[lgb.LGBMClassifier] = []
        for _ in tqdm(range(cfg.num_bags), "peak_gbqr bag", disable=not cfg.progress_bar):
            bag_seasons = rng.choice(seasons, size=max(1, int(len(seasons) * cfg.bag_frac_samples)), replace=False)
            in_bag = rows["season"].isin(bag_seasons).to_numpy()
            seeds = rng.integers(1e8, size=len(self.size_levels) + 1)
            bag_models = []
            for q_ind, q_level in enumerate(self.size_levels):
                m = lgb.LGBMRegressor(objective="quantile", alpha=q_level, **self._lgb_params(int(seeds[q_ind])))
                m.fit(x_size.loc[in_bag], y_size[in_bag], categorical_feature=self._categorical(self._size_features))
                bag_models.append(m)
            self._size_models.append(bag_models)
            clf = lgb.LGBMClassifier(objective="multiclass", **self._lgb_params(int(seeds[-1])))
            clf.fit(x_time.loc[in_bag], y_class[in_bag], categorical_feature=self._categorical(self._timing_features))
            self._timing_models.append(clf)

    def _predict(self, feats: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.model_config
        x_size = self._design(feats, self._size_features)
        x_time = self._design(feats, self._timing_features)
        n = len(feats)

        size_by_bag = np.stack(
            [np.column_stack([m.predict(x_size) for m in bag_models]) for bag_models in self._size_models]
        )  # (bags, n, levels)
        size = np.sort(np.median(size_by_bag, axis=0), axis=1)

        n_cls = cfg.max_k + 2
        probs = np.zeros((n, n_cls))
        for clf in self._timing_models:
            probs[:, clf.classes_.astype(int)] += clf.predict_proba(x_time)
        probs /= len(self._timing_models)

        t = feats["season_week"].to_numpy().astype(int)
        timing = np.zeros((n, self.kmax + 1))
        timing[:, : cfg.max_k + 1] = probs[:, : cfg.max_k + 1]
        tail = self._spread_climatology(t, first_k=cfg.max_k + 1)
        timing += probs[:, [cfg.max_k + 1]] * tail
        return timing, size
