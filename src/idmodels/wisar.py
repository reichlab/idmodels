"""
WISAR: a pooled AR(p) + Fourier seasonality model fit with PyMC, optionally under a Weighted Interval Score (WIS)
Gibbs posterior (Bissiri, Holmes & Walker 2016) in place of a likelihood.

Model, per location l, on the transformed scale z (fourth root of rate, then center/scale):

    z_{l,t} = mu0_l + S_l(t) + sum_{j=1}^{p} phi_j * (z_{l,t-j} - mu0_l - S_l(t-j)) + eps_{l,t}
    S_l(t)  = sum_{k=1}^{K} [a_k sin(2 pi k doy(t) / 365.25) + b_k cos(2 pi k doy(t) / 365.25)]

with eps ~ sigma_l * Normal or sigma_l * StudentT(t_dof). phi and the Fourier coefficients are shared across
locations or per location (theta_pooling / fourier_pooling); sigma_l is always per location. Priors mirror SARIX:
sigma ~ HalfCauchy, phi ~ Normal(0, theta_sd) with theta_sd ~ HalfCauchy, and likewise for the Fourier coefficients.

Under objective=WIS, the observation model is replaced by pm.Potential(-w * WIS), where WIS is computed from the
model's implied quantiles at wis_q_levels and w is a loss scale calibrated from an OLS plug-in fit so the loss sits on
a footing comparable to a Gaussian log-likelihood.

Forecasts are Monte Carlo simulations: n_param_draws posterior draws x n_innov_per_param innovation paths each.

Ported from the WISAR6_fourthroot_adaptive_t model in reichlab/forecast-sandbox-2025-2026. Deliberate differences
from that implementation: data come from iddata (not the FluSight hub's target-data archive), so training starts
2022-08-06 where iddata's NHSN series begins rather than 2022-02-05; and quantiles are taken on the transformed scale
before inverting rather than on the natural scale (equivalent up to interpolation between adjacent simulated draws).

Requires the `wisar` extra: pip install "idmodels[wisar]".
"""

import sys

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from iddata.sources.nhsn import NHSNDataSource
from scipy import stats

from idmodels.config import (
    InnovationDist,
    PoolingStrategy,
    PowerTransform,
    RunConfig,
    SourceType,
    WISARModelConfig,
    WISARObjective,
)
from idmodels.features import FeaturePipeline
from idmodels.model import IDModel
from idmodels.transforms import (
    ComposedTransform,
    FourthRootTransform,
    IdentityTransform,
    Transform,
    WindowCenterScaleTransform,
)

if sys.platform == "darwin":
    # PyTensor adds a "-ld64" linker flag on macOS 15+ that current Command Line Tools no longer accept
    # ("ld: library 'd64' not found"); strip it. No-op on Linux.
    from pytensor.link.c.cmodule import GCC_compiler as _GCC_compiler

    _orig_compile_args = _GCC_compiler.compile_args

    def _patched_compile_args(march_flags=True):
        return [f for f in _orig_compile_args(march_flags) if f != "-ld64"]

    _GCC_compiler.compile_args = staticmethod(_patched_compile_args)


class WISARModel(IDModel):
    """Pooled AR(p) + Fourier model with a WIS-loss (or likelihood) posterior, fit with PyMC."""


    def __init__(self, model_config: WISARModelConfig):
        # Narrow self.model_config from ModelConfig to WISARModelConfig so that type checkers resolve
        # WISARModelConfig-specific attributes in this class.
        super().__init__(model_config)
        self.model_config: WISARModelConfig = model_config
        if model_config.n_param_draws > model_config.num_chains * model_config.num_samples:
            raise ValueError(f"n_param_draws ({model_config.n_param_draws}) exceeds the posterior draws available "
                             f"(num_chains * num_samples = {model_config.num_chains * model_config.num_samples})")


    def _build_sources(self, run_config: RunConfig):
        if self.model_config.main_source != SourceType.NHSN:
            raise ValueError("WISARModel only supports NHSN as main source.")
        if self.model_config.fit_locations_separately:
            raise ValueError("WISARModel fits locations jointly; use theta_pooling/fourier_pooling=NONE instead.")

        return [NHSNDataSource(disease=run_config.disease)]


    def _build_transform(self) -> Transform:
        # No NHSN floor (SourceScaleTransform) and center/scale over the whole training window rather than in-season
        # weeks only, matching the transform WISAR was developed and evaluated with.
        if self.model_config.power_transform == PowerTransform.FOURTH_ROOT:
            power_t: Transform = FourthRootTransform()
        else:
            power_t = IdentityTransform()
        return ComposedTransform([power_t, WindowCenterScaleTransform()])


    def _filter_locations(self, df: pd.DataFrame, run_config: RunConfig) -> pd.DataFrame:
        """Filter to requested locations and the training window, then fill interior gaps in each series.

        The window is applied here, before the transform, so the center/scale factors are computed over exactly the
        training data. Interpolation is on the raw rate scale; trailing gaps are left as NaN and caught downstream.
        """
        df = super()._filter_locations(df, run_config)
        train_start = pd.Timestamp(self.model_config.train_start)
        df = df.loc[(df["wk_end_date"] >= train_start) & (df["wk_end_date"] < pd.Timestamp(run_config.ref_date))]
        df = df.sort_values(["agg_level", "location", "wk_end_date"]).reset_index(drop=True)
        df["inc"] = df.groupby(["agg_level", "location"])["inc"].transform(
            lambda s: s.interpolate(limit_area="inside")
        )
        return df


    def _build_feature_pipeline(self, run_config: RunConfig) -> FeaturePipeline:
        return FeaturePipeline(features=[], initial_feat_names=["inc_trans_cs"])


    def _fit_and_predict(self, df: pd.DataFrame, feat_names: list[str], run_config: RunConfig) -> pd.DataFrame:
        """Fit the pooled model and return quantile predictions in long format (inc_trans_cs space)."""
        mc = self.model_config
        p, K = mc.p, mc.fourier_K

        # order locations as requested in run_config (fixes loc_idx and per-location forecast seeds)
        requested = list(run_config.states) + list(run_config.hsas)
        present = set(df["location"])
        locs = [loc for loc in requested if loc in present]

        z_by_loc, dates_by_loc, last_rows = {}, {}, []
        for loc in locs:
            sub = df.loc[df["location"] == loc].sort_values("wk_end_date")
            if sub["inc_trans_cs"].isna().any():
                raise ValueError(f"WISARModel: location {loc} has missing training data that interpolation could not "
                                 f"fill (e.g. a trailing gap) at ref_date={run_config.ref_date}")
            z_by_loc[loc] = sub["inc_trans_cs"].values
            dates_by_loc[loc] = sub["wk_end_date"].values
            last_rows.append(sub.iloc[[-1]])

        date_ranges = {loc: (len(d), d[0], d[-1]) for loc, d in dates_by_loc.items()}
        if len(set(date_ranges.values())) != 1:
            raise ValueError(f"WISARModel requires every location to share the same training weeks; got {date_ranges}")

        y, loc_idx, F_target, F_lags, z_lags = _build_full_design(z_by_loc, dates_by_loc, locs, p, K)
        wis_alpha = _wis_alphas(mc.wis_q_levels)

        loss_scale = 1.0
        if mc.objective == WISARObjective.WIS:
            loss_scale = _calibrate_loss_scale_pooled(y, loc_idx, F_target, F_lags, z_lags, wis_alpha)

        idata = _fit_model(
            y, loc_idx, F_target, F_lags, z_lags, n_loc=len(locs), p=p, K=K, wis_alpha=wis_alpha,
            objective=mc.objective, theta_pooling=mc.theta_pooling, fourier_pooling=mc.fourier_pooling,
            loss_scale=loss_scale, draws=mc.num_samples, tune=mc.num_warmup, chains=mc.num_chains,
            cores=mc.num_cores, seed=mc.seed_fit, target_accept=mc.target_accept,
            sigma_prior_scale=mc.sigma_prior_scale, theta_sd_prior_scale=mc.theta_sd_prior_scale,
            fourier_beta_sd_prior_scale=mc.fourier_beta_sd_prior_scale,
            innovation_dist=mc.innovation_dist, t_dof=mc.t_dof,
        )

        q_levels = np.array(run_config.q_levels)
        pred_frames = []
        for li, loc in enumerate(locs):
            sims = _simulate_forecast_draws(
                idata, mc.theta_pooling, mc.fourier_pooling, li, z_by_loc[loc], dates_by_loc[loc],
                h_max=run_config.max_horizon, p=p, K=K, n_param_draws=mc.n_param_draws,
                n_innov_per_param=mc.n_innov_per_param, seed=mc.seed_forecast + li,
                innovation_dist=mc.innovation_dist, t_dof=mc.t_dof,
            )  # shape (n_sims, max_horizon)
            qs = np.quantile(sims, q_levels, axis=0)  # shape (n_q, max_horizon)
            pred_frames.append(pd.DataFrame({
                "location": loc,
                "horizon": np.tile(np.arange(1, run_config.max_horizon + 1), len(q_levels)),
                "output_type_id": np.repeat(run_config.q_labels, run_config.max_horizon),
                "value": qs.reshape(-1),
            }))

        # carry the last observed row's metadata (wk_end_date, pop, transform factors) for inverting and formatting
        cols_to_keep = ["source", "agg_level", "location", "wk_end_date", "pop", "inc_trans_cs",
                        "inc_trans_center_factor", "inc_trans_scale_factor"]
        last_obs = pd.concat(last_rows)[cols_to_keep]
        return pd.concat(pred_frames).merge(last_obs, on="location", how="left")


# --------------------------------------------------------------------------------------------------------------------
# model internals
# --------------------------------------------------------------------------------------------------------------------

def _wis_alphas(q_levels: list[float]) -> np.ndarray:
    """Central interval levels alpha implied by a symmetric quantile grid, e.g. 0.025/0.975 -> 0.05."""
    return np.array(sorted({round(2 * lv, 6) if lv < 0.5 else round(2 * (1 - lv), 6) for lv in q_levels if lv != 0.5}))


def _fourier_design_from_dates(dates, K: int) -> np.ndarray:
    """Fourier design matrix on calendar day-of-year, period 365.25 days."""
    doy = pd.DatetimeIndex(dates).dayofyear.values.astype(float)
    cols = []
    for k in range(1, K + 1):
        cols.append(np.sin(2 * np.pi * k * doy / 365.25))
        cols.append(np.cos(2 * np.pi * k * doy / 365.25))
    return np.column_stack(cols)


def _wis_from_quantiles(y, median_q, lower, upper, alpha, n_alpha):
    total = 0.5 * pt.abs(y - median_q)
    denom = 0.5
    for k in range(n_alpha):
        w_k = alpha[k] / 2
        pen = upper[:, k] - lower[:, k]
        pen = pen + (2 / alpha[k]) * pt.maximum(lower[:, k] - y, 0)
        pen = pen + (2 / alpha[k]) * pt.maximum(y - upper[:, k], 0)
        total = total + w_k * pen
        denom = denom + w_k
    return total / denom


def _gaussian_wis_loss(mu, sigma_vec, y, alpha, n_alpha):
    """Summed WIS of Normal(mu, sigma_vec) predictive quantiles. sigma_vec is per observation."""
    z_lower = pt.sqrt(2) * pt.erfinv(2 * (alpha / 2) - 1)
    z_upper = pt.sqrt(2) * pt.erfinv(2 * (1 - alpha / 2) - 1)
    lower = mu[:, None] + sigma_vec[:, None] * z_lower[None, :]
    upper = mu[:, None] + sigma_vec[:, None] * z_upper[None, :]
    return pt.sum(_wis_from_quantiles(y, mu, lower, upper, alpha, n_alpha))


def _studentt_wis_loss(mu, sigma_vec, nu, y, alpha, n_alpha):
    """Student-t analogue of _gaussian_wis_loss. nu must be a constant: pm.icdf(StudentT) has no gradient w.r.t. nu."""
    dist = pm.StudentT.dist(nu=nu, mu=0.0, sigma=1.0)
    t_lower = pm.icdf(dist, alpha / 2)
    t_upper = pm.icdf(dist, 1 - alpha / 2)
    lower = mu[:, None] + sigma_vec[:, None] * t_lower[None, :]
    upper = mu[:, None] + sigma_vec[:, None] * t_upper[None, :]
    return pt.sum(_wis_from_quantiles(y, mu, lower, upper, alpha, n_alpha))


def _calibrate_loss_scale_pooled(y, loc_idx, F_target, F_lags, z_lags, wis_alpha,
                                 target_relative_scale: float = 2.0) -> float:
    """OLS plug-in heuristic for the WIS loss scale w, so exp(-w * WIS) is comparable to a Gaussian log-likelihood.

    Always uses the Gaussian WIS: this is a rough order-of-magnitude calibration, not the fitted model.
    """
    n_loc = loc_idx.max() + 1
    loc_dummies = np.eye(n_loc)[loc_idx]
    F_lags_flat = F_lags.reshape(F_lags.shape[0], -1)
    X = np.column_stack([loc_dummies, F_target, F_lags_flat, z_lags])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    mu_hat = X @ beta
    resid = y - mu_hat
    sigma_hat_by_loc = np.clip(np.array([resid[loc_idx == loc].std() for loc in range(n_loc)]), 1e-3, None)
    sigma_hat_obs = sigma_hat_by_loc[loc_idx]
    loglik = stats.norm.logpdf(y, mu_hat, sigma_hat_obs).sum()
    with pm.Model():
        loss_val = _gaussian_wis_loss(pt.constant(mu_hat), pt.constant(sigma_hat_obs), pt.constant(y),
                                      pt.constant(wis_alpha), len(wis_alpha)).eval()
    return float(target_relative_scale * abs(loglik) / loss_val)


def _build_full_design(z_by_loc: dict, dates_by_loc: dict, locs: list, p: int, K: int):
    """Stack all locations' training series into flat arrays. Returns (y, loc_idx, F_target, F_lags, z_lags)."""
    y_all, loc_idx_all, F_target_all, F_lags_all, z_lags_all = [], [], [], [], []
    for li, loc in enumerate(locs):
        z = z_by_loc[loc]
        F = _fourier_design_from_dates(dates_by_loc[loc], K)
        T = len(z)
        y_all.append(z[p:])
        loc_idx_all.append(np.full(T - p, li))
        F_target_all.append(F[p:])
        F_lags_all.append(np.stack([F[p - j: T - j] for j in range(1, p + 1)], axis=1))
        z_lags_all.append(np.stack([z[p - j: T - j] for j in range(1, p + 1)], axis=1))
    return (np.concatenate(y_all), np.concatenate(loc_idx_all), np.concatenate(F_target_all, axis=0),
            np.concatenate(F_lags_all, axis=0), np.concatenate(z_lags_all, axis=0))


def _build_pooled_mean(phi, fourier_coefs, mu0, theta_pooling, fourier_pooling, loc_idx, F_target, F_lags, z_lags):
    mu0_obs = mu0[loc_idx]
    if fourier_pooling == PoolingStrategy.SHARED:
        S_target = pt.dot(F_target, fourier_coefs)
        S_lags = pt.dot(F_lags.reshape((-1, F_lags.shape[-1])), fourier_coefs).reshape(F_lags.shape[:-1])
    else:
        fc_obs = fourier_coefs[loc_idx]
        S_target = pt.sum(F_target * fc_obs, axis=1)
        S_lags = pt.sum(F_lags * fc_obs[:, None, :], axis=2)

    dev_lags = z_lags - mu0_obs[:, None] - S_lags

    if theta_pooling == PoolingStrategy.SHARED:
        ar_term = pt.dot(dev_lags, phi)
    else:
        ar_term = pt.sum(dev_lags * phi[loc_idx], axis=1)

    return mu0_obs + S_target + ar_term


def _fit_model(y, loc_idx, F_target, F_lags, z_lags, n_loc: int, p: int, K: int, wis_alpha: np.ndarray,
               objective: WISARObjective, theta_pooling: PoolingStrategy, fourier_pooling: PoolingStrategy,
               loss_scale: float, draws: int, tune: int, chains: int, cores: int, seed: int, target_accept: float,
               sigma_prior_scale: float, theta_sd_prior_scale: float, fourier_beta_sd_prior_scale: float,
               innovation_dist: InnovationDist, t_dof: float):
    """Fit the pooled AR(p) + Fourier model and return the InferenceData.

    cores=1 runs chains sequentially; PyMC's per-chain multiprocessing has crashed (EOFError) at full scale on macOS.
    """
    loc_idx_pt = pt.constant(loc_idx)
    y_pt = pt.constant(y)

    with pm.Model():
        mu0 = pm.Normal("mu0", 0, 0.1, shape=n_loc)
        sigma = pm.HalfCauchy("sigma", sigma_prior_scale, shape=n_loc)

        theta_sd = pm.HalfCauchy("theta_sd", theta_sd_prior_scale)
        phi_shape = p if theta_pooling == PoolingStrategy.SHARED else (n_loc, p)
        phi = pm.Normal("phi", 0, theta_sd, shape=phi_shape)

        fourier_beta_sd = pm.HalfCauchy("fourier_beta_sd", fourier_beta_sd_prior_scale)
        fc_shape = 2 * K if fourier_pooling == PoolingStrategy.SHARED else (n_loc, 2 * K)
        fourier_coefs = pm.Normal("fourier_coefs", 0, fourier_beta_sd, shape=fc_shape)

        mu = _build_pooled_mean(phi, fourier_coefs, mu0, theta_pooling, fourier_pooling, loc_idx_pt,
                                pt.constant(F_target), pt.constant(F_lags), pt.constant(z_lags))
        sigma_obs = sigma[loc_idx_pt]

        if objective == WISARObjective.LIKELIHOOD:
            if innovation_dist == InnovationDist.GAUSSIAN:
                pm.Normal("y_obs", mu, sigma_obs, observed=y_pt)
            else:
                pm.StudentT("y_obs", nu=t_dof, mu=mu, sigma=sigma_obs, observed=y_pt)
        else:
            alpha_pt = pt.constant(wis_alpha)
            if innovation_dist == InnovationDist.GAUSSIAN:
                loss = _gaussian_wis_loss(mu, sigma_obs, y_pt, alpha_pt, len(wis_alpha))
            else:
                loss = _studentt_wis_loss(mu, sigma_obs, t_dof, y_pt, alpha_pt, len(wis_alpha))
            pm.Potential("wis_potential", -loss_scale * loss)

        return pm.sample(draws, tune=tune, chains=chains, cores=cores, target_accept=target_accept,
                         random_seed=seed, progressbar=False)


def _simulate_forecast_draws(idata, theta_pooling: PoolingStrategy, fourier_pooling: PoolingStrategy, loc_index: int,
                             z_history: np.ndarray, dates_history, h_max: int, p: int, K: int, n_param_draws: int,
                             n_innov_per_param: int, seed: int, innovation_dist: InnovationDist,
                             t_dof: float) -> np.ndarray:
    """Simulate forecast paths for one location. Returns shape (n_param_draws * n_innov_per_param, h_max)."""
    rng = np.random.default_rng(seed)
    post = idata.posterior
    n_chains, n_draws = post.sizes["chain"], post.sizes["draw"]
    flat_idx = rng.choice(n_chains * n_draws, size=n_param_draws, replace=False)
    chain_idx, draw_idx = np.unravel_index(flat_idx, (n_chains, n_draws))

    mu0_all = post["mu0"].values[chain_idx, draw_idx, loc_index]
    sigma_all = post["sigma"].values[chain_idx, draw_idx, loc_index]
    phi_draws = post["phi"].values[chain_idx, draw_idx]
    phi_all = phi_draws if theta_pooling == PoolingStrategy.SHARED else phi_draws[:, loc_index, :]
    fc_draws = post["fourier_coefs"].values[chain_idx, draw_idx]
    fc_all = fc_draws if fourier_pooling == PoolingStrategy.SHARED else fc_draws[:, loc_index, :]

    future_dates = pd.date_range(dates_history[-1], periods=h_max + 1, freq="W-SAT")[1:]
    F_future = _fourier_design_from_dates(future_dates, K)
    F_recent = _fourier_design_from_dates(pd.DatetimeIndex(dates_history[-p:]), K)

    all_sims = np.zeros((n_param_draws, n_innov_per_param, h_max))
    for i in range(n_param_draws):
        mu0, sigma, phi, fc = mu0_all[i], sigma_all[i], phi_all[i], fc_all[i]
        dev_recent = z_history[-p:] - mu0 - F_recent @ fc
        for j in range(n_innov_per_param):
            dev = list(dev_recent)
            for h in range(h_max):
                ar_mean = np.dot(phi, np.array(dev[-p:][::-1]))
                S_h = F_future[h] @ fc
                if innovation_dist == InnovationDist.GAUSSIAN:
                    innovation = rng.normal(0, sigma)
                else:
                    innovation = rng.standard_t(t_dof) * sigma
                z_h = mu0 + S_h + ar_mean + innovation
                all_sims[i, j, h] = z_h
                dev.append(z_h - mu0 - S_h)
    return all_sims.reshape(-1, h_max)
