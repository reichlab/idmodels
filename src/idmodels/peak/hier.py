"""
Bayesian hierarchical model of the timing and size of the seasonal peak, with partial pooling across locations and
seasons.

Each season-replay row i (source g, location l, season s, current week t, standardized features x_i) has outcomes
k_i = peak_week - t and z_i = log(peak + eps) - log(M_t + eps).

Timing. The probability that the peak has already occurred (only possible once the window has opened, t >= w0) is
    pi0_i = logistic(A0(t) + x_i' beta0 + gamma_g0 + r_i0),
and, given that it has not, the peak week is t + m, m >= 1, with discrete hazard
    h_im = logistic(F(t + m) + G(m) + x_i' beta + (log m - c) x_i' beta_m + gamma_g1 + r_i1),
forced to 1 at the last window week. A0, F and G are cubic B-splines of the current week, the calendar week of the
candidate peak and log m.

Size. For rows whose peak is still ahead (k_i >= 1),
    log(max(z_i, z_floor)) ~ Student-t_nu(mu_i, sigma_i),
    mu_i = B(t) + b1 log k_i + b2 (log k_i)^2 + x_i' beta_z + (log k_i - c) x_i' beta_zm + gamma_g2 + r_i2,
    log sigma_i = s0 + s1 log k_i + s2 t_std + s_g.
Size is modeled conditionally on the timing, so the model gives joint (k, z) draws. If the peak has occurred, z = 0.

Random effects. r_i = u_l + v_s, with location effects u_l (shared by all sources for the same location code, so a
state's ILINet history informs its NHSN forecasts) and season effects v_s (shared by all locations and sources in a
season), each 3-dimensional (already-peaked, hazard, size) with an LKJ correlation and half-normal scales.

Tempering. The ~39 replay rows of one (source, location, season) series all describe the same peak, so treating them
as independent would overstate the evidence by more than an order of magnitude. Every row's log likelihood is
multiplied by `likelihood_weight` (a power, or "tempered", likelihood).

Current season. The current season's effect v_0 is learned at forecast time from what every location has shown so
far. For location l with running maximum M attained at week A, and each earlier origin t' < A, we know that
k_{t'} >= A - t' and z_{t'} >= log(M + eps) - log(M_{t'} + eps), a right-censored observation whose likelihood
    P(k >= K, z >= delta) = (1 - pi0) * sum_{m >= K} P(k = m | k >= 1) P(z >= delta | k = m)
involves v_0. For each posterior draw of the other parameters, v_0 is drawn from a Laplace approximation to its
conditional posterior given these censored observations (tempered like the training rows).

Fitting uses NUTS (numpyro); prediction uses `num_posterior_draws` posterior draws.
"""

import time
import zlib
from typing import Any

import numpy as np
import pandas as pd
from scipy import special, stats
from scipy.interpolate import BSpline

from idmodels.config import PeakHierModelConfig
from idmodels.peak.base import CurrentSeason, PeakModel
from idmodels.peak.series import SOURCE_CODES, SYNC_BURDEN_FEATURES, state_features

HIER_FEATURES = [
    "rel_max",
    "wks_since_max",
    "g1",
    "g3",
    "rm3",
    "cum_rel",
    "hist_rel",
    "hist_missing",
    "nat_rel_max",
    "nat_wks_since_max",
    "nat_g3",
    "nat_missing",
]


def bspline_basis(x: np.ndarray, lo: float, hi: float, df: int) -> np.ndarray:
    """Cubic B-spline basis with df functions and equally spaced knots on [lo, hi] (sums to 1 at every x)."""
    inner = np.linspace(lo, hi, df - 2)
    knots = np.concatenate([[lo] * 3, inner, [hi] * 3])
    x = np.clip(np.asarray(x, dtype=float), lo, hi - 1e-9)
    return BSpline.design_matrix(x, knots, 3).toarray()


def design_features(feats: pd.DataFrame, sync_burden: bool = False, wsm_dummies: bool = False) -> pd.DataFrame:
    """The raw (unstandardized) feature matrix, with missing national and historical features filled in."""
    x = pd.DataFrame(index=feats.index)
    x["rel_max"] = feats["rel_max"]
    x["wks_since_max"] = np.log1p(feats["wks_since_max"].clip(0, 15))
    x["g1"] = feats["g1"]
    x["g3"] = feats["g3"]
    x["rm3"] = feats["rm3"]
    x["cum_rel"] = feats["cum_rel"]
    x["hist_missing"] = feats["hist_rel"].isna().astype(float)
    x["hist_rel"] = feats["hist_rel"].fillna(0.0)
    nat_missing = feats["nat_rel_max"].isna()
    x["nat_rel_max"] = feats["nat_rel_max"].fillna(feats["rel_max"])
    x["nat_wks_since_max"] = np.log1p(feats["nat_wks_since_max"].fillna(feats["wks_since_max"]).clip(0, 15))
    x["nat_g3"] = feats["nat_g3"].fillna(feats["g3"])
    x["nat_missing"] = nat_missing.astype(float)
    cols = list(HIER_FEATURES)
    if wsm_dummies:  # indicators for 0, 1, 2 and 3 weeks since the running maximum (4+ is the reference)
        for d in range(4):
            x[f"wsm_{d}"] = (feats["wks_since_max"] == d).astype(float)
        cols += [f"wsm_{d}" for d in range(4)]
    if (
        sync_burden
    ):  # missing when there is no history or no other location; filled with 0 (hist_missing flags the former)
        for col in SYNC_BURDEN_FEATURES:
            x[col] = feats[col]
        cols += SYNC_BURDEN_FEATURES
    return x[cols].astype(float).fillna(0.0)


class _Terms:
    """
    Linear predictors and log class probabilities, written once for numpy (prediction) and jax.numpy (fitting and the
    current-season update). `P` is a dict of parameters for a single posterior draw.
    """

    def __init__(self, xp, w0: int, w1: int, M: int, basis: dict, c_logm: float, c_logk: float):
        self.xp, self.w0, self.w1, self.M, self.B = xp, w0, w1, M, basis
        self.c_m, self.c_k = c_logm, c_logk  # centering constants for log m (hazard) and log k (size)

    def week_grid(self, t):
        """Calendar week of each candidate peak (n, M), validity mask and forced-hazard mask."""
        xp = self.xp
        m = xp.arange(1, self.M + 1)
        week = t[:, None] + m[None, :]
        valid = (week >= self.w0) & (week <= self.w1)
        return week, valid, week == self.w1

    def log_class_probs(self, P, t, bt, x, g, r, off=None):
        """
        log P(c = 0) (n,) and log P(c = m) (n, M), m = 1..M; -inf where impossible. `off`: optional offsets (o0 (n,),
        oh (n, M), oz (n,)) added to the already-peaked logit and the hazard logits with coefficients P["c_off"].
        """
        xp = self.xp
        week, valid, forced = self.week_grid(t)
        f_week = self.B["week"] @ P["a_w"]  # indexed by week - w0
        g_m = self.B["logm"] @ P["a_m"]  # (M,)
        logm_c = xp.log(xp.arange(1, self.M + 1)) - self.c_m
        eta0 = bt @ P["a0"] + x @ P["beta0"] + P["gamma"][g, 0] + r[:, 0]
        lin = x @ P["beta"] + P["gamma"][g, 1] + r[:, 1]
        if "beta0_t" in P:  # time-varying coefficients: feature effects change linearly with the current week
            xt = x * ((t - 24.0) / 10.0)[:, None]
            eta0 = eta0 + xt @ P["beta0_t"]
            lin = lin + xt @ P["beta_t"]
        eta = (
            f_week[xp.clip(week - self.w0, 0, self.w1 - self.w0)]
            + g_m[None, :]
            + lin[:, None]
            + logm_c[None, :] * (x @ P["beta_m"])[:, None]
        )
        if off is not None:
            eta0 = eta0 + P["c_off"][0] * off[0]
            eta = eta + P["c_off"][1] * off[1]
        lh = xp.where(forced, 0.0, _log_sigmoid(xp, eta))
        l1mh = xp.where(valid & ~forced, _log_sigmoid(xp, -eta), 0.0)
        before = xp.cumsum(l1mh, axis=1) - l1mh
        open_ = t >= self.w0
        lp0 = xp.where(open_, _log_sigmoid(xp, eta0), -xp.inf)
        lnot0 = xp.where(open_, _log_sigmoid(xp, -eta0), 0.0)
        lpm = xp.where(valid, lh + before + lnot0[:, None], -xp.inf)
        return lp0, lpm

    def size_params(self, P, t_std, bt, x, g, r, logk, off=None):
        """mu and sigma of the modeled transform of z given log k; logk has shape (n,) or (n, M)."""
        xp = self.xp
        expand = (lambda a: a[:, None]) if logk.ndim == 2 else (lambda a: a)
        base = bt @ P["b_t"] + x @ P["beta_z"] + P["gamma"][g, 2] + r[:, 2]
        if off is not None:
            base = base + P["c_off"][2] * off[2]
        mu = (
            expand(base)
            + P["b_k"][0] * (logk - self.c_k)
            + P["b_k"][1] * (logk - self.c_k) ** 2
            + (logk - self.c_k) * expand(x @ P["beta_zm"])
        )
        log_sigma = P["s"][0] + P["s"][1] * (logk - self.c_k) + expand(P["s"][2] * t_std + P["s_g"][g])
        return mu, xp.exp(log_sigma)


def _log_sigmoid(xp, a):
    if xp is np:
        return -np.logaddexp(0.0, -a)
    import jax

    return jax.nn.log_sigmoid(a)


def _log_sf(std, nu):
    """
    log P(T > std) for a standard Student-t with nu degrees of freedom (standard normal if nu is None), in jax. The
    incomplete-beta argument is kept away from 0 and 1, where its derivatives are infinite and the Hessians used by the
    Laplace approximation would be NaN (this changes the result by at most about 3e-4, for |std| < 1e-3).
    """
    import jax.numpy as jnp
    from jax.scipy.special import betainc, log_ndtr

    if nu is None:
        return log_ndtr(-std)
    x = jnp.clip(nu / (nu + std**2), 1e-7, 1.0 - 1e-7)
    tail = 0.5 * betainc(0.5 * nu, 0.5, x)  # P(T > |std|)
    return jnp.log(jnp.where(std > 0, tail, 1.0 - tail) + 1e-30)


class PeakHierModel(PeakModel):
    def __init__(self, model_config: PeakHierModelConfig):
        super().__init__(model_config)
        self.model_config: PeakHierModelConfig = model_config

    # ---------------------------------------------------------------------------------------------------------------
    # design

    def _setup_design(self, rows: pd.DataFrame) -> None:
        cfg = self.model_config
        w0, w1 = cfg.window_start_week, cfg.window_end_week
        self.M_ = self.kmax
        self.t_lo_ = float(cfg.replay_start_week)
        x_raw = self._design(rows)
        self.x_mu_ = x_raw.mean(axis=0).to_numpy()
        sd = x_raw.std(axis=0).to_numpy()
        self.x_sd_ = np.where(sd > 1e-6, sd, 1.0)
        self.x_white_ = np.eye(x_raw.shape[1])
        self.n_features_ = x_raw.shape[1]
        if cfg.whiten_features:
            # decorrelate the standardized features, which are strongly collinear (this made the posterior very badly
            # conditioned): x_w = x_s V diag(lambda^{-1/2}) from the eigendecomposition of cov(x_s); directions with
            # (near) zero variance in training are dropped (their columns are zero)
            xs = self._x(rows)
            lam, vec = np.linalg.eigh(np.atleast_2d(np.cov(xs, rowvar=False)))
            keep = lam > 1e-6 * lam.max()
            self.x_white_ = vec * np.where(keep, 1.0 / np.sqrt(np.where(keep, lam, 1.0)), 0.0)[None, :]
        self.basis_ = {
            "week": bspline_basis(np.arange(w0, w1 + 1), w0, w1, cfg.df_week),
            "logm": bspline_basis(np.log(np.arange(1, self.M_ + 1)), 0.0, np.log(self.M_), cfg.df_offset)[:, 1:],
        }
        # centering constants: mean log offset over the hazard's risk-set entries, and mean log k over size rows
        t = rows["season_week"].to_numpy()
        k = rows["k"].to_numpy()
        m = np.arange(1, self.M_ + 1)
        at_risk = (k[:, None] >= m[None, :]) & (t[:, None] + m[None, :] >= w0)
        self.c_logm_ = float((at_risk * np.log(m)[None, :]).sum() / max(at_risk.sum(), 1))
        self.c_logk_ = float(np.log(k[k >= 1]).mean()) if (k >= 1).any() else float(np.log(4.0))
        self.sources_ = sorted(SOURCE_CODES, key=lambda k: SOURCE_CODES[k])
        self.locations_ = sorted(rows["location"].unique())
        self.seasons_ = sorted(rows["season"].unique())

    def _design(self, feats: pd.DataFrame) -> pd.DataFrame:
        cfg = self.model_config
        return design_features(feats, cfg.sync_burden_features, cfg.wsm_dummies)

    def _x(self, feats: pd.DataFrame) -> np.ndarray:
        xs = self._design(feats).to_numpy()
        xs = np.clip((xs - self.x_mu_) / self.x_sd_, -5, 5)
        return xs @ self.x_white_ if hasattr(self, "x_white_") else xs

    def _bt(self, t: np.ndarray) -> np.ndarray:
        return bspline_basis(t, self.t_lo_, self.model_config.window_end_week, self.model_config.df_t)

    def _t_std(self, t: np.ndarray) -> np.ndarray:
        return (np.asarray(t, dtype=float) - 24.0) / 10.0

    def _terms(self, xp):
        cfg = self.model_config
        basis = self.basis_ if xp is np else {k: xp.asarray(v) for k, v in self.basis_.items()}
        return _Terms(xp, cfg.window_start_week, cfg.window_end_week, self.M_, basis, self.c_logm_, self.c_logk_)

    # ---------------------------------------------------------------------------------------------------------------
    # fitting

    def _build(self, rows: pd.DataFrame):
        """The numpyro model, its data, and the reference source code."""
        import jax
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist

        cfg = self.model_config
        rows = rows.loc[rows["season_week"] % cfg.origin_stride == 0].reset_index(drop=True)
        self._setup_design(rows)
        terms = self._terms(jnp)
        n_src, n_loc, n_sea = len(self.sources_), len(self.locations_), len(self.seasons_)
        p = self.n_features_
        ref_src = SOURCE_CODES["ilinet"]

        t_np = rows["season_week"].to_numpy().astype(int)
        k_np = rows["k"].to_numpy().astype(int)
        data = dict(
            t=jnp.asarray(t_np),
            bt=jnp.asarray(self._bt(t_np)),
            t_std=jnp.asarray(self._t_std(t_np)),
            x=jnp.asarray(self._x(rows)),
            g=jnp.asarray(rows["src_code"].to_numpy().astype(int)),
            loc=jnp.asarray(pd.Categorical(rows["location"], categories=self.locations_).codes),
            sea=jnp.asarray(pd.Categorical(rows["season"], categories=self.seasons_).codes),
            peaked=jnp.asarray(k_np <= 0),
            kidx=jnp.asarray(np.clip(k_np, 1, self.M_) - 1),
            logk=jnp.asarray(np.log(np.clip(k_np, 1, None))),
            y=jnp.asarray(self._size_y(rows["z"].to_numpy())),
        )
        off = self._training_offsets(rows)
        if off is not None:
            data.update(o0=jnp.asarray(off[0]), oh=jnp.asarray(off[1]), oz=jnp.asarray(off[2]))
        self.z_max_ = float(rows["z"].max()) + 1.0
        # thinning origins keeps each series' total weight: the rows of a series are near-duplicates
        weight = cfg.likelihood_weight * cfg.origin_stride

        def model(t, bt, t_std, x, g, loc, sea, peaked, kidx, logk, y, o0=None, oh=None, oz=None):
            P = {
                "a0": numpyro.sample("a0", dist.Normal(0, 2).expand([cfg.df_t]).to_event(1)),
                "a_w": numpyro.sample("a_w", dist.Normal(0, 2).expand([cfg.df_week]).to_event(1)),
                "a_m": numpyro.sample("a_m", dist.Normal(0, 2).expand([cfg.df_offset - 1]).to_event(1)),
                "beta0": numpyro.sample("beta0", dist.Normal(0, 1).expand([p]).to_event(1)),
                "beta": numpyro.sample("beta", dist.Normal(0, 1).expand([p]).to_event(1)),
                "beta_m": numpyro.sample("beta_m", dist.Normal(0, 0.5).expand([p]).to_event(1)),
                "b_t": numpyro.sample("b_t", dist.Normal(0, 2).expand([cfg.df_t]).to_event(1)),
                "b_k": numpyro.sample("b_k", dist.Normal(0, 1).expand([2]).to_event(1)),
                "beta_z": numpyro.sample("beta_z", dist.Normal(0, 1).expand([p]).to_event(1)),
                "beta_zm": numpyro.sample("beta_zm", dist.Normal(0, 0.5).expand([p]).to_event(1)),
                "s": numpyro.sample("s", dist.Normal(0, 1).expand([3]).to_event(1)),
            }
            off = None
            if o0 is not None:
                sd = cfg.offset_prior_sd
                P["c_off"] = numpyro.sample("c_off", dist.Normal(1.0, sd).expand([3]).to_event(1))
                off = (o0, oh, oz)
            if cfg.time_varying_coefs:
                P["beta0_t"] = numpyro.sample("beta0_t", dist.Normal(0, 0.5).expand([p]).to_event(1))
                P["beta_t"] = numpyro.sample("beta_t", dist.Normal(0, 0.5).expand([p]).to_event(1))
            s_g_raw = numpyro.sample("s_g_raw", dist.Normal(0, 0.5).expand([n_src]).to_event(1))
            P["s_g"] = s_g_raw.at[ref_src].set(0.0)
            gamma_raw = numpyro.sample("gamma_raw", dist.Normal(0, 1).expand([n_src, 3]).to_event(2))
            P["gamma"] = gamma_raw.at[ref_src].set(0.0)
            nu = numpyro.sample("nu", dist.Gamma(2.0, 0.1)) + 1.0 if self._student else None

            r = jnp.zeros((t.shape[0], 3))
            if cfg.location_effects:
                tau_u = self._effect_scales("tau_u", cfg.location_components)
                L_u = numpyro.sample("L_u", dist.LKJCholesky(3, 2.0)) if cfg.correlated_effects else jnp.eye(3)
                u = self._effects("u", n_loc, tau_u[:, None] * L_u, cfg.location_components, [])
                r = r + u[loc]
            if cfg.season_effects:
                tau_v = self._effect_scales("tau_v", cfg.season_components)
                L_v = numpyro.sample("L_v", dist.LKJCholesky(3, 2.0)) if cfg.correlated_effects else jnp.eye(3)
                v = self._effects(
                    "v", n_sea, tau_v[:, None] * L_v, cfg.season_components, cfg.season_noncentered_components
                )
                r = r + v[sea]

            lp0, lpm = terms.log_class_probs(P, t, bt, x, g, r, off)
            ll_timing = jnp.where(peaked, lp0, jnp.take_along_axis(lpm, kidx[:, None], axis=1)[:, 0])
            numpyro.factor("timing", weight * jnp.sum(ll_timing))
            mu, sigma = terms.size_params(P, t_std, bt, x, g, r, logk, off)
            if self._sqrt:  # normal truncated to (0, inf)
                ll_size = dist.Normal(mu, sigma).log_prob(y) - jax.scipy.special.log_ndtr(mu / sigma)
            else:
                ll_size = (dist.StudentT(nu, mu, sigma) if self._student else dist.Normal(mu, sigma)).log_prob(y)
            ll_size = jnp.where(peaked, 0.0, ll_size)
            numpyro.factor("size", weight * jnp.sum(ll_size))

        return model, data, ref_src

    def _fit(self, rows: pd.DataFrame) -> None:
        import jax
        import numpyro
        from numpyro.infer import MCMC, NUTS

        cfg = self.model_config
        model, data, ref_src = self._build(rows)
        seed = zlib.crc32("|".join(self.seasons_).encode())
        fixed = (
            [
                "a0",
                "a_w",
                "a_m",
                "beta0",
                "beta",
                "beta_m",
                "b_t",
                "b_k",
                "beta_z",
                "beta_zm",
                "s",
                "s_g_raw",
                "gamma_raw",
            ]
            + (["nu"] if self._student else [])
            + self._tv_keys
        )
        t0 = time.time()
        kernel_args: dict[str, Any] = dict(target_accept_prob=cfg.target_accept_prob, max_tree_depth=cfg.max_tree_depth)
        if cfg.laplace_mass:
            # start at the MAP, with the inverse Hessian there as a dense metric: the posterior is badly conditioned
            # (posterior SDs from ~0.004 for well-identified size coefficients to ~2 for prior-dominated spline
            # coefficients, in correlated directions), and NUTS's own mass-matrix adaptation does not recover from that
            # in a practical warm-up
            map_values = self._map_values(model, data, max(cfg.init_map_steps, 1), seed)
            names, inv_mass = self._laplace_metric(model, data, map_values, seed)
            kernel_args.update(
                init_strategy=numpyro.infer.init_to_value(values=map_values),
                dense_mass=[names],
                inverse_mass_matrix={names: inv_mass},
                adapt_mass_matrix=cfg.adapt_mass_matrix,
            )
        else:
            init = numpyro.infer.init_to_median
            if cfg.init_map_steps > 0:
                init = numpyro.infer.init_to_value(values=self._map_values(model, data, cfg.init_map_steps, seed))
            kernel_args.update(init_strategy=init, dense_mass=[tuple(fixed)] if cfg.dense_mass else False)
        t_map = time.time() - t0
        kernel = NUTS(model, **kernel_args)
        mcmc = MCMC(
            kernel,
            num_warmup=cfg.num_warmup,
            num_samples=cfg.num_samples,
            num_chains=cfg.num_chains,
            chain_method="sequential",
            progress_bar=cfg.progress_bar,
        )
        mcmc.run(jax.random.PRNGKey(seed), extra_fields=("num_steps", "diverging"), **data)
        extra = mcmc.get_extra_fields()
        self.mcmc_stats_ = {
            "map_seconds": t_map,
            "mcmc_seconds": time.time() - t0 - t_map,
            "mean_num_steps": float(np.mean(extra["num_steps"])),
            "step_size": float(np.asarray(mcmc.last_state.adapt_state.step_size).mean()),
        }
        samples = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
        self._store_draws(samples, ref_src)

        # convergence diagnostics
        from numpyro.diagnostics import summary

        summ = summary(mcmc.get_samples(group_by_chain=True))
        scalars = {k: v for k, v in summ.items() if k not in ("u", "v")}
        self.max_rhat_ = float(max(np.nanmax(v["r_hat"]) for v in scalars.values()))
        self.min_ess_ = float(min(np.nanmin(v["n_eff"]) for v in scalars.values()))
        self.num_divergences_ = int(np.asarray(extra["diverging"]).sum())
        self.mcmc_stats_.update(max_rhat=self.max_rhat_, min_ess=self.min_ess_, divergences=self.num_divergences_)

    @property
    def _tv_keys(self) -> list[str]:
        """Optional parameter sites: time-varying coefficients and (hybrid model) offset coefficients."""
        keys = ["beta0_t", "beta_t"] if self.model_config.time_varying_coefs else []
        return keys + (["c_off"] if self._uses_offsets else [])

    # offsets hook, used by the hybrid model (idmodels.peak.hybrid): the base model has none
    _uses_offsets = False

    def _training_offsets(self, rows: pd.DataFrame):
        return None

    def _offsets(self, feats: pd.DataFrame):
        """(o0 (n,), oh (n, M), oz (n,)) for rows with state features `feats`, or None."""
        return None

    @property
    def _sqrt(self) -> bool:
        return self.model_config.size_scale == "sqrt"

    @property
    def _student(self) -> bool:
        return self.model_config.student_t and not self._sqrt

    def _size_y(self, z: np.ndarray) -> np.ndarray:
        """The modeled transform of z: sqrt(max(z, 0)), or log(max(z, z_floor))."""
        if self._sqrt:
            return np.sqrt(np.maximum(z, 0.0))
        return np.log(np.maximum(z, self.model_config.z_floor))

    def _effect_scales(self, name: str, comps: list[int]):
        """Half-normal SDs for the active components of an effect (0 for the others), as a 3-vector."""
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist

        active = numpyro.sample(f"{name}_active", dist.HalfNormal(self.model_config.effect_scale).expand([len(comps)]))
        return numpyro.deterministic(name, jnp.zeros(3).at[jnp.asarray(comps)].set(active))

    def _effects(self, name: str, n: int, chol, comps: list[int], noncentered: list[int]):
        """
        n draws of a 3-dimensional effect with covariance chol chol', centered or non-centered. Without correlation,
        only the components in `comps` are sampled and the others are 0; components in `noncentered` are then always
        non-centered (better when their SD may be near zero).
        """
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist

        cfg = self.model_config
        if not cfg.correlated_effects:
            out = jnp.zeros((n, 3))
            cen = [c for c in comps if cfg.centered_effects and c not in noncentered]
            nonc = [c for c in comps if c not in cen]
            sd = jnp.diagonal(chol)
            if cen:
                i = jnp.asarray(cen)
                out = out.at[:, i].set(
                    numpyro.sample(f"{name}_active", dist.Normal(0, sd[i]).expand([n, len(cen)]).to_event(2))
                )
            if nonc:
                i = jnp.asarray(nonc)
                raw = numpyro.sample(f"{name}_raw", dist.Normal(0, 1).expand([n, len(nonc)]).to_event(2))
                out = out.at[:, i].set(sd[i] * raw)
            return numpyro.deterministic(name, out)
        if cfg.centered_effects:
            return numpyro.sample(name, dist.MultivariateNormal(np.zeros(3), scale_tril=chol).expand([n]).to_event(1))
        raw = numpyro.sample(f"{name}_raw", dist.Normal(0, 1).expand([n, 3]).to_event(2))
        return numpyro.deterministic(name, raw @ chol.T)

    @staticmethod
    def _map_values(model, data: dict, steps: int, seed: int) -> dict:
        """A MAP estimate (constrained values of every sample site) found by Adam."""
        import jax
        import numpyro
        from numpyro.infer import SVI, Trace_ELBO
        from numpyro.infer.autoguide import AutoDelta

        guide = AutoDelta(model, init_loc_fn=numpyro.infer.init_to_median)
        svi = SVI(model, guide, numpyro.optim.Adam(0.01), Trace_ELBO())
        res = svi.run(jax.random.PRNGKey(seed + 1), steps, progress_bar=False, **data)
        return guide.median(res.params)

    def _laplace_metric(self, model, data: dict, values: dict, seed: int) -> tuple[tuple, np.ndarray]:
        """
        Inverse Hessian of the potential (negative log posterior, unconstrained space) at `values`, for use as a dense
        NUTS inverse mass matrix over all sites. Eigenvalues are floored so that no direction has SD above 10.
        """
        import jax
        import numpyro
        from jax.flatten_util import ravel_pytree
        from numpyro.infer.util import initialize_model

        info = initialize_model(
            jax.random.PRNGKey(seed + 2),
            model,
            model_kwargs=data,
            init_strategy=numpyro.infer.init_to_value(values=values),
        )
        z = info.param_info.z
        names = tuple(sorted(z))
        flat, unravel = ravel_pytree({k: z[k] for k in names})
        H = np.asarray(jax.hessian(lambda f: info.potential_fn(unravel(f)))(flat), dtype=np.float64)
        lam, vec = np.linalg.eigh(0.5 * (H + H.T))
        lam = np.maximum(lam, 1e-2)
        self.laplace_condition_ = float(lam.max() / lam.min())
        return names, (vec / lam[None, :]) @ vec.T

    def _store_draws(self, samples: dict, ref_src: int) -> None:
        """Keep num_posterior_draws evenly spaced draws, as a dict of arrays with a leading draw axis."""
        cfg = self.model_config
        n = len(samples["a0"])
        idx = np.linspace(0, n - 1, min(cfg.num_posterior_draws, n)).round().astype(int)
        D = len(idx)
        d = {
            k: samples[k][idx]
            for k in ["a0", "a_w", "a_m", "beta0", "beta", "beta_m", "b_t", "b_k", "beta_z", "beta_zm", "s"]
            + self._tv_keys
        }
        d["s_g"] = samples["s_g_raw"][idx].copy()
        d["s_g"][:, ref_src] = 0.0
        d["gamma"] = samples["gamma_raw"][idx].copy()
        d["gamma"][:, ref_src] = 0.0
        d["nu"] = samples["nu"][idx] + 1.0 if self._student else np.full(D, np.inf)
        for name, flag in (("u", cfg.location_effects), ("v", cfg.season_effects)):
            if flag:
                d[name] = samples[name][idx]
                tau = samples[f"tau_{name}"][idx]
                L = samples[f"L_{name}"][idx] if f"L_{name}" in samples else np.broadcast_to(np.eye(3), (D, 3, 3))
                chol = tau[:, :, None] * L
                d[f"chol_{name}"] = chol  # Cholesky factor of the effect covariance
            else:
                d[name] = np.zeros((D, 1, 3))
                d[f"chol_{name}"] = np.zeros((D, 3, 3))
        self.draws_ = d
        self.n_draws_ = D

    def _draw(self, i: int) -> dict:
        return {k: v[i] for k, v in self.draws_.items()}

    # ---------------------------------------------------------------------------------------------------------------
    # current season

    def _observe_current_season(self, cur: CurrentSeason) -> None:
        cfg = self.model_config
        rng = np.random.default_rng(zlib.crc32(f"{cur.source}|{cur.t.tolist()}|{self.seasons_[-1]}".encode()))
        self._cur_locations = list(cur.locations)
        self._cur_loc_u = self._location_effect_draws(cur.locations, rng)  # (D, n_loc, 3)
        prior = np.einsum("dij,dnj->dni", self.draws_["chol_v"], rng.standard_normal((self.n_draws_, 1, 3)))[:, 0]
        self.v_current_ = prior
        self.v_current_info_ = {"n_censored": 0}
        if not (cfg.season_effects and cfg.current_season_update):
            return
        cens = self._censored_rows(cur)
        if cens is None:
            return
        self.v_current_ = self._laplace_v(cens, rng)
        self.v_current_info_ = {"n_censored": len(cens["t"]), "v_mean": self.v_current_.mean(axis=0)}

    def _location_effect_draws(self, locations: list[str], rng: np.random.Generator) -> np.ndarray:
        """Posterior draws of u_l for known locations; new locations get draws from the population distribution."""
        D = self.n_draws_
        out = np.einsum("dij,dnj->dni", self.draws_["chol_u"], rng.standard_normal((D, len(locations), 3)))
        if self.model_config.location_effects:
            for j, loc in enumerate(locations):
                if loc in self.locations_:
                    out[:, j] = self.draws_["u"][:, self.locations_.index(loc)]
        return out

    def _censored_rows(self, cur: CurrentSeason) -> dict | None:
        """Right-censored (k, z) observations at earlier origins of the current season (see module docstring)."""
        from idmodels.peak.series import running_max

        cfg = self.model_config
        w0 = cfg.window_start_week
        y = cur.reported
        n = y.shape[0]
        nat_idx = np.full(n, cur.nat_row) if cur.nat_row >= 0 else None
        eps = np.full(n, cur.eps)
        A = np.full(n, -1)
        logM = np.full(n, np.nan)
        for ti in np.unique(cur.t):
            sel = cur.t == ti
            sel &= ~np.all(np.isnan(y[:, w0 - 1 : ti]), axis=1)  # needs an observed in-window week
            if ti < w0 + 1 or not sel.any():
                continue
            m, mw = running_max(y, int(ti), w0)
            A[sel], logM[sel] = mw[sel].astype(int), np.log(m[sel] + cur.eps)
        frames = []
        for tp in range(cfg.replay_start_week, int(A.max()) if A.max() > 0 else 0):
            use = (A > tp) & ~np.isnan(y[:, tp - 1])
            if not use.any():
                continue
            f = state_features(
                y,
                tp,
                eps,
                w0,
                nat_idx=nat_idx,
                hist_peak=cur.hist_peak,
                group=np.zeros(n, int),
                pool=np.arange(n) != cur.nat_row,
                hist_total=cur.hist_total,
                hist_cum=cur.hist_cum,
            )
            f = f.loc[use].assign(row=np.flatnonzero(use), K=A[use] - tp, delta=logM[use] - f.loc[use, "lm"])
            frames.append(f)
        if not frames:
            return None
        f = pd.concat(frames, ignore_index=True)
        t = f["season_week"].to_numpy().astype(int)
        delta = f["delta"].to_numpy()
        return dict(
            t=t,
            bt=self._bt(t),
            t_std=self._t_std(t),
            x=self._x(f),
            g=np.full(len(f), SOURCE_CODES[cur.source]),
            row=f["row"].to_numpy(),
            K=f["K"].to_numpy(),
            # lower bound on the modeled transform of z (-inf when the size bound carries no information)
            y_bound=np.where(delta > (0.0 if self._sqrt else cfg.z_floor), self._size_y(delta), -np.inf),
            off=self._offsets(f.assign(src_code=SOURCE_CODES[cur.source])),
        )

    def _laplace_v(self, cens: dict, rng: np.random.Generator) -> np.ndarray:
        """For each posterior draw, a draw of v_0 from the Laplace approximation of its conditional posterior."""
        import jax
        import jax.numpy as jnp

        cfg = self.model_config
        terms = self._terms(jnp)
        weight = cfg.likelihood_weight if cfg.current_update_weight is None else cfg.current_update_weight
        n = len(cens["t"])
        n_pad = int(np.ceil(n / 256) * 256)  # pad to limit recompilation

        def pad(a, value=0):
            a = np.asarray(a)
            out = np.full((n_pad,) + a.shape[1:], value, dtype=a.dtype)
            out[:n] = a
            return jnp.asarray(out)

        mask = pad(np.ones(n))
        t, bt, t_std, x, g = pad(cens["t"], 30), pad(cens["bt"]), pad(cens["t_std"]), pad(cens["x"]), pad(cens["g"])
        K, y_bound = pad(cens["K"], 1), pad(cens["y_bound"], -np.inf)
        off = None if cens["off"] is None else tuple(pad(o) for o in cens["off"])
        m_idx = jnp.arange(1, self.M_ + 1)
        logm = jnp.log(m_idx.astype(float))
        u_rows = jnp.asarray(self._cur_loc_u[:, cens["row"]])  # (D, n, 3)
        u_rows = jnp.concatenate([u_rows, jnp.zeros((u_rows.shape[0], n_pad - n, 3))], axis=1)
        nu_all = jnp.asarray(self.draws_["nu"])

        # components of v_0 informed by the censored observations (the others keep their prior draws)
        comp_mask = jnp.asarray([float(c in cfg.current_update_components) for c in range(3)])

        def loglik(v, P, u, nu):
            r = u + (v * comp_mask)[None, :]
            _, lpm = terms.log_class_probs(P, t, bt, x, g, r, off)
            mu, sigma = terms.size_params(P, t_std, bt, x, g, r, jnp.broadcast_to(logm, lpm.shape), off)
            yb = jnp.broadcast_to(y_bound[:, None], lpm.shape)
            finite = jnp.isfinite(yb)
            std = (jnp.where(finite, yb, 0.0) - mu) / sigma
            if self._sqrt:
                log_ndtr = jax.scipy.special.log_ndtr
                log_surv = jnp.where(finite, log_ndtr(-std) - log_ndtr(mu / sigma), 0.0)
            else:
                log_surv = jnp.where(finite, _log_sf(std, nu if self._student else None), 0.0)
            keep = m_idx[None, :] >= K[:, None]
            ll = jax.scipy.special.logsumexp(jnp.where(keep, lpm + log_surv, -jnp.inf), axis=1)
            return weight * jnp.sum(jnp.where(mask > 0, ll, 0.0))

        def neg_log_post(v, P, u, nu, prec):
            return -(loglik(v, P, u, nu) - 0.5 * v @ prec @ v)

        grad = jax.grad(neg_log_post)
        hess = jax.hessian(neg_log_post)

        def newton(P, u, nu, prec):
            def step(v, _):
                H = hess(v, P, u, nu, prec) + 1e-6 * jnp.eye(3)
                dv = jnp.linalg.solve(H, grad(v, P, u, nu, prec))
                dv = dv * jnp.minimum(1.0, 1.0 / (jnp.linalg.norm(dv) + 1e-12))  # damped steps of norm <= 1
                return v - dv, None

            v, _ = jax.lax.scan(step, jnp.zeros(3), None, length=cfg.laplace_iters)
            return v, hess(v, P, u, nu, prec)

        keys = ["a0", "a_w", "a_m", "beta0", "beta", "beta_m", "b_t", "b_k", "beta_z", "beta_zm", "s", "s_g", "gamma"]
        keys += self._tv_keys
        Ps = {k: jnp.asarray(self.draws_[k]) for k in keys}
        chol = self.draws_["chol_v"]
        cov = np.einsum("dij,dkj->dik", chol, chol) + 1e-6 * np.eye(3)
        prec = jnp.asarray(np.linalg.inv(cov))
        v_hat, H = jax.jit(jax.vmap(newton))(Ps, u_rows, nu_all, prec)
        v_hat, H = np.asarray(v_hat), np.asarray(H)
        out = np.empty_like(v_hat)
        for d in range(len(v_hat)):
            try:
                L = np.linalg.cholesky(np.linalg.inv(H[d] + 1e-6 * np.eye(3)))
            except np.linalg.LinAlgError:
                L = np.linalg.cholesky(cov[d])
            out[d] = v_hat[d] + L @ rng.standard_normal(3)
        free = [c for c in range(3) if c not in cfg.current_update_components]
        if free:
            prior = np.einsum("dij,dj->di", self.draws_["chol_v"], rng.standard_normal((len(out), 3)))
            out[:, free] = prior[:, free]
        return out

    # ---------------------------------------------------------------------------------------------------------------
    # prediction

    def _predict(self, feats: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.model_config
        terms = self._terms(np)
        n = len(feats)
        t = feats["season_week"].to_numpy().astype(int)
        bt, t_std, x = self._bt(t), self._t_std(t), self._x(feats)
        g = feats["src_code"].to_numpy().astype(int)
        locs = feats["location"].tolist() if "location" in feats else ["?"] * n
        uniq = list(dict.fromkeys(locs))
        loc_idx = np.array([uniq.index(loc) for loc in locs])
        rng = np.random.default_rng(zlib.crc32(f"{t.tolist()[:5]}|{n}|{self.seasons_[-1]}".encode()))
        if getattr(self, "_cur_locations", None) == uniq:
            u_all = self._cur_loc_u
        else:
            u_all = self._location_effect_draws(uniq, rng)
        v_all = getattr(self, "v_current_", None)
        if v_all is None:
            v_all = np.einsum("dij,dj->di", self.draws_["chol_v"], rng.standard_normal((self.n_draws_, 3)))

        off = self._offsets(feats)
        S = cfg.size_samples_per_draw
        timing = np.zeros((n, self.kmax + 1))
        zs = np.zeros((n, self.n_draws_ * S))
        logm = np.log(np.arange(1, self.M_ + 1, dtype=float))
        for d in range(self.n_draws_):
            P = self._draw(d)
            r = u_all[d][loc_idx] + v_all[d][None, :]
            lp0, lpm = terms.log_class_probs(P, t, bt, x, g, r, off)
            probs = np.concatenate([np.exp(lp0)[:, None], np.exp(lpm)], axis=1)
            probs /= probs.sum(axis=1, keepdims=True)
            timing += probs

            # joint draws: class, then log z given the class
            cum = np.cumsum(probs, axis=1)
            cls = np.minimum((cum[:, None, :] < rng.random((n, S, 1))).sum(axis=2), self.M_)  # (n, S)
            mu, sigma = terms.size_params(P, t_std, bt, x, g, r, np.broadcast_to(logm, (n, self.M_)), off)
            m_i = np.maximum(cls - 1, 0)
            mu_s = np.take_along_axis(mu, m_i, axis=1)
            sd_s = np.take_along_axis(sigma, m_i, axis=1)
            if self._sqrt:  # truncated normal by inversion
                lo = special.ndtr(-mu_s / sd_s)
                e = special.ndtri(np.minimum(lo + (1.0 - lo) * rng.random((n, S)), 1 - 1e-12))
                z = np.minimum(np.maximum(mu_s + sd_s * e, 0.0) ** 2, self.z_max_)
            else:
                e = (
                    stats.t.rvs(P["nu"], size=(n, S), random_state=rng)
                    if self._student
                    else rng.standard_normal((n, S))
                )
                z = np.minimum(np.exp(mu_s + sd_s * e), self.z_max_)
            zs[:, d * S : (d + 1) * S] = np.where(cls == 0, 0.0, z)
        timing /= self.n_draws_
        size = np.quantile(zs, self.size_levels, axis=1).T
        return timing, size
