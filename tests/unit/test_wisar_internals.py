import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import pytest
import xarray as xr
from scipy import stats

from idmodels.config import (
    FLUSIGHT_Q_LEVELS,
    InnovationDist,
    PoolingStrategy,
    PowerTransform,
    SourceType,
    WISARModelConfig,
)
from idmodels.transforms import WindowCenterScaleTransform
from idmodels.wisar import (
    WISARModel,
    _gaussian_wis_loss,
    _simulate_forecast_draws,
    _studentt_wis_loss,
    _wis_alphas,
)


def _wis_numpy(y, quantiles: dict) -> float:
    """Reference WIS (Bracher et al. 2021) from a {level: value} dict."""
    total, denom = 0.5 * abs(y - quantiles[0.5]), 0.5
    for alpha in _wis_alphas(list(quantiles)):
        lo, up = quantiles[round(alpha / 2, 6)], quantiles[round(1 - alpha / 2, 6)]
        total += alpha / 2 * ((up - lo) + 2 / alpha * max(lo - y, 0) + 2 / alpha * max(y - up, 0))
        denom += alpha / 2
    return total / denom


def test_wis_alphas_flusight_grid():
    expected = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    np.testing.assert_allclose(_wis_alphas(FLUSIGHT_Q_LEVELS), expected)


@pytest.mark.parametrize("dist", ["gaussian", "studentt"])
def test_wis_loss_matches_numpy_reference(dist):
    mu, sigma, y, nu = np.array([0.1, -0.4, 0.0]), np.array([0.5, 1.2, 0.3]), np.array([0.6, -0.5, 2.0]), 5.0
    alpha = _wis_alphas(FLUSIGHT_Q_LEVELS)
    with pm.Model():
        args = (pt.constant(mu), pt.constant(sigma))
        if dist == "gaussian":
            actual = _gaussian_wis_loss(*args, pt.constant(y), pt.constant(alpha), len(alpha)).eval()
            ppf = stats.norm.ppf
        else:
            actual = _studentt_wis_loss(*args, nu, pt.constant(y), pt.constant(alpha), len(alpha)).eval()
            ppf = lambda q: stats.t.ppf(q, nu)  # noqa: E731

    expected = sum(_wis_numpy(y[i], {q: mu[i] + sigma[i] * ppf(q) for q in FLUSIGHT_Q_LEVELS}) for i in range(3))
    assert actual == pytest.approx(expected, rel=1e-6)


def test_window_center_scale_transform_roundtrip():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"source": "nhsn", "location": np.repeat(["01", "02"], 50), "inc_trans": rng.gamma(2, 1, 100)})
    out = WindowCenterScaleTransform().apply(df.copy())

    r = df.loc[df["location"] == "01", "inc_trans"].values
    q95 = np.quantile(r, 0.95)
    expected = r / q95 - np.mean(r / q95)
    np.testing.assert_allclose(out.loc[out["location"] == "01", "inc_trans_cs"].values, expected)
    np.testing.assert_allclose(WindowCenterScaleTransform().invert(out["inc_trans_cs"].values, out), df["inc_trans"])


def _fake_idata(mu0, sigma, phi, fc, n_draws=10):
    """Posterior with every draw equal to the given values (one chain)."""
    def var(x, dims):
        x = np.asarray(x, dtype=float)
        return xr.DataArray(np.broadcast_to(x, (1, n_draws) + x.shape), dims=("chain", "draw") + dims)
    posterior = xr.Dataset({"mu0": var(mu0, ("loc",)), "sigma": var(sigma, ("loc",)),
                            "phi": var(phi, ("lag",)), "fourier_coefs": var(fc, ("fc",))})
    return type("IData", (), {"posterior": posterior})()


@pytest.mark.parametrize("dist", list(InnovationDist))
def test_simulate_forecast_draws_deterministic_cases(dist):
    dates = pd.date_range("2024-01-06", periods=10, freq="W-SAT")
    z_hist = np.linspace(-1, 1, 10)
    kwargs = dict(dates_history=dates.values, h_max=3, p=6, K=2, n_param_draws=4, n_innov_per_param=2, seed=1,
                  innovation_dist=dist, t_dof=5.0)
    shared = PoolingStrategy.SHARED

    # no AR, no seasonality, no noise: every path sits at the location's mean
    idata = _fake_idata(mu0=[0.3, -0.2], sigma=[0.0, 0.0], phi=np.zeros(6), fc=np.zeros(4))
    sims = _simulate_forecast_draws(idata, shared, shared, 1, z_history=z_hist, **kwargs)
    assert sims.shape == (8, 3)
    np.testing.assert_allclose(sims, -0.2)

    # phi_1 = 1, no noise: a random walk without innovations stays at the last observation
    idata = _fake_idata(mu0=[0.0, 0.0], sigma=[0.0, 0.0], phi=[1, 0, 0, 0, 0, 0], fc=np.zeros(4))
    sims = _simulate_forecast_draws(idata, shared, shared, 0, z_history=z_hist, **kwargs)
    np.testing.assert_allclose(sims, z_hist[-1])

    # with noise, the same seed reproduces the same paths
    idata = _fake_idata(mu0=[0.0, 0.0], sigma=[0.5, 0.5], phi=[0.5, 0, 0, 0, 0, 0], fc=[0.1, 0.2, 0, 0])
    np.testing.assert_array_equal(_simulate_forecast_draws(idata, shared, shared, 0, z_history=z_hist, **kwargs),
                                  _simulate_forecast_draws(idata, shared, shared, 0, z_history=z_hist, **kwargs))


def _config(**overrides):
    kwargs = dict(model_name="wisar", main_source=SourceType.NHSN, fit_locations_separately=False,
                  power_transform=PowerTransform.FOURTH_ROOT)
    return WISARModelConfig(**(kwargs | overrides))


def test_config_validation():
    with pytest.raises(ValueError, match="n_param_draws"):
        WISARModel(_config(num_chains=1, num_samples=100, n_param_draws=200))

    with pytest.raises(ValueError, match="only supports NHSN"):
        WISARModel(_config(main_source=SourceType.NSSP))._build_sources(run_config=None)

    with pytest.raises(ValueError, match="jointly"):
        WISARModel(_config(fit_locations_separately=True))._build_sources(run_config=None)
