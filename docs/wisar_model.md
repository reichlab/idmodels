# WISAR Model

## Overview

`WISARModel` (`idmodels.wisar`) is a pooled AR(p) + Fourier seasonality model for NHSN hospital admissions, fit with
PyMC. Its default configuration is the `WISAR6_fourthroot_adaptive_t` model developed in
reichlab/forecast-sandbox-2025-2026, where it is one of the components of the `gbqr_3src_spatial_wisar` ensemble.

Per location `l`, on the transformed scale `z`:

```
z[l,t] = mu0[l] + S[l](t) + sum_{j=1..p} phi[j] * (z[l,t-j] - mu0[l] - S[l](t-j)) + eps[l,t]
S[l](t) = sum_{k=1..K} a[k] sin(2 pi k doy(t) / 365.25) + b[k] cos(2 pi k doy(t) / 365.25)
```

- `phi` and the Fourier coefficients are shared across locations by default (`theta_pooling` / `fourier_pooling`);
  `mu0` and the innovation scale `sigma` are always per location.
- Innovations are Student-t with fixed degrees of freedom (`innovation_dist`, `t_dof`) or Gaussian.
- With `objective=WISARObjective.WIS` (the default), the likelihood is replaced by a Gibbs posterior: the model's
  implied quantiles at `wis_q_levels` are scored by the weighted interval score, scaled by a loss weight calibrated
  from an OLS plug-in fit. `objective=WISARObjective.LIKELIHOOD` fits an ordinary likelihood instead.
- Prior scales (`sigma_prior_scale`, `theta_sd_prior_scale`, `fourier_beta_sd_prior_scale`) set HalfCauchy
  hyperpriors, so the amount of shrinkage is learned from the data.
- Forecasts are simulated: `n_param_draws` posterior draws × `n_innov_per_param` innovation paths per location.
- Runs are reproducible: `seed_fit` seeds the MCMC and `seed_forecast` (plus each location's index) seeds the
  simulations, so the same data, configuration and package versions give bit-identical output.

## Installation

PyMC is not a core dependency of idmodels. Install the `wisar` extra, and import the model from its module:

```bash
pip install "idmodels[wisar] @ git+https://github.com/reichlab/idmodels.git"
```

```python
from idmodels.wisar import WISARModel  # not exported from idmodels/__init__.py
```

## Data and transform

- Data come from iddata's `NHSNDataSource` (versioned by `ref_date`), filtered to `train_start <= wk_end_date <
  ref_date`. Interior gaps in a location's series are linearly interpolated; a trailing gap raises an error.
- All requested locations must share the same training weeks (the model is fit jointly).
- The transform is the fourth root of the rate per 100k (no NHSN floor), then division by the per-location 95th
  percentile and centering on the per-location mean, both computed over the whole training window
  (`WindowCenterScaleTransform`). GBQR and SARIX instead add an NHSN floor and compute these factors over in-season
  weeks only; for WISAR the floor would change the model materially (it shrinks the spread of `z` by about a quarter).
- Horizons are labeled relative to the last observed week, as for the other idmodels models.

## Usage

```python
import datetime
from pathlib import Path

from iddata.enums import Disease

from idmodels.config import FLUSIGHT_Q_LEVELS, PowerTransform, RunConfig, SourceType, WISARModelConfig
from idmodels.wisar import WISARModel

model_config = WISARModelConfig(
    model_name="WISAR6_fourthroot_adaptive_t",
    main_source=SourceType.NHSN,
    fit_locations_separately=False,  # must be False; use theta_pooling/fourier_pooling to un-pool
    power_transform=PowerTransform.FOURTH_ROOT,
    # all other defaults reproduce WISAR6_fourthroot_adaptive_t
)
run_config = RunConfig(
    disease=Disease.FLU,
    ref_date=datetime.date(2026, 10, 24),
    output_root=Path("model-output"),
    artifact_store_root=None,
    max_horizon=4,
    states=["US", "01", "02"],  # ...
    hsas=[],
    q_levels=FLUSIGHT_Q_LEVELS,
    q_labels=[str(q) for q in FLUSIGHT_Q_LEVELS],
)
WISARModel(model_config).run(run_config)
# -> model-output/UMass-WISAR6_fourthroot_adaptive_t/2026-10-24-UMass-WISAR6_fourthroot_adaptive_t.csv
```

The output is a standard hub submission file, like those of `GBQRModel` and `SARIXModel`, so WISAR can be run on its
own or combined with other models' outputs downstream (e.g. with hubEnsembles).

`num_cores` defaults to 1, which runs the MCMC chains one after another. Setting it higher samples chains in parallel
processes; this has crashed with an `EOFError` at full scale on macOS, but not on Linux (Unity), where the sandbox
used `num_cores=2`.

For a quick test run, reduce `num_warmup`, `num_samples` and `n_param_draws` (`n_param_draws` must not exceed
`num_chains * num_samples`). A full run of 53 locations takes about 50 minutes on an otherwise idle laptop, and
longer when sharing the CPU with other models.

## Differences from the sandbox implementation

- Training starts 2022-08-06, where iddata's NHSN series begins, rather than 2022-02-05 (the FluSight hub's target
  data). On one reference date this raised medians by about 2% on average and widened 95% intervals by about 5%.
- Quantiles are taken on the transformed scale and then inverted, rather than on the natural scale; these agree up to
  interpolation between adjacent simulated draws.

Given identical input arrays, posterior draws and forecast paths match the sandbox implementation exactly.
