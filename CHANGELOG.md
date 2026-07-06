# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.0]

### Added
- `features.py` module with composable feature engineering classes: `Feature` (abstract base), `OneHotEncodingFeature`, `HolidayFeature`, `TaylorFeature`, `RollingMeanFeature`, `LagFeature`, `HorizonTargetFeature`, `LevelFeatureFilter`, `DirectionalWaveFeature`, `FeaturePipeline`
- `transforms.py` module with `Transform` (abstract base), `FourthRootTransform`, `IdentityTransform`, `CenterScaleTransform`, `ComposedTransform`, `SourceScaleTransform`
- `model.py` module with `IDModel` abstract base class shared by `GBQRModel` and `SARIXModel`
- `SourceType` exported from `idmodels` top-level (sourced from `iddata.enums`)
- Unit tests for `HolidayFeature`, `TaylorFeature`, `RollingMeanFeature`, `CenterScaleTransform`, `FeaturePipeline` edge cases, and `SourceScaleTransform`
- `tests/integration/conftest.py` with shared `make_run_config` pytest fixture, replacing duplicate `create_test_gbqr_run_config` and `create_test_sarix_run_config` helpers
- `ILINET_FLOOR`, `ILINET_SCALE`, `FLUSURVNET_FLOOR`, `FLUSURVNET_SCALE` constants in `constants.py`

### Changed
- **Breaking**: `DataSource` enum removed from `idmodels`; replace with `SourceType` from `iddata.enums`
- **Breaking**: `model_config.sources` now expects `list[SourceType]` instead of `list[DataSource]`
- **Breaking**: `preprocess.py` replaced by `features.py`; feature engineering is now class-based via `FeaturePipeline`
- **Breaking**: source-specific numeric pre-transforms moved from `iddata` `DataSource.load()` into `idmodels` `_build_transform()`: NHSN `+0.75**4`, ILINet `(+exp(-7))*4`, FluSurvNet `(+exp(-3))/2.5` are now applied and explicitly inverted in idmodels; predictions for ILINet and FluSurvNet now come out in original measurement units rather than a silently rescaled space
- **Breaking**: `FourthRootTransform` and `IdentityTransform` `additive_shift` parameter removed; source-specific shifts are now handled by `SourceScaleTransform`
- `_build_transform()` pipeline extended to `[SourceScaleTransform, power_transform, CenterScaleTransform]`
- `ADDITIVE_SHIFT` renamed to `NHSN_FLOOR` in `constants.py`
- `GBQRModel` and `SARIXModel` refactored to extend `IDModel` base class and use `FeaturePipeline`
- `Disease` and `SourceType` enums now sourced directly from `iddata.enums`
- Updated `iddata` dependency to latest commit
- Ruff isort config: pinned `idmodels` as `known-first-party` for consistent cross-environment linting

### Removed
- `DataSource` enum (use `SourceType` from `iddata.enums`)
- `preprocess.py` (superseded by `features.py`)

## [1.3.1]

### Changed
- **Breaking**: `SARIXRunConfig` and `GBQRRunConfig` removed; use `RunConfig` directly (was abstract but is now directly instantiable)
- **Breaking**: `num_warmup`, `num_samples`, `num_chains` moved from `SARIXRunConfig` to `SARIXModelConfig`
- **Breaking**: `save_feat_importance` moved from `GBQRRunConfig` to `GBQRModelConfig`

## [1.3.0]

### Added
- Concrete configuration dataclasses: `ModelConfig`, `RunConfig` (abstract bases), `SARIXModelConfig`, `SARIXRunConfig`, `GBQRModelConfig`, `GBQRRunConfig`
- `SARIXFourierModelConfig` dataclass with `fourier_K` and `fourier_pooling` fields
- Wave feature fields on `GBQRModelConfig` (`use_directional_waves`, `wave_directions`, etc.), disabled by default
- Enum types: `DataSource`, `Disease`, `PowerTransform`, `PoolingStrategy`
- Docstrings for `ModelConfig` and `RunConfig` base classes
- All config types exported from `idmodels.__init__`

### Changed
- **Breaking**: `model_config.sources` now expects `list[DataSource]` instead of `list[str]`
- **Breaking**: `model_config.power_transform` now expects `PowerTransform` instead of `str`
- **Breaking**: `model_config.disease` now expects `Disease` instead of `str`
- Source validation in `sarix.py` and `gbqr.py` uses `DataSource` enums and set operations instead of `np.isin` with string arrays
- All tests use concrete config dataclasses instead of `SimpleNamespace`
- Updated `directional_wave_features.md` examples to use config dataclasses

### Removed
- `SimpleNamespace` usage throughout tests and documentation
- `model_class` field from model configurations (implied by the dataclass type)
- `num_bags` from `GBQRRunConfig` test helper (it is a `GBQRModelConfig` field)
- `save_feat_importance` from `SARIXRunConfig` test helper (not a SARIX field)

## [1.1.0] - 2025-12-08

### Added
- Directional wave features for spatial-temporal disease modeling in GBQR
- Spatial utilities module (`spatial_utils.py`) with location centroids for US states
- State centroids data file (`state_centroids.csv`) with geographic coordinates for all 50 US states, DC, and territories
- Data directory README documenting available data files
- Haversine distance and bearing calculations for spatial analysis
- `create_directional_wave_features()` function in preprocessing pipeline
- Configuration options for directional wave features (disabled by default for backwards compatibility)
- Comprehensive test suite: 20 unit tests for spatial utilities, 14 for feature generation, 7 integration tests
- Documentation for directional wave features implementation

### Removed
- Optional "examples" dependencies (jupyter, matplotlib, plotly)

## [1.0.0] - 2025-11-24

### Added
- NSSP data source support for SARIX and GBQR models
- HSA (Hospital Service Area) level forecasting support
- State-level forecasting for NSSP data

### Changed
- **Breaking**: `run_config.locations` superseded by `run_config.states` and `run_config.hsas`
- NSSP predictions restricted to [0, 1] range for proportion data
- Updated test infrastructure with config creation helpers

## [0.1.0] - 2025-11-03

### Added
- Support for Fourier pooling option in SARIX models
- `SARIXFourierModel` as a proper subclass of `SARIXModel`
- `uv.lock` for reproducible builds
- Location filter option for models

### Changed
- Pinned sarix dependency to commit 35eea237 for stability
- Updated dependencies (iddata, sarix)
- Refactored `SARIXFourierModel` implementation as subclass

### Fixed
- SARIX `sigma_pooling='shared'` bug with multiple batches
- Import sorting issues (ruff compliance)
- Covariate ordering issue (#10)
- Test determinism across different operating systems
- Horizon handling for SARIX and GBQR models
- Missing value handling in input NHSN data

## [0.0.1] - 2024

### Added
- Initial package setup and structure
- SARIX model support for infectious disease forecasting
- GBQR (Gradient Boosted Quantile Regression) model
- COVID-19 disease support
- Integration tests for core models
- Basic model implementations sourced from flusion-experiments
- Pre-commit hooks with ruff linting
- Python 3.11+ support

### Changed
- Updated to latest iddata API

[Unreleased]: https://github.com/reichlab/idmodels/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/reichlab/idmodels/compare/v1.3.1...v2.0.0
[1.3.0]: https://github.com/reichlab/idmodels/compare/v1.1.0...v1.3.0
[1.1.0]: https://github.com/reichlab/idmodels/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/reichlab/idmodels/compare/v0.1.0...v1.0.0
[0.1.0]: https://github.com/reichlab/idmodels/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/reichlab/idmodels/releases/tag/v0.0.1
