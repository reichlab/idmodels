# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/reichlab/idmodels/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/reichlab/idmodels/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/reichlab/idmodels/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/reichlab/idmodels/releases/tag/v0.0.1
