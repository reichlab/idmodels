# Directional Wave Features for GBQR Model

## Overview

Directional wave features capture spatial-temporal patterns in disease spread by computing distance-weighted averages of neighboring locations' incidence in specified directions (N, NE, E, SE, S, SW, W, NW).

These features allow the GBQR model to learn how disease "waves" propagate geographically over time, improving forecast accuracy when spatial spread patterns are important.

## Motivation

Traditional forecasting models treat each location independently or use simple spatial averaging. Directional wave features enable the model to:

1. **Capture directional spread patterns**: Disease may spread preferentially in certain directions (e.g., following travel corridors, climate patterns)
2. **Learn wave propagation speed**: By including temporal lags, the model can learn how long it takes for a wave to travel between locations
3. **Distinguish between spreading and receding waves**: Velocity features capture acceleration/deceleration of spread

## Feature Types

For each location and time point, the following features are generated:

### 1. Base Directional Features
- `inc_trans_cs_wave_N`: Distance-weighted average of northern neighbors' incidence
- `inc_trans_cs_wave_NE`: Distance-weighted average of northeastern neighbors' incidence
- `inc_trans_cs_wave_E`: Distance-weighted average of eastern neighbors' incidence
- ... (one for each specified direction)

### 2. Aggregate Feature
- `inc_trans_cs_wave_avg`: Overall distance-weighted average of all neighbors (regardless of direction)

### 3. Temporal Lag Features
- `inc_trans_cs_wave_N_lag1`: Northern neighbors' incidence from 1 week ago
- `inc_trans_cs_wave_N_lag2`: Northern neighbors' incidence from 2 weeks ago
- ... (for each direction and lag)

**Important**: `lag1` refers to time t-1, `lag2` refers to time t-2, etc.

### 4. Velocity Features (optional)
- `inc_trans_cs_wave_N_velocity`: Rate of change = current - lag1
- ... (one for each direction)

## Configuration

Directional wave features are **disabled by default** for backwards compatibility. To enable them, add the following parameters to your `model_config`:

```python
from types import SimpleNamespace

model_config = SimpleNamespace(
    # ... existing parameters ...

    # Directional wave features (disabled by default)
    use_directional_waves = True,  # Set to True to enable

    # Which directions to compute (subset of: N, NE, E, SE, S, SW, W, NW)
    wave_directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'],  # Default: all 8

    # Temporal lags to include (lag1 = t-1, lag2 = t-2)
    wave_temporal_lags = [1, 2],  # Default: [1, 2]

    # Maximum distance (km) to consider as neighbor
    wave_max_distance_km = 1000,  # Default: 1000

    # Include velocity (rate of change) features
    wave_include_velocity = False,  # Default: False

    # Include aggregate weighted average feature
    wave_include_aggregate = True  # Default: True
)
```

### Configuration Parameters Explained

#### `use_directional_waves` (bool, default: False)
- Master switch to enable/disable directional wave features
- Must be set to `True` to generate wave features

#### `wave_directions` (list of str, default: ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
- Which directions to compute features for
- Valid directions: N, NE, E, SE, S, SW, W, NW
- Each direction has a 45° cone (±22.5° around center)
- Examples:
  - `['N', 'S', 'E', 'W']` - Just cardinal directions (4 features)
  - `['NE', 'SW']` - Just diagonal directions (2 features)

#### `wave_temporal_lags` (list of int, default: [1, 2])
- Which temporal lags to include
- `lag1` means t-1 (last week), `lag2` means t-2 (two weeks ago)
- Example: `[1, 2, 3]` includes 1, 2, and 3 week lags

#### `wave_max_distance_km` (float, default: 1000)
- Maximum distance (kilometers) to consider a location as a neighbor
- Only locations within this distance are included in directional averages
- Larger values include more distant neighbors (slower computation)
- Typical values:
  - 500-1000 km for state-level analysis (immediate neighbors)
  - 2000-3000 km for regional patterns
  - 5000+ km for continent-wide patterns

#### `wave_include_velocity` (bool, default: False)
- Whether to include velocity features (rate of change)
- Velocity = current - lag1
- Captures acceleration/deceleration of wave spread
- Increases feature count by ~50% (one velocity per direction)

#### `wave_include_aggregate` (bool, default: True)
- Whether to include overall weighted average (all neighbors, any direction)
- Provides general spatial context independent of direction
- Recommended to keep enabled

## Example Configurations

### Minimal Configuration (4 cardinal directions)
```python
model_config = SimpleNamespace(
    # ... other params ...
    use_directional_waves = True,
    wave_directions = ['N', 'S', 'E', 'W']
)
```
Generates: 4 base + 4 aggregate + (4+1)×2 lags = **14 features**

### Standard Configuration (8 directions)
```python
model_config = SimpleNamespace(
    # ... other params ...
    use_directional_waves = True,
    wave_directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'],
    wave_temporal_lags = [1, 2]
)
```
Generates: 8 base + 1 aggregate + (8+1)×2 lags = **27 features**

### Maximum Information (all options)
```python
model_config = SimpleNamespace(
    # ... other params ...
    use_directional_waves = True,
    wave_directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'],
    wave_temporal_lags = [1, 2],
    wave_max_distance_km = 2000,
    wave_include_velocity = True,
    wave_include_aggregate = True
)
```
Generates: 8 base + 1 aggregate + (8+1)×2 lags + (8+1) velocity = **36 features**

### Hypothesis-Driven (specific directions)
```python
# If you suspect disease spreads along NE-SW axis
model_config = SimpleNamespace(
    # ... other params ...
    use_directional_waves = True,
    wave_directions = ['NE', 'SW'],
    wave_temporal_lags = [1, 2, 3],  # Longer lags for slower spread
    wave_max_distance_km = 1500
)
```

## Technical Details

### Directional Cone Definition
Each direction captures neighbors within a 45° cone:
- **N (North)**: 0° ±22.5° (337.5° to 22.5°)
- **NE (Northeast)**: 45° ±22.5° (22.5° to 67.5°)
- **E (East)**: 90° ±22.5° (67.5° to 112.5°)
- ... and so on

### Distance Weighting
Inverse distance weighting is used:
```
weight = 1 / distance
weighted_average = Σ(weight × neighbor_value) / Σ(weight)
```

Closer neighbors have more influence on the feature value.

### Lag Semantics
- **Base feature** (no lag suffix): Uses current time t
- **lag1**: Uses time t-1 (one week ago)
- **lag2**: Uses time t-2 (two weeks ago)

This allows the model to learn patterns like: "If northern neighbors had high incidence last week (lag1), expect it here this week."

### Missing Values
- If a location has no neighbors in a direction (within max_distance_km), the feature value is NaN
- These are handled by LightGBM during training
- Edge locations (e.g., coastal states) may have missing values for certain directions

## Location Support

Currently supported:
- **State-level** (agg_level='state'): US states, DC, PR, and national level

**Data Source:** State centroids are loaded from `src/idmodels/data/state_centroids.csv`, which contains geographic centroids computed from US Census Bureau TIGER/Line shapefiles. See `src/idmodels/data/README.md` for detailed source information.

To add support for other aggregation levels (county, HSA, etc.):
1. Create a CSV file (e.g., `county_centroids.csv`) with columns: `fips`, `name`, `latitude`, `longitude`
2. Place it in `src/idmodels/data/`
3. Update `_load_state_centroids()` function in `src/idmodels/spatial_utils.py` to support the new level
4. Document the data source in `src/idmodels/data/README.md`

## Performance Considerations

### Computational Cost
- Scales O(n²) with number of locations (n)
- For 50 states: ~2,500 distance calculations (precomputed)
- Feature computation is done once per training run

### Feature Count Impact
Feature count depends on configuration:
- Base: n_directions + (1 if aggregate else 0)
- With lags: base × (1 + len(temporal_lags))
- With velocity: total × 1.5 (approximately)

More features → longer training time, but potentially better predictions

### Recommendations
- Start with default configuration (8 directions, 2 lags)
- Experiment with fewer directions if training is slow
- Use `wave_include_velocity=False` unless you have evidence of acceleration patterns

## Interpretation

### Feature Importance
After training, you can examine feature importance to understand:
- Which directions are most predictive (e.g., is NE spread more important than SW?)
- Whether lags matter (are lag1 features more important than current?)
- Whether velocity features add value

### Example Interpretations
- **High importance for `wave_N_lag1`**: Disease tends to arrive from the north with 1-week delay
- **High importance for `wave_avg`**: General spatial clustering matters more than direction
- **High importance for `wave_NE_velocity`**: Acceleration of northeastern spread is predictive

## Warnings and Validation

The implementation includes validation that warns about:
- **Opposite directions included**: If both N and S (or E and W, etc.) are included, they may be correlated in datasets with uniform spatial patterns. However, in typical epidemic scenarios with directional spread, opposite directions provide independent information. Tree-based models like LightGBM are also robust to multicollinearity, so this warning is informational rather than critical.

## Example: Complete GBQR Configuration

```python
from types import SimpleNamespace
from idmodels.gbqr import GBQRModel

# Model configuration with directional wave features
model_config = SimpleNamespace(
    model_class = "gbqr",
    model_name = "gbqr_with_waves",

    # Standard GBQR parameters
    incl_level_feats = True,
    num_bags = 10,
    bag_frac_samples = 0.7,
    reporting_adj = False,
    sources = ["nhsn"],
    fit_locations_separately = False,
    power_transform = "4rt",

    # Directional wave features
    use_directional_waves = True,
    wave_directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'],
    wave_temporal_lags = [1, 2],
    wave_max_distance_km = 1500,
    wave_include_velocity = False,
    wave_include_aggregate = True
)

# Run configuration
run_config = SimpleNamespace(
    disease = "flu",
    ref_date = datetime.date(2024, 1, 6),
    output_root = "output/",
    artifact_store_root = "artifacts/",
    save_feat_importance = True,
    locations = None,  # All locations
    max_horizon = 4,
    q_levels = [0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975],
    q_labels = ["0.025", "0.1", "0.25", "0.5", "0.75", "0.9", "0.975"]
)

# Run model
model = GBQRModel(model_config)
model.run(run_config)
```

## Backwards Compatibility

The implementation is fully backwards compatible:
- Disabled by default (`use_directional_waves = False`)
- Existing configurations without wave parameters work unchanged
- Uses `hasattr()` checks to gracefully handle missing attributes

Old configurations will continue to work without modification.

## Testing

The implementation includes comprehensive tests:

### Unit Tests
- `tests/unit/test_spatial_utils.py`: Tests for spatial calculations (distance, bearing, neighbors)
- `tests/unit/test_directional_wave_features.py`: Tests for feature generation logic

### Integration Tests
- `tests/integration/test_gbqr_wave_features.py`: End-to-end tests with realistic data

Run tests with:
```bash
uv run pytest tests/unit/test_spatial_utils.py -v
uv run pytest tests/unit/test_directional_wave_features.py -v
uv run pytest tests/integration/test_gbqr_wave_features.py -v
```

## References

### Epidemiological Motivation
- Spatial spread of infectious diseases often follows directional patterns
- Travel corridors, population density gradients, and climate patterns create anisotropic spread
- Historical examples: 1918 flu pandemic, COVID-19 spread in US

### Implementation Details
- Haversine distance formula for great circle distance
- Bearing calculation using spherical trigonometry
- Inverse distance weighting for spatial interpolation

## Future Enhancements

Potential extensions:
1. **Additional aggregation levels**: County, HSA, HRR support
2. **Custom distance weighting**: Gaussian kernel, exponential decay
3. **Population-weighted features**: Weight by neighbor population, not just distance
4. **Temporal smoothing**: Moving averages of wave features
5. **Asymmetric cones**: Different cone widths for different directions

## Troubleshooting

### "Missing coordinates for locations" error
- Ensure all locations in your data have entries in `STATE_CENTROIDS` (spatial_utils.py)
- Check that `agg_level` in your data matches supported levels ('state')

### Wave features are all NaN
- Check `wave_max_distance_km` - may be too small
- Verify location codes match those in `STATE_CENTROIDS`
- Some edge locations (islands, coastal states) naturally have fewer neighbors

### Training is slow
- Reduce number of directions (try just ['N', 'S', 'E', 'W'])
- Reduce `wave_max_distance_km`
- Disable velocity features
- Use `fit_locations_separately=True` in model_config

## Contact

For questions or issues, please file an issue on the GitHub repository.
