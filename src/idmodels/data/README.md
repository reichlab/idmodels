# Geographic Data for Spatial Features

This directory contains geographic reference data used for computing directional wave features.

## Files

### `state_centroids.csv`

Geographic centroids (latitude, longitude) for US states, territories, and national level.

**Columns:**
- `fips`: FIPS code (2-digit for states, 'US' for national)
- `state_name`: Human-readable state/territory name
- `latitude`: Latitude in decimal degrees
- `longitude`: Longitude in decimal degrees

**Source:**
These centroids are computed from US Census Bureau TIGER/Line shapefiles representing state boundaries. The centroids represent the geographic center of each state's land area and are suitable for distance and bearing calculations in epidemiological spatial analysis.

**Reference:**
- US Census Bureau TIGER/Line Shapefiles: https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html
- Computed using geographic (not population-weighted) centroids

**Coverage:**
- 50 US states
- District of Columbia
- Puerto Rico
- National aggregate ('US')

**Coordinate System:**
- WGS84 (EPSG:4326)
- Decimal degrees

**Usage:**
These coordinates are used by `spatial_utils.py` to:
1. Calculate distances between locations (Haversine formula)
2. Determine bearing/direction between locations
3. Find directional neighbors for wave feature computation

**Accuracy:**
Centroids are accurate to ~4 decimal places (~10 meters), which is more than sufficient for state-level spatial analysis where typical distances are hundreds of kilometers.

## Future Additions

Additional geographic reference files can be added here:
- `county_centroids.csv` - County-level centroids (3,000+ locations)
- `hsa_centroids.csv` - Hospital Service Area centroids
- `hrr_centroids.csv` - Hospital Referral Region centroids

## Data Update Policy

These centroids are relatively stable (state boundaries change rarely). If updates are needed:
1. Download current TIGER/Line shapefiles from US Census Bureau
2. Compute centroids using GIS software (e.g., QGIS, GeoPandas)
3. Export to CSV with same format
4. Update this README with new source date
5. Run tests to verify no breaking changes

## License

Geographic boundary data from US Census Bureau is in the public domain (US Government work).
