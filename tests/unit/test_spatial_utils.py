"""Unit tests for spatial_utils module."""

import math
import warnings

import pytest

from idmodels.spatial_utils import (
    DIRECTION_ANGLES,
    compute_bearing,
    get_directional_neighbors,
    get_location_centroids,
    haversine_distance,
    validate_wave_directions,
)


def test_get_location_centroids_state():
    """Test that state centroids are returned correctly."""
    coords = get_location_centroids(agg_level='state')

    # Should have all 50 states + DC + PR + US
    assert len(coords) >= 50

    # Check specific states exist
    assert '06' in coords  # California
    assert '36' in coords  # New York
    assert 'US' in coords  # National

    # Check coordinates are tuples of (lat, lon)
    for loc_code, coord in coords.items():
        assert isinstance(coord, tuple)
        assert len(coord) == 2
        lat, lon = coord
        assert -90 <= lat <= 90
        assert -180 <= lon <= 180


def test_get_location_centroids_unsupported():
    """Test that unsupported aggregation levels raise ValueError."""
    with pytest.raises(ValueError, match="not supported"):
        get_location_centroids(agg_level='county')


def test_haversine_distance_same_point():
    """Test that distance from a point to itself is zero."""
    coord = (40.0, -75.0)
    distance = haversine_distance(coord, coord)
    assert distance == 0.0


def test_haversine_distance_known_values():
    """Test haversine distance with known approximate values."""
    # New York (40.7128° N, 74.0060° W) to Los Angeles (34.0522° N, 118.2437° W)
    # Approximate great circle distance: ~3,944 km
    ny = (40.7128, -74.0060)
    la = (34.0522, -118.2437)

    distance = haversine_distance(ny, la)

    # Check within 50km of expected (accounts for approximation)
    assert 3900 < distance < 4000


def test_haversine_distance_symmetric():
    """Test that distance is symmetric."""
    coord1 = (40.0, -75.0)
    coord2 = (42.0, -71.0)

    dist1 = haversine_distance(coord1, coord2)
    dist2 = haversine_distance(coord2, coord1)

    assert abs(dist1 - dist2) < 1e-10


def test_compute_bearing_north():
    """Test bearing calculation for due north direction."""
    origin = (40.0, -75.0)
    north = (41.0, -75.0)

    bearing = compute_bearing(origin, north)

    # Should be approximately 0° (North)
    assert abs(bearing - 0.0) < 5


def test_compute_bearing_east():
    """Test bearing calculation for due east direction."""
    origin = (40.0, -75.0)
    east = (40.0, -74.0)

    bearing = compute_bearing(origin, east)

    # Should be approximately 90° (East)
    assert abs(bearing - 90.0) < 5


def test_compute_bearing_south():
    """Test bearing calculation for due south direction."""
    origin = (40.0, -75.0)
    south = (39.0, -75.0)

    bearing = compute_bearing(origin, south)

    # Should be approximately 180° (South)
    assert abs(bearing - 180.0) < 5


def test_compute_bearing_west():
    """Test bearing calculation for due west direction."""
    origin = (40.0, -75.0)
    west = (40.0, -76.0)

    bearing = compute_bearing(origin, west)

    # Should be approximately 270° (West)
    assert abs(bearing - 270.0) < 5


def test_compute_bearing_range():
    """Test that bearing is always in [0, 360) range."""
    coords = [
        (40.0, -75.0),
        (35.0, -80.0),
        (45.0, -70.0),
        (38.0, -78.0)
    ]

    for i, coord1 in enumerate(coords):
        for j, coord2 in enumerate(coords):
            if i != j:
                bearing = compute_bearing(coord1, coord2)
                assert 0 <= bearing < 360


def test_get_directional_neighbors_north():
    """Test finding neighbors to the north."""
    # Create simple test coordinates
    coords = {
        'origin': (40.0, -75.0),
        'north1': (42.0, -75.0),  # Due north
        'north2': (41.5, -74.8),  # North-ish
        'south': (38.0, -75.0),   # Due south (should not be included)
        'east': (40.0, -73.0),    # Due east (should not be included)
    }

    neighbors = get_directional_neighbors(
        origin_loc='origin',
        origin_coord=coords['origin'],
        all_coords=coords,
        direction='N',
        max_distance_km=1000
    )

    # Should find neighbors to the north
    neighbor_locs = [loc for loc, _ in neighbors]
    assert 'north1' in neighbor_locs
    assert 'south' not in neighbor_locs
    assert 'east' not in neighbor_locs


def test_get_directional_neighbors_max_distance():
    """Test that max_distance_km filters out distant neighbors."""
    coords = {
        'origin': (40.0, -75.0),
        'close': (40.1, -75.0),  # Very close (~11 km)
        'far': (45.0, -75.0),    # Far away (~550 km)
    }

    # With large max distance, should find both
    neighbors_large = get_directional_neighbors(
        origin_loc='origin',
        origin_coord=coords['origin'],
        all_coords=coords,
        direction='N',
        max_distance_km=1000
    )
    assert len(neighbors_large) == 2

    # With small max distance, should only find close one
    neighbors_small = get_directional_neighbors(
        origin_loc='origin',
        origin_coord=coords['origin'],
        all_coords=coords,
        direction='N',
        max_distance_km=100
    )
    assert len(neighbors_small) == 1
    assert neighbors_small[0][0] == 'close'


def test_get_directional_neighbors_sorted_by_distance():
    """Test that neighbors are sorted by distance (nearest first)."""
    coords = {
        'origin': (40.0, -75.0),
        'close': (40.5, -75.0),
        'medium': (41.0, -75.0),
        'far': (42.0, -75.0),
    }

    neighbors = get_directional_neighbors(
        origin_loc='origin',
        origin_coord=coords['origin'],
        all_coords=coords,
        direction='N',
        max_distance_km=1000
    )

    # Extract distances
    distances = [dist for _, dist in neighbors]

    # Should be sorted in ascending order
    assert distances == sorted(distances)


def test_get_directional_neighbors_invalid_direction():
    """Test that invalid direction raises ValueError."""
    coords = {'origin': (40.0, -75.0)}

    with pytest.raises(ValueError, match="Invalid direction"):
        get_directional_neighbors(
            origin_loc='origin',
            origin_coord=coords['origin'],
            all_coords=coords,
            direction='INVALID',
            max_distance_km=1000
        )


def test_validate_wave_directions_valid():
    """Test validation with valid directions."""
    # Should not raise
    validate_wave_directions(['N', 'S', 'E', 'W'])
    validate_wave_directions(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
    validate_wave_directions(['NE', 'SW'])


def test_validate_wave_directions_invalid():
    """Test validation with invalid directions."""
    with pytest.raises(ValueError, match="Invalid direction"):
        validate_wave_directions(['N', 'INVALID', 'S'])


def test_validate_wave_directions_opposite_warning():
    """Test that opposite directions trigger a warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validate_wave_directions(['N', 'S'])

        # Should have generated a warning about opposite directions
        assert len(w) >= 1
        assert "opposite" in str(w[0].message).lower()


def test_direction_angles_coverage():
    """Test that all 8 directions are defined."""
    expected_directions = {'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'}
    assert set(DIRECTION_ANGLES.keys()) == expected_directions


def test_direction_angles_values():
    """Test that direction angles are correct."""
    assert DIRECTION_ANGLES['N'] == 0
    assert DIRECTION_ANGLES['NE'] == 45
    assert DIRECTION_ANGLES['E'] == 90
    assert DIRECTION_ANGLES['SE'] == 135
    assert DIRECTION_ANGLES['S'] == 180
    assert DIRECTION_ANGLES['SW'] == 225
    assert DIRECTION_ANGLES['W'] == 270
    assert DIRECTION_ANGLES['NW'] == 315


def test_get_directional_neighbors_with_real_states():
    """Test directional neighbors using real state centroids."""
    coords = get_location_centroids('state')

    # Pennsylvania (42) should have neighbors in all directions
    pa_coord = coords['42']

    # Check NE direction - should include NY (36)
    # NY is actually northeast of PA (bearing ~38°), not due north
    ne_neighbors = get_directional_neighbors(
        origin_loc='42',
        origin_coord=pa_coord,
        all_coords=coords,
        direction='NE',
        max_distance_km=500
    )

    ne_locs = [loc for loc, _ in ne_neighbors]
    assert '36' in ne_locs  # New York is northeast of Pennsylvania

    # Check South direction - should include MD (24) or WV (54)
    south_neighbors = get_directional_neighbors(
        origin_loc='42',
        origin_coord=pa_coord,
        all_coords=coords,
        direction='S',
        max_distance_km=500
    )

    south_locs = [loc for loc, _ in south_neighbors]
    # At least one southern neighbor
    assert len(south_locs) > 0
    # NY should not be in south neighbors
    assert '36' not in south_locs
