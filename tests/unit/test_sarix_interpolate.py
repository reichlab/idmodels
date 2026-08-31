import numpy as np
import pandas as pd

from idmodels.sarix import _interpolate_by_location


def test_interpolate_by_location_does_not_blend_across_locations():
    """Reproduces the incident bug: interpolating the whole concatenated frame at once fills a
    trailing NaN in one location's block from the next location's leading row (e.g. a state's
    missing final week getting blended with the next state's first observation). Grouping by
    unique_id must leave a genuine trailing gap as NaN instead of fabricating a blended value.
    """
    df = pd.DataFrame({
        "unique_id": ["stateA"] * 3 + ["stateB"] * 3,
        "x": [1.0, 2.0, np.nan, 10.0, 11.0, 12.0],
    })

    out = _interpolate_by_location(df, ["x"])

    assert pd.isna(out.loc[out["unique_id"] == "stateA", "x"].iloc[-1])
    assert out.loc[out["unique_id"] == "stateB", "x"].tolist() == [10.0, 11.0, 12.0]


def test_interpolate_by_location_fills_genuine_interior_gap():
    """A real interior gap within a single location's own series should still be interpolated."""
    df = pd.DataFrame({
        "unique_id": ["stateA"] * 4,
        "x": [1.0, np.nan, 3.0, 4.0],
    })

    out = _interpolate_by_location(df, ["x"])

    assert out["x"].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_interpolate_by_location_independent_gaps_across_locations():
    """A gap in one location's series must not be influenced by another location's values."""
    df = pd.DataFrame({
        "unique_id": ["stateA"] * 3 + ["stateB"] * 3,
        "x": [1.0, np.nan, 3.0, 100.0, np.nan, 300.0],
    })

    out = _interpolate_by_location(df, ["x"])

    assert out.loc[out["unique_id"] == "stateA", "x"].tolist() == [1.0, 2.0, 3.0]
    assert out.loc[out["unique_id"] == "stateB", "x"].tolist() == [100.0, 200.0, 300.0]


def test_interpolate_by_location_does_not_mutate_input():
    df = pd.DataFrame({
        "unique_id": ["stateA"] * 3,
        "x": [1.0, np.nan, 3.0],
    })
    original = df.copy()

    _interpolate_by_location(df, ["x"])

    pd.testing.assert_frame_equal(df, original)
