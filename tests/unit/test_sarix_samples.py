import numpy as np
import pytest

from idmodels.sarix import _draws_to_samples_df, _select_draw_indices


def test_select_draw_indices_is_reproducible_and_without_replacement():
    inds = _select_draw_indices(num_draws=200, num_sample_forecasts=50, seed=123)
    assert len(set(inds)) == 50
    assert all(0 <= i < 200 for i in inds)
    np.testing.assert_array_equal(inds, _select_draw_indices(num_draws=200, num_sample_forecasts=50, seed=123))


def test_select_draw_indices_raises_when_too_few_draws():
    with pytest.raises(ValueError, match="exceeds the 10 available SARIX posterior draws"):
        _select_draw_indices(num_draws=10, num_sample_forecasts=11, seed=123)


def test_draws_to_samples_df_uses_same_draw_for_every_location():
    # value encodes (draw, location, horizon) as draw * 100 + location * 10 + horizon
    num_draws, n_locations, max_horizon = 6, 3, 4
    draws = np.fromfunction(lambda d, loc, h: d * 100 + loc * 10 + h, (num_draws, n_locations, max_horizon))
    unique_ids = np.array(["nationalUS", "state01", "hsa1"])
    draw_inds = np.array([4, 1])

    samples_df = _draws_to_samples_df(draws, unique_ids, draw_inds)

    assert len(samples_df) == len(draw_inds) * n_locations * max_horizon
    for _, row in samples_df.iterrows():
        loc_ind = list(unique_ids).index(row["unique_id"])
        expected = draw_inds[row["sample_index"]] * 100 + loc_ind * 10 + (row["horizon"] - 1)
        assert row["value"] == expected
    assert sorted(samples_df["sample_index"].unique()) == [0, 1]
    assert sorted(samples_df["horizon"].unique()) == [1, 2, 3, 4]
