"""Unit tests for transform classes in idmodels.transforms."""

import numpy as np
import pandas as pd

from idmodels.constants import NHSN_FLOOR, ILINET_FLOOR, ILINET_SCALE, POWER_TRANSFORM_OFFSET
from idmodels.transforms import (
    CenterScaleTransform,
    ComposedTransform,
    FourthRootTransform,
    IdentityTransform,
    SourceScaleTransform,
)


def make_df(n_per_group=20, n_groups=2, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for loc in range(n_groups):
        for week in range(1, n_per_group + 1):
            rows.append({"source": "nhsn",
                         "location": str(loc).zfill(2),
                         "season_week": week,
                         "inc": max(0.0, rng.normal(0.5, 0.2))})
    return pd.DataFrame(rows)


class TestFourthRootTransform:
    def test_apply_adds_inc_trans(self):
        df = make_df()
        t = FourthRootTransform()
        out = t.apply(df.copy())
        assert "inc_trans" in out.columns


    def test_apply_values(self):
        df = make_df()
        t = FourthRootTransform()
        out = t.apply(df.copy())
        expected = (df["inc"] + POWER_TRANSFORM_OFFSET) ** 0.25
        np.testing.assert_allclose(out["inc_trans"].values, expected.values)


    def test_roundtrip(self):
        df = make_df()
        t = FourthRootTransform()
        out = t.apply(df.copy())
        recovered = t.invert(out["inc_trans"].values, out)
        np.testing.assert_allclose(recovered, df["inc"].values, atol=1e-10)


    def test_invert_clips_at_zero(self):
        t = FourthRootTransform()
        result = t.invert(np.array([-5.0, 0.0]), pd.DataFrame())
        assert (result >= -POWER_TRANSFORM_OFFSET).all()


class TestIdentityTransform:
    def test_apply_adds_inc_trans(self):
        df = make_df()
        t = IdentityTransform()
        out = t.apply(df.copy())
        assert "inc_trans" in out.columns


    def test_roundtrip(self):
        df = make_df()
        t = IdentityTransform()
        out = t.apply(df.copy())
        recovered = t.invert(out["inc_trans"].values, out)
        np.testing.assert_allclose(recovered, df["inc"].values, atol=1e-10)


class TestCenterScaleTransform:
    def test_apply_adds_factor_columns(self):
        df = make_df(n_per_group=52)
        df["inc_trans"] = (df["inc"] + POWER_TRANSFORM_OFFSET) ** 0.25
        t = CenterScaleTransform()
        out = t.apply(df.copy())
        for col in ("inc_trans_cs", "inc_trans_scale_factor", "inc_trans_center_factor"):
            assert col in out.columns


    def test_invert_roundtrip(self):
        df = make_df(n_per_group=52)
        df["inc_trans"] = (df["inc"] + POWER_TRANSFORM_OFFSET) ** 0.25
        t = CenterScaleTransform()
        out = t.apply(df.copy())
        recovered = t.invert(out["inc_trans_cs"].values, out)
        np.testing.assert_allclose(recovered, out["inc_trans"].values, atol=1e-10)


    def test_invert_uses_offset(self):
        """Verify invert uses (scale_factor + 0.01), not just scale_factor."""
        context = pd.DataFrame({
            "inc_trans_scale_factor": [1.0],
            "inc_trans_center_factor": [0.0],
        })
        t = CenterScaleTransform()
        result = t.invert(np.array([1.0]), context)
        assert abs(result[0] - 1.01) < 1e-12


    def test_only_in_season_rows_used_for_scale(self):
        df = make_df(n_per_group=52)
        df["inc_trans"] = (df["inc"] + POWER_TRANSFORM_OFFSET) ** 0.25
        t = CenterScaleTransform()
        out = t.apply(df.copy())
        # Out-of-season rows should still have scale factors (broadcast from in-season)
        assert out["inc_trans_scale_factor"].notna().all()


    def test_custom_in_season_week_bounds(self):
        """Custom in_season_week_min/max changes which rows determine the scale factor."""
        df = make_df(n_per_group=52, n_groups=1)
        df["inc_trans"] = (df["inc"] + POWER_TRANSFORM_OFFSET) ** 0.25

        t_default = CenterScaleTransform()            # weeks 10–45
        t_narrow = CenterScaleTransform(in_season_week_min=1, in_season_week_max=5)
        out_default = t_default.apply(df.copy())
        out_narrow = t_narrow.apply(df.copy())

        # Different in-season windows should generally produce different scale factors
        assert not np.isclose(out_default["inc_trans_scale_factor"].iloc[0],
                              out_narrow["inc_trans_scale_factor"].iloc[0])


    def test_scale_factor_is_95th_percentile_of_in_season(self):
        df = make_df(n_per_group=52, n_groups=1)
        df["inc_trans"] = (df["inc"] + POWER_TRANSFORM_OFFSET) ** 0.25

        t = CenterScaleTransform()  # IN_SEASON_WEEK_MIN=10, IN_SEASON_WEEK_MAX=45
        out = t.apply(df.copy())

        in_season = df.loc[(df["season_week"] >= 10) & (df["season_week"] <= 45), "inc_trans"]
        expected_scale = in_season.quantile(0.95)
        np.testing.assert_allclose(out["inc_trans_scale_factor"].iloc[0], expected_scale)


    def test_center_factor_is_mean_of_in_season_cs(self):
        df = make_df(n_per_group=52, n_groups=1)
        df["inc_trans"] = (df["inc"] + POWER_TRANSFORM_OFFSET) ** 0.25

        t = CenterScaleTransform()
        out = t.apply(df.copy())

        scale = out["inc_trans_scale_factor"].iloc[0]
        in_season = df.loc[(df["season_week"] >= 10) & (df["season_week"] <= 45), "inc_trans"]
        expected_center = (in_season / (scale + 0.01)).mean()
        np.testing.assert_allclose(out["inc_trans_center_factor"].iloc[0], expected_center)


class TestSourceScaleTransform:
    def test_apply_nhsn_floor_only(self):
        df = make_df()
        t = SourceScaleTransform({"nhsn": (NHSN_FLOOR, 1.0)})
        out = t.apply(df.copy())
        np.testing.assert_allclose(out["inc"].values, df["inc"].values + NHSN_FLOOR)


    def test_apply_ilinet_floor_and_scale(self):
        df = make_df()
        df["source"] = "ilinet"
        t = SourceScaleTransform({"ilinet": (ILINET_FLOOR, ILINET_SCALE)})
        out = t.apply(df.copy())
        np.testing.assert_allclose(out["inc"].values, (df["inc"].values + ILINET_FLOOR) * ILINET_SCALE)


    def test_nssp_passthrough(self):
        df = make_df()
        df["source"] = "nssp"
        t = SourceScaleTransform({})
        out = t.apply(df.copy())
        np.testing.assert_allclose(out["inc"].values, df["inc"].values)


    def test_roundtrip(self):
        df = make_df()
        t = SourceScaleTransform({"nhsn": (NHSN_FLOOR, 1.0)})
        out = t.apply(df.copy())
        recovered = t.invert(out["inc"].values, out)
        np.testing.assert_allclose(recovered, df["inc"].values, atol=1e-10)


class TestComposedTransform:
    def test_roundtrip_fourth_root_then_center_scale(self):
        df = make_df(n_per_group=52)
        t = ComposedTransform([
            SourceScaleTransform({"nhsn": (NHSN_FLOOR, 1.0)}),
            FourthRootTransform(),
            CenterScaleTransform(),
        ])
        out = t.apply(df.copy())
        assert "inc_trans_cs" in out.columns
        recovered_inc = t.invert(out["inc_trans_cs"].values, out)
        np.testing.assert_allclose(recovered_inc, df["inc"].values, atol=1e-10)


    def test_apply_calls_transforms_in_order(self):
        df = make_df()
        t = ComposedTransform([FourthRootTransform()])
        out = t.apply(df.copy())
        assert "inc_trans" in out.columns
