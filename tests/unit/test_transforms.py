"""Unit tests for transform classes in idmodels.transforms."""

import numpy as np
import pandas as pd

from idmodels.constants import POWER_TRANSFORM_OFFSET
from idmodels.transforms import (
    CenterScaleTransform,
    ComposedTransform,
    FourthRootTransform,
    IdentityTransform,
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


    def test_additive_shift(self):
        df = make_df()
        shift = 0.316
        t = FourthRootTransform(additive_shift=shift)
        out = t.apply(df.copy())
        expected = (df["inc"] + shift + POWER_TRANSFORM_OFFSET) ** 0.25
        np.testing.assert_allclose(out["inc_trans"].values, expected.values)


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


class TestComposedTransform:
    def test_roundtrip_fourth_root_then_center_scale(self):
        df = make_df(n_per_group=52)
        t = ComposedTransform([FourthRootTransform(), CenterScaleTransform()])
        out = t.apply(df.copy())
        assert "inc_trans_cs" in out.columns

        # invert() reverses both transforms: CenterScale then FourthRoot → back to inc
        recovered_inc = t.invert(out["inc_trans_cs"].values, out)
        np.testing.assert_allclose(recovered_inc, df["inc"].values, atol=1e-10)


    def test_apply_calls_transforms_in_order(self):
        df = make_df()
        t = ComposedTransform([FourthRootTransform()])
        out = t.apply(df.copy())
        assert "inc_trans" in out.columns
