import pytest

from idmodels.config import (
    PoolingStrategy,
    PowerTransform,
    SARIXFourierModelConfig,
    SARIXModelConfig,
    SourceType,
)


def _make_config(config_class=SARIXFourierModelConfig, **kwargs):
    return config_class(model_name="sarix_test", main_source=SourceType.NHSN, fit_locations_separately=False,
                        power_transform=PowerTransform.FOURTH_ROOT, **kwargs)


def test_defaults_are_unshared_without_intercept():
    config = _make_config()
    assert config.theta_pooling is PoolingStrategy.UNSHARED
    assert config.sigma_pooling is PoolingStrategy.UNSHARED
    assert config.fourier_pooling is PoolingStrategy.UNSHARED
    assert not config.include_intercept
    assert config.intercept_pooling is None


@pytest.mark.parametrize("field", ["theta_pooling", "sigma_pooling", "fourier_pooling", "intercept_pooling"])
@pytest.mark.parametrize("value", [PoolingStrategy.NONE, "none"])
def test_none_pooling_is_deprecated_alias_for_unshared(field, value):
    kwargs = {field: value}
    if field == "intercept_pooling":
        kwargs["include_intercept"] = True
    with pytest.warns(FutureWarning, match=f"{field}=PoolingStrategy.NONE is deprecated"):
        config = _make_config(**kwargs)
    assert getattr(config, field) is PoolingStrategy.UNSHARED


def test_string_pooling_values_are_converted_to_enum():
    config = _make_config(theta_pooling="shared", include_intercept=True, intercept_pooling="partial")
    assert config.theta_pooling is PoolingStrategy.SHARED
    assert config.intercept_pooling is PoolingStrategy.PARTIAL


@pytest.mark.parametrize("field", ["theta_pooling", "sigma_pooling", "fourier_pooling"])
def test_partial_pooling_only_allowed_for_intercept(field):
    with pytest.raises(ValueError, match=f"{field} must be PoolingStrategy.UNSHARED or PoolingStrategy.SHARED"):
        _make_config(**{field: PoolingStrategy.PARTIAL})


@pytest.mark.parametrize("intercept_pooling",
                         [PoolingStrategy.UNSHARED, PoolingStrategy.SHARED, PoolingStrategy.PARTIAL])
def test_intercept_pooling_values(intercept_pooling):
    config = _make_config(SARIXModelConfig, include_intercept=True, intercept_pooling=intercept_pooling)
    assert config.intercept_pooling is intercept_pooling


def test_include_intercept_requires_intercept_pooling():
    with pytest.raises(ValueError, match="intercept_pooling must be set when include_intercept is True"):
        _make_config(include_intercept=True)


def test_intercept_pooling_requires_include_intercept():
    with pytest.raises(ValueError, match="intercept_pooling must be None when include_intercept is False"):
        _make_config(intercept_pooling=PoolingStrategy.SHARED)


def test_invalid_pooling_value_raises():
    with pytest.raises(ValueError, match="'pooled' is not a valid PoolingStrategy"):
        _make_config(theta_pooling="pooled")
