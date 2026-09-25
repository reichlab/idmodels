"""Direct models for the seasonal peak week and peak size of influenza hospital admissions."""

from idmodels.peak.base import PeakModel
from idmodels.peak.baseline import PeakBaselineModel
from idmodels.peak.gbqr import PeakGBQRModel
from idmodels.peak.kcde import PeakKCDEModel

__all__ = ["PeakBaselineModel", "PeakGBQRModel", "PeakKCDEModel", "PeakModel"]
