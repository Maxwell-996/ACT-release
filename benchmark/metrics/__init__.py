from .plottable_data import (
    field_histograms,
    field_spatial_snapshots,
    make_video,
    plot_all_time_metrics,
    plot_power_spectrum_by_field,
    plot_validation_rollout_panel,
)
from .disp_tv import tv_loss_disp, tv_loss_disps
from .h1 import H1RelLoss
from .spatial import MAE, MSE, NMAE, NMSE, NRMSE, RMSE, VMSE, VRMSE, LInfinity, PearsonR
from .spectral import binned_spectral_mse
from .temporal import HistogramW1, WindowedDTW

__all__ = [
    "tv_loss_disp",
    "tv_loss_disps",
    "H1RelLoss",
    "MAE",
    "NMAE",
    "NRMSE",
    "RMSE",
    "MSE",
    "NMSE",
    "LInfinity",
    "VMSE",
    "VRMSE",
    "binned_spectral_mse",
    "PearsonR",
    "HistogramW1",
    "WindowedDTW",
]

long_time_metrics = ["VRMSE", "RMSE", "binned_spectral_mse", "PearsonR"]
validation_metric_suite = [
    # RMSE(),
    NRMSE(),
    # LInfinity(),
    VRMSE(),
    # binned_spectral_mse(),
    # PearsonR(),
]
validation_plots = [
    plot_power_spectrum_by_field,
    field_histograms,
    field_spatial_snapshots,
]
time_plots = [plot_all_time_metrics]
time_space_plots = [make_video]
