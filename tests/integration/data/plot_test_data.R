# You can use this script to verify that the data objects used as expected
# outcomes for integration tests are reasonable.
# This script expects to be run from the repository root.

library(hubData)
library(hubVis)
library(readr)
library(lubridate)

ref_date <- as.Date("2024-01-06")

locations <- read.csv("https://raw.githubusercontent.com/cdcepi/FluSight-forecast-hub/refs/heads/main/auxiliary-data/locations.csv")

forecasts <- dplyr::bind_rows(
  read.csv("tests/integration/data/UMass-gbqr_no_reporting_adj/2024-01-06-UMass-gbqr_no_reporting_adj.csv") |>
    dplyr::mutate(model_id = "UMass-gbqr"),
  read.csv("tests/integration/data/UMass-sarix_p6_4rt_thetashared_sigmanone/2024-01-06-UMass-sarix_p6_4rt_thetashared_sigmanone.csv") |>
    dplyr::mutate(model_id = "UMass-sarix")
) |>
  dplyr::left_join(locations)

target_data <- readr::read_csv("https://raw.githubusercontent.com/cdcepi/FluSight-forecast-hub/main/target-data/target-hospital-admissions.csv")

data_start <- ref_date - 6 * 7
data_end <- ref_date + 6 * 7

p <- plot_step_ahead_model_output(
  forecasts,
  target_data |>
    dplyr::filter(date >= data_start, date <= data_end) |>
    dplyr::mutate(observation = value),
  x_col_name = "target_end_date",
  x_target_col_name = "date",
  intervals = 0.95,
  facet = "location_name",
  facet_scales = "free_y",
  facet_nrow = 8,
  use_median_as_point = TRUE,
  interactive = FALSE,
  show_plot = FALSE
)

p
