#!/usr/bin/env Rscript
# Calibration Plot — Publication-quality (minimal)

suppressPackageStartupMessages({
  library(argparse)
  library(readr)
  library(dplyr)
  library(ggplot2)
})

# ── Theme (publication style) ────────────────────────────────────────────────
theme_pub <- function(base_size = 11) {
  theme_classic(base_size = base_size) +
    theme(
      axis.title       = element_text(face = "bold"),
      axis.text        = element_text(color = "black"),
      plot.title       = element_text(face = "bold", size = base_size + 2),
      plot.subtitle    = element_text(color = "grey40"),
      legend.position  = "bottom",
      strip.background = element_blank()
    )
}

col_main      <- "#0072B2"
col_reference <- "grey50"

# ── CLI ──────────────────────────────────────────────────────────────────────
parser <- ArgumentParser(description = "Calibration plot from predictions CSV")
parser$add_argument("--predictions", required = TRUE, help = "Path to predictions.csv (y_true, y_prob)")
parser$add_argument("--output_dir", required = TRUE, help = "Output folder for the plot")
parser$add_argument("--n_groups", type = "integer", default = 10L, help = "Number of calibration groups")
args <- parser$parse_args()

dir.create(args$output_dir, recursive = TRUE, showWarnings = FALSE)

# ── Load predictions ─────────────────────────────────────────────────────────
df <- read_csv(args$predictions, show_col_types = FALSE)

# ── Build calibration table (decile-based) ───────────────────────────────────
cal_tbl <- df %>%
  mutate(cal_group = ntile(y_prob, args$n_groups)) %>%
  group_by(cal_group) %>%
  summarise(
    n              = n(),
    mean_predicted = mean(y_prob, na.rm = TRUE),
    observed       = mean(y_true, na.rm = TRUE),
    .groups        = "drop"
  )

# ── Calibration metrics ──────────────────────────────────────────────────────
fit <- glm(y_true ~ offset(qlogis(y_prob)), family = binomial(), data = df)
intercept <- round(coef(fit)[1], 3)

fit_slope <- glm(y_true ~ qlogis(y_prob), family = binomial(), data = df)
slope <- round(coef(fit_slope)[2], 3)

oe_ratio <- round(sum(df$y_true) / sum(df$y_prob), 3)

lbl <- sprintf("Slope = %.3f\nIntercept = %.3f\nO:E = %.3f\nn = %d",
               slope, intercept, oe_ratio, nrow(df))

# ── Plot ─────────────────────────────────────────────────────────────────────
p <- ggplot(cal_tbl, aes(x = mean_predicted, y = observed)) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = col_reference) +
  geom_smooth(method = "loess", se = TRUE, color = col_main, fill = col_main, alpha = 0.15) +
  geom_point(shape = 21, size = 3, fill = col_main, color = "black", stroke = 0.3) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  scale_x_continuous(breaks = seq(0, 1, 0.2)) +
  scale_y_continuous(breaks = seq(0, 1, 0.2)) +
  annotate("text", x = 0.02, y = 0.95, label = lbl,
           hjust = 0, vjust = 1, size = 3, family = "mono") +
  labs(title = "Calibration Plot",
       x = "Predicted probability",
       y = "Observed proportion") +
  theme_pub()

out_path <- file.path(args$output_dir, "calibration_plot.png")
ggsave(out_path, p, width = 7, height = 6, dpi = 300)
cat(sprintf("Saved: %s\n", out_path))

# ── MLflow logging via reticulate ────────────────────────────────────────────
cat("=== MLflow Logging ===\n")
tryCatch({
  suppressPackageStartupMessages(library(reticulate))
  use_python("/opt/pyenv/bin/python", required = TRUE)
  mlflow <- import("mlflow")
  cat("  MLflow client loaded\n")
  mlflow$log_artifact(out_path, artifact_path = "calibration")
  cat("  Logged calibration_plot.png under 'calibration/'\n")
}, error = function(e) {
  cat("  MLflow logging skipped:", conditionMessage(e), "\n")
})
