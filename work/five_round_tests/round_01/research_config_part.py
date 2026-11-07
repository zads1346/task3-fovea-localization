@dataclass
class ResearchConfig(Config):
    OUTPUT_DIR: str = "/kaggle/working/fovea_optimized"
    SUBMISSION_PATH: str = "/kaggle/working/submission.csv"
    PRETRAINED: bool = True
    ALLOW_WEIGHT_DOWNLOAD: bool = False
    PRETRAINED_WEIGHTS_PATH: str | None = None
    N_FOLDS: int = 5
    CALIBRATION_FRACTION: float = 0.25
    EARLY_STOP_FRACTION: float = 0.20
    INNER_CROP_FOLDS: int = 3
    FINE_TRAIN_CROPS: str = "oof"  # "jitter" is a cheaper, explicitly different ablation.
    COARSE_EPOCHS: int = 50
    FINE_EPOCHS: int = 40
    PATIENCE: int = 8
    INPUT_SIZE: int = 512
    HEATMAP_SIZE: int = 128
    FINE_INPUT_SIZE: int = 384
    FINE_HEATMAP_SIZE: int = 96
    BATCH_SIZE: int = 8
    NUM_WORKERS: int = 2
    ROI_FRACTIONS: tuple = (0.20, 0.30, 0.40)
    TRAIN_CENTER_JITTER: float = 0.20
    FIXED_ROI_FRACTION: float = 0.30
    ROI_MIN_FRACTION: float = 0.15
    ROI_MAX_FRACTION: float = 0.60
    ROI_MARGIN_PX: float = 16.0
    ROI_COVERAGE: float = 0.90
    RAW_RADIUS_MULTIPLIER: float = 2.0
    UNCERTAINTY_FLOOR_PX: float = 1.0
    FALLBACK_EDGE_THRESHOLDS: tuple = (0.0, 0.03, 0.08)
    FALLBACK_DISAGREEMENT_THRESHOLDS: tuple = (2.0, 4.0, 1e9)
    EXPERIMENTS: tuple = ("B0", "B1", "B2", "B3", "B4", "C1")
    SUBMISSION_EXPERIMENT: str = "B4"  # Preselected; never auto-pick the best outer OOF score.
    BOOTSTRAP_REPEATS: int = 1000
    REUSE_CHECKPOINTS: bool = True  # Reuse only complete, matching outer-fold results.
    SHOW_PLOTS: bool = False
    SMOKE_TEST: bool = False
    SMOKE_FOLDS: int = 2
    SMOKE_INPUT_SIZE: int = 64
    SMOKE_EPOCHS: int = 1
    REQUIRE_GPU: bool = True


CFG = ResearchConfig()
RESEARCH_VERSION = "3.0.0"
EXPERIMENT_DESCRIPTIONS = {
    "B0": "Single-stage coarse heatmap + DSNT",
    "B1": "Coarse-to-fine, fixed ROI",
    "B2": "Coarse-to-fine, uncalibrated uncertainty ROI",
    "B3": "Coarse-to-fine, empirically calibrated uncertainty ROI",
    "B4": "B3 plus calibration-selected coarse fallback",
    "C1": "Fixed ROI matched to B3 mean calibration crop area",
}
