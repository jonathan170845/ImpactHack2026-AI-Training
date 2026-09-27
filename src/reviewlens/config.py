"""Portable project paths and dataset-specific preparation rules."""

from pathlib import Path


def find_project_root(start: str | Path | None = None) -> Path:
    """Find ReviewLens from this module, the project root, or a nested directory."""
    location = Path(start).resolve() if start is not None else Path(__file__).resolve()
    if location.is_file():
        location = location.parent
    for candidate in (location, *location.parents):
        if (candidate / "src" / "reviewlens" / "config.py").is_file() and (
            candidate / "data" / "raw"
        ).is_dir():
            return candidate
    raise FileNotFoundError(
        "ReviewLens project root not found. Start from the workspace root "
        "or one of its subdirectories; expected src/reviewlens/config.py and data/raw/."
    )


PROJECT_ROOT = find_project_root()
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = OUTPUT_DIR / "reports"

DATASET_PATHS = {
    "simple": DATA_RAW_DIR / "simple.json",
    "challenge": DATA_RAW_DIR / "challange.json",
    "labeled": DATA_RAW_DIR / "labeledReview.datasetFix.json",
    "fake": DATA_RAW_DIR / "fake_reviews.csv",
    "tokopedia": DATA_RAW_DIR / "tokopedia-product-reviews-2019.csv",
}
PROCESSED_PATHS = {
    name: DATA_PROCESSED_DIR / f"{name}_clean.csv" for name in DATASET_PATHS
}
TEXT_COLUMNS = {
    "simple": "comment",
    "challenge": "comment",
    "labeled": "review",
    "fake": "text_",
    "tokopedia": "text",
}
LABEL_COLUMNS = {
    "simple": "sentiment",
    "challenge": "sentiment",
    "labeled": "sentimen",
    "fake": "label",
}
LABEL_AUDIT_COLUMNS = {
    "simple": ("sentiment",),
    "challenge": ("sentiment", "category"),
    "labeled": ("sentimen",),
    "fake": ("label",),
    "tokopedia": ("rating", "category"),
}
DEDUPLICATION_KEYS = {
    "simple": ("clean_text", "sentiment"),
    # Keep category variants in the hard evaluation set, as requested after audit.
    "challenge": ("clean_text", "sentiment", "category"),
    "labeled": ("clean_text", "sentimen"),
    "fake": ("clean_text", "label"),
    "tokopedia": None,
}
