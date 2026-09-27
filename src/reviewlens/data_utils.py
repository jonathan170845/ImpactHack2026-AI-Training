"""Small, notebook-independent helpers for data audit and preparation."""

import hashlib
import html
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


def validate_dataset_files(paths: Mapping[str, Path]) -> None:
    """Raise one readable error listing all missing dataset files."""
    missing = [f"data/raw/{path.name}" for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Expected dataset not found:\n" + "\n".join(missing))


def load_json_dataframe(path: str | Path, records_key: str | None = None) -> pd.DataFrame:
    """Read a UTF-8 JSON array of records, optionally under an explicit object key."""
    path = Path(path)
    with path.open(encoding="utf-8-sig") as source:
        records = json.load(source)
    if records_key is not None:
        if not isinstance(records, dict) or records_key not in records:
            raise ValueError(f"{path.name}: expected JSON object key {records_key!r}.")
        records = records[records_key]
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise ValueError(f"{path.name}: expected a JSON array of record objects.")
    return pd.DataFrame.from_records(records)


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load JSON or CSV without treating literal review strings such as 'NA' as null."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        return load_json_dataframe(path)
    if path.suffix.lower() == ".csv":
        # Identifiers and localized sold values must survive CSV type inference.
        columns = pd.read_csv(path, encoding="utf-8-sig", nrows=0).columns
        text_columns = {
            "comment", "review", "text_", "text", "translate", "category",
            "sentiment", "label", "product_id", "shop_id", "product_name",
            "product_url", "sold", "clean_text",
        }
        return pd.read_csv(
            path,
            encoding="utf-8-sig",
            keep_default_na=False,
            na_values=[""],
            dtype={column: "string" for column in columns if column in text_columns},
        )
    raise ValueError(f"Unsupported dataset extension: {path.suffix}")


def normalize_text(text: Any, replace_urls: bool = False) -> str:
    """Decode entities, lowercase and collapse whitespace; preserve emoji and punctuation.

    Missing scalar values become an empty string. URL replacement is optional and
    off by default so normalization does not collapse reviews with different URLs.
    """
    if text is None or pd.isna(text):
        return ""
    normalized = html.unescape(str(text)).lower()
    if replace_urls:
        # Keep sentence punctuation following an otherwise ordinary URL.
        def replace_match(match: re.Match) -> str:
            url = match.group(0)
            trimmed = url.rstrip(".,!?;:)]}")
            return "<url>" + url[len(trimmed):]

        normalized = re.sub(r"(?:https?://|www\.)[^\s<>]+", replace_match, normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def require_columns(dataframe: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    """Fail with the actual schema when required pipeline columns are unavailable."""
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}; available: {list(dataframe.columns)}")


def audit_dataframe(dataframe: pd.DataFrame, text_column: str) -> dict[str, Any]:
    """Return schema, null, full-row duplicate and unique source-text statistics."""
    require_columns(dataframe, [text_column], "audit_dataframe")
    return {
        "rows": len(dataframe),
        "columns_count": len(dataframe.columns),
        "columns": list(dataframe.columns),
        "dtypes": dataframe.dtypes.astype(str).to_dict(),
        "missing_values": dataframe.isna().sum().astype(int).to_dict(),
        "exact_duplicates": int(dataframe.duplicated().sum()),
        "unique_text_count": int(dataframe[text_column].nunique(dropna=True)),
    }


def add_clean_text(dataframe: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """Return a copy with clean_text, preserving all original columns and values."""
    require_columns(dataframe, [text_column], "add_clean_text")
    result = dataframe.copy()
    result["clean_text"] = result[text_column].map(normalize_text)
    return result


def duplicate_summary(dataframe: pd.DataFrame) -> dict[str, int]:
    """Count normalized-text duplicates beyond the first occurrence, including blanks."""
    require_columns(dataframe, ["clean_text"], "duplicate_summary")
    return {
        "rows_total": len(dataframe),
        "unique_clean_text": int(dataframe["clean_text"].nunique()),
        "duplicate_clean_text_count": int(dataframe["clean_text"].duplicated().sum()),
        "empty_clean_text_count": int(dataframe["clean_text"].eq("").sum()),
    }


def find_label_conflicts(dataframe: pd.DataFrame, label_column: str) -> pd.DataFrame:
    """Return original rows where a nonempty normalized text has differing labels."""
    require_columns(dataframe, ["clean_text", label_column], "find_label_conflicts")
    label_counts = dataframe.groupby("clean_text", dropna=False)[label_column].nunique(dropna=False)
    conflicting_texts = label_counts.index[label_counts.gt(1)]
    return dataframe.loc[
        dataframe["clean_text"].isin(conflicting_texts) & dataframe["clean_text"].ne("")
    ].copy()


def label_distribution(dataframe: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Count labels (including nulls), skipping optional columns that are absent."""
    records = []
    for column in columns:
        if column not in dataframe.columns:
            continue
        for value, count in dataframe[column].value_counts(dropna=False).items():
            records.append({
                "column": column,
                "value": "<missing>" if pd.isna(value) else str(value),
                "count": int(count),
            })
    return pd.DataFrame(records, columns=["column", "value", "count"])


def find_category_variants(dataframe: pd.DataFrame, label_column: str) -> pd.DataFrame:
    """List text/label groups with multiple original categories without modifying them."""
    require_columns(dataframe, ["clean_text", label_column, "category"], "find_category_variants")
    variants = (
        dataframe.groupby(["clean_text", label_column], dropna=False)["category"]
        .agg(lambda values: json.dumps(sorted(values.dropna().unique().tolist()), ensure_ascii=False))
        .rename("categories")
        .reset_index()
    )
    variants["category_count"] = variants["categories"].map(lambda value: len(json.loads(value)))
    return variants.loc[variants["category_count"].gt(1)].reset_index(drop=True)


def clean_dataframe(dataframe: pd.DataFrame, deduplication_keys: Iterable[str] | None) -> pd.DataFrame:
    """Keep the first row per explicit key, or every row if keys are None.

    Ratings never filter rows. Blank text rows are retained individually for review,
    because missing text is not evidence that two observations are duplicates.
    No labels are remapped and no metadata is aggregated or overwritten.
    """
    require_columns(dataframe, ["clean_text"], "clean_dataframe")
    if deduplication_keys is None:
        return dataframe.copy().reset_index(drop=True)
    keys = list(deduplication_keys)
    if not keys:
        raise ValueError("Use None to retain every row, or provide deduplication keys.")
    require_columns(dataframe, keys, "clean_dataframe")
    duplicate_mask = dataframe.duplicated(subset=keys, keep="first")
    blank_mask = dataframe["clean_text"].isna() | dataframe["clean_text"].eq("")
    return dataframe.loc[~duplicate_mask | blank_mask].copy().reset_index(drop=True)


def cleaning_summary(before: Mapping[str, pd.DataFrame], after: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute row retention directly from the dataframes."""
    rows = []
    for name, original in before.items():
        rows_before = len(original)
        rows_after = len(after[name])
        removed = rows_before - rows_after
        rows.append({
            "dataset": name,
            "rows_before": rows_before,
            "rows_after": rows_after,
            "removed": removed,
            "removal_percent": round(100 * removed / rows_before, 2) if rows_before else 0.0,
        })
    return pd.DataFrame(rows)


def save_csv(dataframe: pd.DataFrame, path: str | Path) -> Path:
    """Write an index-free UTF-8 BOM CSV, creating the destination directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def validate_saved_csv(path: str | Path, expected: pd.DataFrame) -> pd.DataFrame:
    """Reopen a processed CSV and validate schema, rows and every serialized value.

    Read all columns as strings to preserve IDs, literal 'NA', punctuation and
    leading zeros. CSV has no dtype/null schema; empty source cells serialize as
    empty strings. This comparison intentionally checks that CSV representation.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Processed CSV not found: {path}")
    actual = pd.read_csv(path, encoding="utf-8-sig", dtype="string", keep_default_na=False)
    if actual.empty:
        raise ValueError(f"{path.name}: processed dataset is empty.")
    if len(actual) != len(expected):
        raise ValueError(f"{path.name}: row count differs ({len(actual)} != {len(expected)}).")
    if list(actual.columns) != list(expected.columns):
        raise ValueError(f"{path.name}: saved columns do not match the expected schema.")
    require_columns(actual, ["clean_text"], path.name)
    if actual["clean_text"].eq("").all():
        raise ValueError(f"{path.name}: all clean_text values are empty.")
    expected_csv = expected.reset_index(drop=True).astype("string").fillna("")
    pd.testing.assert_frame_equal(actual, expected_csv, check_dtype=False, obj=path.name)
    return actual


def file_sha256(path: str | Path) -> str:
    """Hash the raw file bytes so an audit can verify that sources stay unchanged."""
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()
