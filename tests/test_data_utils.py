"""Regression checks for decisions that can silently lose review information."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from reviewlens.config import DEDUPLICATION_KEYS, find_project_root
from reviewlens.data_utils import (
    add_clean_text, audit_dataframe, clean_dataframe, find_category_variants,
    find_label_conflicts, label_distribution, load_dataset, load_json_dataframe,
    normalize_text, save_csv, validate_dataset_files, validate_saved_csv,
)


class DataPreparationTests(unittest.TestCase):
    def test_normalization_preserves_sentiment_cues(self):
        self.assertEqual(normalize_text("  BAGUS\tsekali!!! 😍 &amp; bukan palsu\n"),
                         "bagus sekali!!! 😍 & bukan palsu")
        self.assertEqual(normalize_text("Bagus sekali... barangnya rusak semua 🙂"),
                         "bagus sekali... barangnya rusak semua 🙂")
        self.assertEqual(normalize_text(42), "42")
        for value in [None, np.nan, pd.NA, pd.NaT]:
            self.assertEqual(normalize_text(value), "")

    def test_url_replacement_is_optional_and_keeps_punctuation(self):
        text = "Bukan rusak! https://example.org/ITEM?a=1&amp;b=2."
        self.assertIn("https://example.org/item?a=1&b=2.", normalize_text(text))
        self.assertEqual(normalize_text(text, replace_urls=True), "bukan rusak! <url>.")

    def test_category_and_contradictory_label_variants_survive(self):
        original = pd.DataFrame({
            "comment": ["Oke!", "OKE!", "oke!", "oke!"],
            "sentiment": ["positive", "positive", "neutral", "positive"],
            "category": ["sarcasm", "ambiguous", "ambiguous", "sarcasm"],
            "rating": [1, 5, 3, 4],
        })
        normalized = add_clean_text(original, "comment")
        actual = clean_dataframe(normalized, DEDUPLICATION_KEYS["challenge"])
        self.assertEqual(len(actual), 3)
        self.assertEqual(actual["rating"].tolist(), [1, 5, 3])
        self.assertEqual(actual["category"].tolist(), ["sarcasm", "ambiguous", "ambiguous"])
        self.assertEqual(len(find_label_conflicts(normalized, "sentiment")), 4)
        self.assertEqual(len(find_category_variants(normalized, "sentiment")), 1)
        self.assertNotIn("clean_text", original)

    def test_empty_reviews_are_not_assumed_to_be_duplicates(self):
        dataframe = add_clean_text(pd.DataFrame({
            "review": [None, " ", "Sama", "sama", "sama"],
            "sentimen": [0, 0, 0, 0, 1],
        }), "review")
        actual = clean_dataframe(dataframe, DEDUPLICATION_KEYS["labeled"])
        self.assertEqual(len(actual), 4)
        self.assertEqual(int(actual.clean_text.eq("").sum()), 2)
        self.assertEqual(set(actual.loc[actual.clean_text.eq("sama"), "sentimen"]), {0, 1})

    def test_tokopedia_keeps_repeated_observations_and_metadata(self):
        dataframe = add_clean_text(pd.DataFrame({
            "text": ["mantap", "mantap", "mantap"],
            "product_id": ["001", "001", "002"],
            "shop_id": ["009", "009", "008"],
        }), "text")
        pd.testing.assert_frame_equal(clean_dataframe(dataframe, DEDUPLICATION_KEYS["tokopedia"]), dataframe)

    def test_csv_roundtrip_keeps_unicode_ids_literal_na_and_missing_metadata(self):
        dataframe = pd.DataFrame({
            "text": ['Bagus, "aman"! 😍\n', "NA", "null"],
            "product_id": ["0007", "0008", "0009"],
            "sold": ["3,2rb", None, "0"],
            "rating": [5.0, 3.0, 1.0],
        })
        dataframe = add_clean_text(dataframe, "text")
        with tempfile.TemporaryDirectory() as temporary:
            path = save_csv(dataframe, Path(temporary) / "processed.csv")
            self.assertTrue(path.read_bytes().startswith(b"\xef\xbb\xbf"))
            saved = validate_saved_csv(path, dataframe)
            self.assertEqual(saved["product_id"].tolist(), ["0007", "0008", "0009"])
            reread = load_dataset(path)
            self.assertEqual(reread["text"].iloc[1], "NA")
            self.assertEqual(reread["text"].iloc[2], "null")
            self.assertTrue(pd.isna(reread["sold"].iloc[1]))
            corrupted = saved.copy()
            corrupted.loc[0, "clean_text"] = "changed"
            save_csv(corrupted, path)
            with self.assertRaises(AssertionError):
                validate_saved_csv(path, dataframe)

    def test_json_loading_and_missing_file_errors_are_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "reviews.json"
            path.write_text(json.dumps({"reviews": [{"review": "mantap"}]}), encoding="utf-8-sig")
            self.assertEqual(len(load_json_dataframe(path, records_key="reviews")), 1)
            with self.assertRaisesRegex(ValueError, "JSON array"):
                load_json_dataframe(path)
            with self.assertRaisesRegex(FileNotFoundError, "data/raw/missing.json"):
                validate_dataset_files({"missing": Path(temporary) / "missing.json"})

    def test_audit_and_optional_label_columns(self):
        dataframe = pd.DataFrame({"text": ["bagus", "bagus", None], "label": ["OR", "OR", "CG"]})
        audit = audit_dataframe(dataframe, "text")
        self.assertEqual(audit["exact_duplicates"], 1)
        self.assertEqual(audit["missing_values"]["text"], 1)
        self.assertEqual(audit["unique_text_count"], 1)
        counts = label_distribution(dataframe, ["label", "absent"])
        self.assertEqual(set(counts["value"]), {"CG", "OR"})
        self.assertEqual(int(counts["count"].sum()), 3)

    def test_root_discovery_from_nested_notebook(self):
        self.assertEqual(find_project_root(PROJECT_DIR / "notebooks" / "01_data_audit"), PROJECT_DIR)
        self.assertEqual(find_project_root(PROJECT_DIR), PROJECT_DIR)
        self.assertEqual(find_project_root(), PROJECT_DIR)


if __name__ == "__main__":
    unittest.main()
