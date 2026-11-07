"""Independent regression cases for image-ID resolution.

Usage: python work/test_id_resolution_review.py [work/data_part.py]
These are synthetic filename layouts, not claims about the competition files.
"""
from __future__ import annotations

import importlib.util
import inspect
import sys
import unittest
from pathlib import Path


SOURCE = Path(sys.argv.pop(1)) if len(sys.argv) > 1 else Path(__file__).with_name("data_part.py")
SPEC = importlib.util.spec_from_file_location("id_resolution_under_review", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class IDResolverReview(unittest.TestCase):
    def resolve(self, names, query, role="train"):
        root = Path(__file__).resolve().parent / "synthetic_id_layout"
        paths = [root / name for name in names]
        parameters = inspect.signature(MODULE.build_image_index).parameters
        kwargs = {"root": root} if "root" in parameters else {}
        index = MODULE.build_image_index(paths, **kwargs)
        return Path(MODULE.resolve_image(index, query, role)), root

    def expect_path(self, names, query, expected, role="train"):
        actual, root = self.resolve(names, query, role)
        self.assertEqual(actual, root / expected)

    def expect_error(self, names, query, role="train"):
        with self.assertRaises(ValueError):
            self.resolve(names, query, role)

    def test_exact_case_and_extension(self):
        self.expect_path(["train/left_eye.JPG"], "LEFT_EYE.jpg", "train/left_eye.JPG")

    def test_numeric_zero_padding(self):
        self.expect_path(["train/001.jpg"], "1", "train/001.jpg")

    def test_csv_integer_decimal_string(self):
        self.expect_path(["train/001.jpg"], "1.0", "train/001.jpg")

    def test_identical_prefix_with_zero_padding(self):
        self.expect_path(["train/eye_001.jpg"], "eye_1", "train/eye_001.jpg")

    def test_prefix_numeric_when_unique(self):
        self.expect_path(["train/image_001.jpg"], "1", "train/image_001.jpg")

    def test_train_test_numeric_collision_is_split_scoped(self):
        self.expect_path(
            ["train/001.jpg", "test/001.jpg"], "1", "train/001.jpg", "train"
        )
        self.expect_path(
            ["train/001.jpg", "test/001.jpg"], "1", "test/001.jpg", "test"
        )

    def test_opposite_exact_match_cannot_beat_correct_split_alias(self):
        self.expect_path(
            ["train/image_001.jpg", "test/1.jpg"], "1", "train/image_001.jpg"
        )

    def test_only_opposite_split_must_be_rejected(self):
        self.expect_error(["test/1.jpg"], "1", "train")

    def test_prefixed_split_names(self):
        self.expect_path(
            ["train_images/001.jpg", "test_images/001.jpg"],
            "1", "train_images/001.jpg"
        )

    def test_filename_split_markers(self):
        self.expect_path(
            ["images/train_001.jpg", "images/test_001.jpg"],
            "1", "images/train_001.jpg"
        )

    def test_multiple_exact_paths_in_split_are_ambiguous(self):
        self.expect_error(["train/a/1.jpg", "train/b/1.jpg"], "1")

    def test_multiple_numeric_aliases_are_ambiguous(self):
        self.expect_error(["train/01.jpg", "train/001.jpg"], "1")

    def test_multiple_prefixes_are_ambiguous(self):
        self.expect_error(["train/image_001.jpg", "train/patient_001.jpg"], "1")

    def test_unknown_role_cannot_disambiguate(self):
        self.expect_error(["unknown_a/001.jpg", "unknown_b/001.jpg"], "1")

    def test_multi_number_filename_must_not_be_collapsed(self):
        self.expect_error(["train/patient2_eye1.jpg"], "1")

    def test_different_nonempty_prefixes_must_not_be_collapsed(self):
        self.expect_error(["train/right_001.jpg"], "left_1")

    def test_original_long_identifiers_preserved(self):
        self.expect_path(
            ["train/patient2_eye1.jpg", "test/patient2_eye1.jpg"],
            "patient2_eye1", "train/patient2_eye1.jpg"
        )

    def test_sample_parser_keeps_coordinate_axes_and_id(self):
        self.assertEqual(MODULE.parse_submission_id("001_Fovea_X"), ("001", "x"))
        self.assertEqual(MODULE.parse_submission_id("image_001_Fovea_Y"), ("image_001", "y"))
        self.assertEqual(MODULE.parse_submission_id("left_eye_001_X"), ("left_eye_001", "x"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
