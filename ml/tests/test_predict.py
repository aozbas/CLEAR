import unittest

from ml.inference.predict import (
    CURRENT_CV_SUMMARY_SHA256,
    CURRENT_HIBA_VIEW_WEIGHTING,
    CURRENT_MANIFEST_FINGERPRINT,
    CURRENT_MODEL_VERSION,
    CURRENT_SOURCE_CLASS_WEIGHTING,
    CURRENT_TRAINING_PROTOCOL,
    DEFAULT_LABELS,
    HAM10000_LABELS,
    PAD_HIBA_LABELS,
    PREPROCESSING,
    InvalidImageError,
    get_checkpoint_architecture,
    get_checkpoint_labels,
    get_checkpoint_preprocessing,
    load_image,
)


class CheckpointLabelTests(unittest.TestCase):
    def test_default_labels_are_ham10000_phase_2_labels(self) -> None:
        self.assertEqual(DEFAULT_LABELS, HAM10000_LABELS)
        self.assertEqual(len(DEFAULT_LABELS), 7)
        self.assertIn("melanoma", DEFAULT_LABELS)
        self.assertIn("vascular_lesion", DEFAULT_LABELS)

    def test_checkpoint_labels_default_for_plain_state_dict(self) -> None:
        self.assertEqual(get_checkpoint_labels({"fc.weight": object()}), DEFAULT_LABELS)

    def test_checkpoint_labels_use_saved_labels(self) -> None:
        labels = ["melanoma", "nevus", "basal_cell_carcinoma"]
        self.assertEqual(get_checkpoint_labels({"labels": labels}), labels)

    def test_checkpoint_labels_reject_bad_values(self) -> None:
        with self.assertRaises(ValueError):
            get_checkpoint_labels({"labels": []})

    def test_convnext_checkpoint_metadata_matches_demo_contract(self) -> None:
        checkpoint = self.approved_convnext_metadata()

        architecture = get_checkpoint_architecture(checkpoint)

        self.assertEqual(architecture, "convnext_tiny")
        self.assertEqual(
            get_checkpoint_preprocessing(checkpoint, architecture=architecture),
            PREPROCESSING,
        )

    def test_convnext_checkpoint_rejects_wrong_label_order(self) -> None:
        checkpoint = {
            "architecture": "convnext_tiny",
            "labels": list(reversed(PAD_HIBA_LABELS)),
            "preprocessing": PREPROCESSING,
        }

        with self.assertRaises(ValueError):
            get_checkpoint_preprocessing(checkpoint, architecture="convnext_tiny")

    def test_convnext_checkpoint_rejects_non_final_fit_metadata(self) -> None:
        checkpoint = self.approved_convnext_metadata()
        checkpoint["epoch"] = 12

        with self.assertRaises(ValueError):
            get_checkpoint_preprocessing(checkpoint, architecture="convnext_tiny")

    @staticmethod
    def approved_convnext_metadata() -> dict[str, object]:
        return {
            "architecture": "convnext_tiny",
            "labels": PAD_HIBA_LABELS,
            "preprocessing": PREPROCESSING,
            "dataset": "pad_ufes_hiba",
            "dataset_role": "multisource_development_final_fit",
            "training_protocol": CURRENT_TRAINING_PROTOCOL,
            "model_version": CURRENT_MODEL_VERSION,
            "sources": ["pad_ufes", "hiba"],
            "source_class_weighting": CURRENT_SOURCE_CLASS_WEIGHTING,
            "hiba_view_weighting": CURRENT_HIBA_VIEW_WEIGHTING,
            "manifest_fingerprint": CURRENT_MANIFEST_FINGERPRINT,
            "cv_summary_sha256": CURRENT_CV_SUMMARY_SHA256,
            "cv_decision_all_pass": False,
            "selection_status": "owner_selected_despite_failed_preregistered_gates",
            "pretrained_weights": "imagenet",
            "pretrained_weights_id": "IMAGENET1K_V1",
            "epoch": 11,
            "seed": 42,
            "source_total_raw_image_counts": {"pad_ufes": 2_298, "hiba": 309},
            "source_total_effective_unit_counts": {"pad_ufes": 2_298.0, "hiba": 308.0},
            "hyperparameters": {
                "epochs": 11,
                "epoch_rule": "median_of_locked_cv_selected_epochs",
                "locked_cv_selected_epochs": [15, 8, 10, 11, 13],
                "batch_size": 32,
                "learning_rate": 1e-4,
                "weight_decay": 1e-4,
                "optimizer": "AdamW",
                "schedule": "none",
                "augmentation_profile": "baseline",
                "label_smoothing": 0.0,
                "sampling": "random_shuffle_without_replacement",
                "source_class_weighting": CURRENT_SOURCE_CLASS_WEIGHTING,
                "hiba_view_weighting": CURRENT_HIBA_VIEW_WEIGHTING,
            },
        }

    def test_checkpoint_rejects_unknown_architecture_and_preprocessing(self) -> None:
        with self.assertRaises(ValueError):
            get_checkpoint_architecture({"architecture": "convnext_base"})
        with self.assertRaises(ValueError):
            get_checkpoint_preprocessing(
                {"preprocessing": "unknown"},
                architecture="resnet18",
            )

    def test_load_image_rejects_unreadable_bytes(self) -> None:
        with self.assertRaises(InvalidImageError):
            load_image(b"not an image")


if __name__ == "__main__":
    unittest.main()
