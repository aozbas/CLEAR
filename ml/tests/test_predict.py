import unittest
from unittest.mock import Mock, patch

import torch

import ml.inference.predict as prediction_module
from ml.inference.predict import (
    CURRENT_CV_SUMMARY_SHA256,
    CURRENT_HIBA_VIEW_WEIGHTING,
    CURRENT_MANIFEST_FINGERPRINT,
    CURRENT_MANIFEST_IDENTITY_FINGERPRINT,
    CURRENT_MODEL_VERSION,
    CURRENT_SOURCE_CLASS_WEIGHTING,
    CURRENT_TRAINING_PROTOCOL,
    DEFAULT_LABELS,
    PAD_HIBA_LABELS,
    PREPROCESSING,
    SUPPORTED_INPUT_GATE_COHORT_FINGERPRINT,
    SUPPORTED_INPUT_GATE_METHOD,
    SUPPORTED_INPUT_GATE_PROTOCOL,
    SUPPORTED_INPUT_GATE_REPORT_SHA256,
    SUPPORTED_INPUT_GATE_THRESHOLD,
    SUPPORTED_INPUT_GATE_VERSION,
    InvalidImageError,
    get_checkpoint_architecture,
    get_checkpoint_labels,
    get_checkpoint_preprocessing,
    load_image,
    supported_input_score,
)


class CheckpointLabelTests(unittest.TestCase):
    def test_default_labels_match_the_approved_demo_order(self) -> None:
        self.assertEqual(DEFAULT_LABELS, PAD_HIBA_LABELS)
        self.assertEqual(len(DEFAULT_LABELS), 6)
        self.assertIn("melanoma", DEFAULT_LABELS)
        self.assertIn("squamous_cell_carcinoma", DEFAULT_LABELS)

    def test_checkpoint_labels_reject_plain_state_dict(self) -> None:
        with self.assertRaises(ValueError):
            get_checkpoint_labels({"features.0.weight": object()})

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
            "manifest_identity_fingerprint": CURRENT_MANIFEST_IDENTITY_FINGERPRINT,
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
                architecture="convnext_tiny",
            )

    def test_load_image_rejects_unreadable_bytes(self) -> None:
        with self.assertRaises(InvalidImageError):
            load_image(b"not an image")


class SupportedInputGateTests(unittest.TestCase):
    def test_gate_contract_matches_the_frozen_aggregate_report(self) -> None:
        self.assertEqual(
            SUPPORTED_INPUT_GATE_PROTOCOL,
            "pad_hiba_open_images_supported_input_gate_v1",
        )
        self.assertEqual(SUPPORTED_INPUT_GATE_METHOD, "logsumexp")
        self.assertEqual(SUPPORTED_INPUT_GATE_THRESHOLD, 4.4970903396606445)
        self.assertEqual(
            SUPPORTED_INPUT_GATE_COHORT_FINGERPRINT,
            "fe5cfd2dc03a79a40eed07fc2b7cc79e28e176a1a5e82d72dc0e572ab56ee1b2",
        )
        self.assertEqual(
            SUPPORTED_INPUT_GATE_REPORT_SHA256,
            "79f53e4ff3c76f56d3375aee38c728a739220e8194e5c2b0bbc3f278e621e6ee",
        )
        self.assertEqual(
            SUPPORTED_INPUT_GATE_VERSION,
            "pad-hiba-open-images-supported-input-gate-v1-79f53e4ff3c76f56",
        )

    def test_supported_input_score_is_logsumexp(self) -> None:
        logits = torch.tensor([[1.0, 2.0], [3.0, -1.0]])

        score = supported_input_score(logits)

        torch.testing.assert_close(score, torch.logsumexp(logits, dim=1))

    def test_predict_suppresses_label_and_score_for_an_unsupported_input(self) -> None:
        result = self._predict_with_logits(torch.zeros((1, 6)))

        self.assertEqual(
            result,
            {
                "label": None,
                "confidence": None,
                "input_supported": False,
                "input_gate_version": SUPPORTED_INPUT_GATE_VERSION,
            },
        )

    def test_predict_returns_classification_fields_for_a_supported_input(self) -> None:
        result = self._predict_with_logits(torch.tensor([[5.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))

        self.assertEqual(result["label"], PAD_HIBA_LABELS[0])
        self.assertGreater(result["confidence"], 0.0)
        self.assertTrue(result["input_supported"])
        self.assertEqual(result["input_gate_version"], SUPPORTED_INPUT_GATE_VERSION)

    @staticmethod
    def _predict_with_logits(logits: torch.Tensor) -> dict[str, object]:
        model = Mock(return_value=logits)
        transform = Mock(return_value=torch.zeros((3, 224, 224)))
        with (
            patch.object(prediction_module, "load_model", return_value=model),
            patch.object(prediction_module, "load_image", return_value=object()),
            patch.object(prediction_module, "get_transforms", return_value=transform),
            patch.object(prediction_module, "_MODEL_LABELS", PAD_HIBA_LABELS),
            patch.object(prediction_module, "_MODEL_PREPROCESSING", PREPROCESSING),
        ):
            return prediction_module.predict(b"synthetic", device="cpu")


if __name__ == "__main__":
    unittest.main()
