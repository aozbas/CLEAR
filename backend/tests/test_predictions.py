import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ["SUPABASE_URL"] = "https://example.supabase.co"
os.environ["SUPABASE_SERVICE_ROLE_KEY"] = "test-service-role-key"

from backend.app.routers import predictions


class Upload:
    def __init__(self, data: bytes = b"\xff\xd8\xffimage-bytes") -> None:
        self.data = data

    async def read(self) -> bytes:
        return self.data


class PredictionRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_low_confidence_result_is_not_saved(self) -> None:
        fake_supabase = Mock()
        with (
            patch.object(predictions, "supabase", fake_supabase),
            patch.object(
                predictions,
                "predict_lesion",
                return_value={"label": "nevus", "confidence": 0.10},
            ),
        ):
            response = await predictions.create_prediction(Upload(), "user-id")

        fake_supabase.table.assert_not_called()
        self.assertFalse(response["saved"])
        self.assertTrue(response["should_retry"])
        self.assertIsNone(response["scan_id"])

    async def test_saved_prediction_includes_model_version(self) -> None:
        insert_query = Mock()
        insert_query.insert.return_value = insert_query
        insert_query.execute.return_value = SimpleNamespace(data=[{"id": "scan-id"}])

        fake_supabase = Mock()
        fake_supabase.table.return_value = insert_query

        with (
            patch.object(predictions, "supabase", fake_supabase),
            patch.object(
                predictions,
                "predict_lesion",
                return_value={"label": "nevus", "confidence": 0.90},
            ),
        ):
            response = await predictions.create_prediction(Upload(), "user-id")

        insert_query.insert.assert_called_once()
        payload = insert_query.insert.call_args.args[0]
        self.assertIsNone(payload["image_url"])
        self.assertEqual(payload["model_version"], predictions.settings.model_version)
        self.assertTrue(response["saved"])
        self.assertFalse(response["should_retry"])
        self.assertIsNone(response["image_url"])
        self.assertIsNone(response["signed_image_url"])
        self.assertEqual(response["model_version"], predictions.settings.model_version)
        self.assertEqual(response["message"], "Saved to history. Photo was not saved.")

    async def test_demo_prediction_does_not_store_image_or_scan(self) -> None:
        fake_supabase = Mock()

        with (
            patch.object(predictions, "supabase", fake_supabase),
            patch.object(
                predictions,
                "predict_lesion",
                return_value={"label": "nevus", "confidence": 0.90},
            ),
        ):
            response = await predictions.create_demo_prediction(Upload())

        fake_supabase.table.assert_not_called()
        self.assertEqual(response["label"], "nevus")
        self.assertEqual(response["confidence"], 0.90)
        self.assertIsNone(response["image_url"])
        self.assertIsNone(response["signed_image_url"])
        self.assertIsNone(response["scan_id"])
        self.assertFalse(response["saved"])
        self.assertFalse(response["should_retry"])
        self.assertEqual(response["message"], "Demo result only. No photo or result was saved.")
        self.assertEqual(response["model_version"], predictions.settings.model_version)

    async def test_low_confidence_demo_prediction_is_not_saved(self) -> None:
        fake_supabase = Mock()

        with (
            patch.object(predictions, "supabase", fake_supabase),
            patch.object(
                predictions,
                "predict_lesion",
                return_value={"label": "melanoma", "confidence": 0.10},
            ),
        ):
            response = await predictions.create_demo_prediction(Upload())

        fake_supabase.table.assert_not_called()
        self.assertEqual(response["label"], "melanoma")
        self.assertEqual(response["confidence"], 0.10)
        self.assertFalse(response["saved"])
        self.assertTrue(response["should_retry"])
        self.assertEqual(response["message"], "Image unclear - try again.")


if __name__ == "__main__":
    unittest.main()
