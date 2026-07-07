import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ["HOSTED_DATABASE_URL"] = "https://example.hosted_database.co"
os.environ["HOSTED_DATABASE_SERVICE_ROLE_KEY"] = "test-service-role-key"

from backend.app.routers import scans


class ScanRouteTests(unittest.TestCase):
    def test_scan_history_allows_missing_image_url(self) -> None:
        query = Mock()
        query.select.return_value = query
        query.eq.return_value = query
        query.order.return_value = query
        query.execute.return_value = SimpleNamespace(
            data=[
                {
                    "id": "scan-id",
                    "image_url": None,
                    "prediction": "nevus",
                    "confidence": 0.90,
                    "model_version": "ham10000-test",
                    "created_at": "2026-07-07T00:00:00Z",
                }
            ]
        )

        fake_hosted_database = Mock()
        fake_hosted_database.table.return_value = query

        with (
            patch.object(scans, "hosted_database", fake_hosted_database),
            patch.object(scans, "create_signed_image_url") as sign_url,
        ):
            response = scans.list_scans("user-id")

        sign_url.assert_not_called()
        self.assertEqual(response["scans"][0]["id"], "scan-id")
        self.assertIsNone(response["scans"][0]["image_url"])
        self.assertIsNone(response["scans"][0]["signed_image_url"])


if __name__ == "__main__":
    unittest.main()
