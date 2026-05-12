import os
import unittest
from unittest.mock import Mock

os.environ["HOSTED_DATABASE_URL"] = "https://example.hosted_database.co"
os.environ["HOSTED_DATABASE_SERVICE_ROLE_KEY"] = "test-service-role-key"

from backend.app.services import storage


class SignedImageUrlTests(unittest.TestCase):
    def test_detect_image_type_rejects_unknown_bytes(self) -> None:
        with self.assertRaises(storage.UnsupportedImageFormatError):
            storage._detect_image_type(b"not an image")

    def test_create_signed_image_url_accepts_dict_response(self) -> None:
        original_hosted_database = storage.hosted_database
        signed_url = "https://example.hosted_database.co/storage/v1/object/sign/scan-images/a.jpg"
        bucket = Mock()
        bucket.create_signed_url.return_value = {"signedURL": signed_url, "signedUrl": signed_url}
        fake_hosted_database = Mock()
        fake_hosted_database.storage.from_.return_value = bucket

        try:
            storage.hosted_database = fake_hosted_database
            self.assertEqual(storage.create_signed_image_url("user/a.jpg"), signed_url)
        finally:
            storage.hosted_database = original_hosted_database


if __name__ == "__main__":
    unittest.main()
