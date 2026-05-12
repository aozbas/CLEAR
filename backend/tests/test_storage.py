import os
import unittest
from unittest.mock import Mock

os.environ["SUPABASE_URL"] = "https://example.supabase.co"
os.environ["SUPABASE_SERVICE_ROLE_KEY"] = "test-service-role-key"

from backend.app.services import storage


class SignedImageUrlTests(unittest.TestCase):
    def test_detect_image_type_rejects_unknown_bytes(self) -> None:
        with self.assertRaises(storage.UnsupportedImageFormatError):
            storage._detect_image_type(b"not an image")

    def test_create_signed_image_url_accepts_dict_response(self) -> None:
        original_supabase = storage.supabase
        signed_url = "https://example.supabase.co/storage/v1/object/sign/scan-images/a.jpg"
        bucket = Mock()
        bucket.create_signed_url.return_value = {"signedURL": signed_url, "signedUrl": signed_url}
        fake_supabase = Mock()
        fake_supabase.storage.from_.return_value = bucket

        try:
            storage.supabase = fake_supabase
            self.assertEqual(storage.create_signed_image_url("user/a.jpg"), signed_url)
        finally:
            storage.supabase = original_supabase


if __name__ == "__main__":
    unittest.main()
