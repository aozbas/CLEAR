import unittest

from pydantic import ValidationError

from backend.app.config import Settings


class SettingsTests(unittest.TestCase):
    def test_defaults_select_owner_approved_source_balanced_checkpoint(self) -> None:
        configured = Settings(_env_file=None)

        self.assertEqual(
            configured.model_path,
            "ml/models/pad_hiba_convnext_tiny_source_balanced_final_seed42.pt",
        )
        self.assertEqual(
            configured.model_version,
            "pad-hiba-convnext-tiny-source-balanced-final-2026-07-22",
        )

    def test_rejects_wildcard_cors_origin(self) -> None:
        with self.assertRaises(ValidationError):
            Settings(_env_file=None, cors_origins="*")

    def test_rejects_wildcard_allowed_host(self) -> None:
        with self.assertRaises(ValidationError):
            Settings(_env_file=None, allowed_hosts="*")

    def test_parses_explicit_origin_and_host_lists(self) -> None:
        configured = Settings(
            _env_file=None,
            cors_origins="https://demo.example, https://preview.example",
            allowed_hosts="api.example,preview-api.example",
        )

        self.assertEqual(
            configured.parsed_cors_origins,
            ["https://demo.example", "https://preview.example"],
        )
        self.assertEqual(
            configured.parsed_allowed_hosts,
            ["api.example", "preview-api.example"],
        )


if __name__ == "__main__":
    unittest.main()
