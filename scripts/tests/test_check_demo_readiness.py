import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.check_demo_readiness import (
    PROJECT_ROOT,
    ReadinessError,
    requirement_pins,
    validate_configuration,
    validate_runtime,
)


class DemoReadinessTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_directory.name)
        self.checkpoint = self.root / "models" / "checkpoint.pt"
        self.checkpoint.parent.mkdir()
        self.checkpoint.write_bytes(b"prepared-checkpoint-fixture")
        self.expected_hash = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        self.backend_env = self.root / "backend.env"
        self.mobile_env = self.root / "mobile.env"
        self.write_valid_environment()

    def tearDown(self):
        self.temp_directory.cleanup()

    def write_valid_environment(self):
        self.backend_env.write_text(
            "\n".join(
                (
                    "MODEL_PATH=models/checkpoint.pt",
                    "ALLOWED_HOSTS=localhost,127.0.0.1,192.168.1.42",
                )
            ),
            encoding="utf-8",
        )
        self.mobile_env.write_text(
            "EXPO_PUBLIC_API_URL=http://192.168.1.42:8000\n",
            encoding="utf-8",
        )

    def validate(self):
        validate_configuration(
            project_root=self.root,
            backend_env=self.backend_env,
            mobile_env=self.mobile_env,
            checkpoint=self.checkpoint,
            expected_checkpoint_bytes=self.checkpoint.stat().st_size,
            expected_checkpoint_sha256=self.expected_hash,
        )

    def test_valid_configuration_passes_without_ml_imports(self):
        before = set(sys.modules)

        self.validate()

        imported = set(sys.modules) - before
        self.assertFalse(
            any(
                name == "ml"
                or name.startswith("ml.")
                or name == "torch"
                or name.startswith("torch.")
                for name in imported
            )
        )

    def test_mobile_host_must_be_allowed(self):
        self.backend_env.write_text(
            "MODEL_PATH=models/checkpoint.pt\nALLOWED_HOSTS=localhost,127.0.0.1\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ReadinessError, "also appear in ALLOWED_HOSTS"):
            self.validate()

    def test_wildcard_host_is_rejected(self):
        self.backend_env.write_text(
            "MODEL_PATH=models/checkpoint.pt\nALLOWED_HOSTS=*\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ReadinessError, "only explicit hosts"):
            self.validate()

    def test_unrelated_allowed_host_is_rejected(self):
        self.backend_env.write_text(
            "MODEL_PATH=models/checkpoint.pt\n"
            "ALLOWED_HOSTS=localhost,127.0.0.1,192.168.1.42,example.com\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ReadinessError, "outside the local demo boundary"):
            self.validate()

    def test_non_rfc1918_host_is_rejected(self):
        self.backend_env.write_text(
            "MODEL_PATH=models/checkpoint.pt\nALLOWED_HOSTS=203.0.113.5\n",
            encoding="utf-8",
        )
        self.mobile_env.write_text(
            "EXPO_PUBLIC_API_URL=http://203.0.113.5:8000\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ReadinessError, "RFC1918"):
            self.validate()

    def test_malformed_api_port_is_rejected_cleanly(self):
        self.mobile_env.write_text(
            "EXPO_PUBLIC_API_URL=http://192.168.1.42:not-a-port\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ReadinessError, "PRIVATE_IPV4"):
            self.validate()

    def test_wrong_checkpoint_hash_is_rejected(self):
        self.checkpoint.write_bytes(b"different-checkpoint-fixture")

        with self.assertRaisesRegex(ReadinessError, "wrong file size|wrong SHA-256"):
            self.validate()

    def test_recursive_requirement_pins_and_runtime_versions(self):
        api_requirements = self.root / "api.txt"
        api_requirements.write_text(
            "fastapi==0.139.2\npydantic==2.13.4\npydantic-settings==2.14.2\npillow==12.3.0\n",
            encoding="utf-8",
        )
        requirements = self.root / "requirements.txt"
        requirements.write_text(
            "-r api.txt\nuvicorn==0.51.0\ntorch==2.13.0+cpu\ntorchvision==0.28.0+cpu\n",
            encoding="utf-8",
        )
        installed = requirement_pins(requirements)

        validate_runtime(
            requirements,
            version_provider=installed.__getitem__,
            python_version=(3, 13),
        )

    def test_runtime_mismatch_is_rejected(self):
        requirements = self.root / "requirements.txt"
        requirements.write_text(
            "fastapi==0.139.2\n"
            "pydantic==2.13.4\n"
            "pydantic-settings==2.14.2\n"
            "pillow==12.3.0\n"
            "uvicorn==0.51.0\n"
            "torch==2.13.0+cpu\n"
            "torchvision==0.28.0+cpu\n",
            encoding="utf-8",
        )
        installed = requirement_pins(requirements)
        installed["torchvision"] = "0.26.0"

        with self.assertRaisesRegex(ReadinessError, "torchvision version"):
            validate_runtime(
                requirements,
                version_provider=installed.__getitem__,
                python_version=(3, 13),
            )

    def test_phone_lock_matches_backend_direct_pins(self):
        direct_pins = requirement_pins(PROJECT_ROOT / "backend" / "requirements.txt")
        phone_pins = requirement_pins(PROJECT_ROOT / "backend" / "requirements-phone.txt")

        for package, version in direct_pins.items():
            expected = f"{version}+cpu" if package in {"torch", "torchvision"} else version
            self.assertEqual(phone_pins.get(package), expected, package)


if __name__ == "__main__":
    unittest.main()
