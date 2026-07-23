"""Validate local phone-demo configuration without importing ML code."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import ipaddress
import re
import sys
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CHECKPOINT = Path("ml/models/pad_hiba_convnext_tiny_source_balanced_final_seed42.pt")
EXPECTED_CHECKPOINT_BYTES = 111_376_483
EXPECTED_CHECKPOINT_SHA256 = "12c7261b06e3da9d1639e5e2c11220837de5a69f972acf25a55c4a0ae31d99b8"
RUNTIME_PACKAGES = (
    "fastapi",
    "pydantic",
    "pydantic-settings",
    "pillow",
    "torch",
    "torchvision",
    "uvicorn",
)
REQUIREMENT_PATTERN = re.compile(r"^(?P<name>[A-Za-z0-9_.-]+)(?:\[[^\]]+\])?==(?P<version>[^\s;]+)")
RFC1918_NETWORKS = tuple(
    ipaddress.ip_network(network) for network in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
)


class ReadinessError(ValueError):
    """Raised when local-demo configuration is unsafe or incomplete."""


def read_dotenv(path: Path) -> dict[str, str]:
    """Read the repository's simple KEY=VALUE environment-file format."""
    if not path.is_file():
        raise ReadinessError(f"Required environment file is missing: {path.name}")

    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ReadinessError(f"{path.name} has an invalid entry on line {line_number}")
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in values:
            raise ReadinessError(f"{path.name} has a missing or duplicate key")
        values[key] = value.strip()
    return values


def _rfc1918_address(value: str) -> ipaddress.IPv4Address:
    try:
        address = ipaddress.ip_address(value)
    except ValueError as exc:
        raise ReadinessError("The mobile API host must be an IPv4 address") from exc
    if not isinstance(address, ipaddress.IPv4Address) or not any(
        address in network for network in RFC1918_NETWORKS
    ):
        raise ReadinessError("The mobile API host must be an RFC1918 private IPv4 address")
    return address


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_configuration(
    *,
    project_root: Path,
    backend_env: Path,
    mobile_env: Path,
    checkpoint: Path,
    expected_checkpoint_bytes: int = EXPECTED_CHECKPOINT_BYTES,
    expected_checkpoint_sha256: str = EXPECTED_CHECKPOINT_SHA256,
) -> None:
    backend_values = read_dotenv(backend_env)
    mobile_values = read_dotenv(mobile_env)

    api_url = mobile_values.get("EXPO_PUBLIC_API_URL", "")
    try:
        parsed_url = urlsplit(api_url)
        api_port = parsed_url.port
    except ValueError as exc:
        raise ReadinessError(
            "EXPO_PUBLIC_API_URL must be http://PRIVATE_IPV4:8000 for the local demo"
        ) from exc
    if (
        parsed_url.scheme != "http"
        or api_port != 8000
        or parsed_url.username
        or parsed_url.password
        or parsed_url.query
        or parsed_url.fragment
        or parsed_url.path not in ("", "/")
    ):
        raise ReadinessError(
            "EXPO_PUBLIC_API_URL must be http://PRIVATE_IPV4:8000 for the local demo"
        )
    api_host = _rfc1918_address(parsed_url.hostname or "")

    allowed_hosts = {
        entry.strip()
        for entry in backend_values.get("ALLOWED_HOSTS", "").split(",")
        if entry.strip()
    }
    if not allowed_hosts or any("*" in entry for entry in allowed_hosts):
        raise ReadinessError("ALLOWED_HOSTS must contain only explicit hosts")
    if str(api_host) not in allowed_hosts:
        raise ReadinessError("The mobile API host must also appear in ALLOWED_HOSTS")
    expected_local_hosts = {"localhost", "127.0.0.1", "testserver", str(api_host)}
    if not allowed_hosts.issubset(expected_local_hosts):
        raise ReadinessError("ALLOWED_HOSTS contains a host outside the local demo boundary")

    configured_model = Path(backend_values.get("MODEL_PATH", EXPECTED_CHECKPOINT.as_posix()))
    if not configured_model.is_absolute():
        configured_model = project_root / configured_model
    if configured_model.resolve() != checkpoint.resolve():
        raise ReadinessError("MODEL_PATH must identify the selected demo checkpoint")

    if not checkpoint.is_file():
        raise ReadinessError("The selected demo checkpoint is missing")
    if checkpoint.stat().st_size != expected_checkpoint_bytes:
        raise ReadinessError("The selected demo checkpoint has the wrong file size")
    if sha256_file(checkpoint) != expected_checkpoint_sha256:
        raise ReadinessError("The selected demo checkpoint has the wrong SHA-256")


def requirement_pins(path: Path, seen: set[Path] | None = None) -> dict[str, str]:
    path = path.resolve()
    visited = seen if seen is not None else set()
    if path in visited:
        return {}
    visited.add(path)

    pins: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("-r ", "--requirement ")):
            nested_path = line.split(maxsplit=1)[1]
            pins.update(requirement_pins(path.parent / nested_path, visited))
            continue
        match = REQUIREMENT_PATTERN.match(line)
        if not match:
            continue
        name = match.group("name").lower().replace("_", "-")
        if name in pins:
            raise ReadinessError(f"Duplicate runtime pin for {name}")
        pins[name] = match.group("version")
    return pins


def validate_runtime(
    requirements_path: Path,
    *,
    version_provider: Callable[[str], str] = importlib.metadata.version,
    python_version: tuple[int, int] | None = None,
) -> None:
    active_python = python_version or (sys.version_info.major, sys.version_info.minor)
    if active_python != (3, 13):
        raise ReadinessError("The phone backend requires Python 3.13")

    pins = requirement_pins(requirements_path)
    for required_package in RUNTIME_PACKAGES:
        if required_package not in pins:
            raise ReadinessError(f"The backend is missing an exact {required_package} runtime pin")

    for package, expected in pins.items():
        try:
            installed = version_provider(package)
        except (importlib.metadata.PackageNotFoundError, KeyError) as exc:
            raise ReadinessError(f"The phone runtime is missing {package}") from exc
        if installed != expected:
            raise ReadinessError(f"The installed {package} version does not match its pin")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check local phone-demo configuration without loading the model."
    )
    parser.add_argument(
        "--backend-env",
        type=Path,
        default=PROJECT_ROOT / "backend" / ".env",
    )
    parser.add_argument(
        "--mobile-env",
        type=Path,
        default=PROJECT_ROOT / "mobile" / ".env",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / EXPECTED_CHECKPOINT,
    )
    parser.add_argument(
        "--requirements",
        type=Path,
        default=PROJECT_ROOT / "backend" / "requirements-phone.txt",
    )
    parser.add_argument(
        "--skip-runtime",
        action="store_true",
        help="Check files and checkpoint identity before creating the pinned runtime.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_configuration(
            project_root=PROJECT_ROOT,
            backend_env=args.backend_env,
            mobile_env=args.mobile_env,
            checkpoint=args.checkpoint,
        )
        if not args.skip_runtime:
            validate_runtime(args.requirements)
    except (OSError, ReadinessError) as exc:
        print(f"Demo readiness check failed: {exc}", file=sys.stderr)
        return 1

    print("Demo readiness checks passed without loading the model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
