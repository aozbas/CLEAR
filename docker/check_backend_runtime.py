import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_PATH = ROOT / "backend" / "requirements.txt"
DOCKERFILE_PATH = ROOT / "docker" / "backend.Dockerfile"
PACKAGES = ("torch", "torchvision")


def extract_pins(text: str, pattern: re.Pattern[str], source: Path) -> dict[str, str]:
    matches = pattern.findall(text)
    pins: dict[str, str] = {}

    for package in PACKAGES:
        versions = [version for name, version in matches if name == package]
        if len(versions) != 1:
            raise SystemExit(
                f"{source.relative_to(ROOT)} must contain exactly one {package} pin; "
                f"found {len(versions)}."
            )
        pins[package] = versions[0]

    return pins


requirements_pins = extract_pins(
    REQUIREMENTS_PATH.read_text(encoding="utf-8"),
    re.compile(r"^(torch|torchvision)==([^\s#]+)$", re.MULTILINE),
    REQUIREMENTS_PATH,
)
dockerfile_text = DOCKERFILE_PATH.read_text(encoding="utf-8")
docker_pins = extract_pins(
    dockerfile_text,
    re.compile(r"\b(torch|torchvision)==([^\s\\]+)"),
    DOCKERFILE_PATH,
)

if "https://download.pytorch.org/whl/cpu" not in dockerfile_text:
    raise SystemExit("docker/backend.Dockerfile must install the CPU-only PyTorch runtime.")

if docker_pins != requirements_pins:
    raise SystemExit(
        "Backend Docker runtime pins do not match backend/requirements.txt: "
        f"docker={docker_pins}, requirements={requirements_pins}."
    )

print("Backend Docker runtime pins match backend/requirements.txt.")
