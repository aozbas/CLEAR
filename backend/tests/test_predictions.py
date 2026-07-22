import base64
import sys
import unittest
from io import BytesIO
from unittest.mock import Mock, patch

from httpx import ASGITransport, AsyncClient
from PIL import Image, ImageDraw

from backend.app.config import settings
from backend.app.main import app
from backend.app.routers import predictions

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def png_fixture(
    *,
    size: tuple[int, int] = (128, 128),
    color: tuple[int, int, int] | None = None,
) -> bytes:
    image = Image.new("RGB", size, color=color or (90, 90, 90))
    if color is None:
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, size[0] // 2, size[1]), fill=(60, 80, 100))
        draw.ellipse((size[0] // 3, size[1] // 3, size[0] - 8, size[1] - 8), fill=(180, 130, 90))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def fake_predictor(_: bytes) -> dict[str, object]:
    return {"label": "nevus", "confidence": 0.90}


class PublicApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        app.dependency_overrides[predictions.get_predictor] = lambda: fake_predictor
        self.client = AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        )

    async def asyncTearDown(self) -> None:
        app.dependency_overrides.clear()
        await self.client.aclose()

    async def post_image(
        self,
        body: bytes | None = None,
        content_type: str = "image/png",
    ):
        return await self.client.post(
            "/predictions/demo",
            content=body if body is not None else png_fixture(),
            headers={"Content-Type": content_type},
        )

    async def test_api_import_and_fake_request_do_not_import_ml_or_torch(self) -> None:
        response = await self.post_image()

        self.assertEqual(response.status_code, 200)
        self.assertFalse(any(name == "ml" or name.startswith("ml.") for name in sys.modules))
        self.assertFalse(any(name == "torch" or name.startswith("torch.") for name in sys.modules))

    async def test_demo_returns_only_one_transient_experimental_result(self) -> None:
        response = await self.post_image()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")
        self.assertEqual(response.headers["x-frame-options"], "DENY")
        self.assertEqual(
            response.json(),
            {
                "result_type": "experimental_classification",
                "outcome": "classification_available",
                "label": "nevus",
                "model_score": 0.9,
                "should_retry": False,
                "message": "Experimental result only. CLEAR does not save the image or result.",
                "model_version": settings.model_version,
            },
        )

    async def test_low_score_hides_the_category_and_score(self) -> None:
        app.dependency_overrides[predictions.get_predictor] = lambda: (
            lambda _: {"label": "melanoma", "confidence": 0.10}
        )

        response = await self.post_image()

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["label"])
        self.assertIsNone(response.json()["model_score"])
        self.assertTrue(response.json()["should_retry"])
        self.assertEqual(response.json()["outcome"], "classifier_uncertain")
        self.assertNotIn("stored", response.json())

    async def test_poor_quality_images_abstain_without_prediction(self) -> None:
        fixtures = {
            "too_small": png_fixture(size=(32, 128)),
            "black": png_fixture(color=(0, 0, 0)),
            "white": png_fixture(color=(255, 255, 255)),
            "uniform": png_fixture(color=(120, 120, 120)),
        }
        for name, body in fixtures.items():
            with self.subTest(name=name):
                predictor = Mock(return_value={"label": "nevus", "confidence": 0.90})
                app.dependency_overrides[predictions.get_predictor] = lambda current=predictor: (
                    current
                )

                response = await self.post_image(body=body)

                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["outcome"], "poor_image_quality")
                self.assertIsNone(response.json()["label"])
                self.assertIsNone(response.json()["model_score"])
                self.assertTrue(response.json()["should_retry"])
                predictor.assert_not_called()

    async def test_legacy_persistence_routes_are_not_exposed(self) -> None:
        scans = await self.client.get("/scans")
        saved_prediction = await self.client.post(
            "/predictions", content=png_fixture(), headers={"Content-Type": "image/png"}
        )

        self.assertEqual(scans.status_code, 404)
        self.assertEqual(saved_prediction.status_code, 404)
        self.assertEqual(scans.headers["cache-control"], "no-store")

    async def test_rejects_unsupported_declared_type_before_prediction(self) -> None:
        predictor = Mock(return_value={"label": "nevus", "confidence": 0.90})
        app.dependency_overrides[predictions.get_predictor] = lambda: predictor

        response = await self.post_image(content_type="image/heic")

        self.assertEqual(response.status_code, 415)
        self.assertEqual(response.json()["detail"], "Submit a JPEG or PNG image.")
        predictor.assert_not_called()

    async def test_rejects_type_mismatch_and_malformed_image(self) -> None:
        mismatch = await self.post_image(body=PNG_1X1, content_type="image/jpeg")
        malformed = await self.post_image(body=b"not-an-image")

        self.assertEqual(mismatch.status_code, 415)
        self.assertIn("do not match", mismatch.json()["detail"])
        self.assertEqual(malformed.status_code, 415)
        self.assertEqual(malformed.json()["detail"], "The image is malformed or incomplete.")

    async def test_rejects_oversize_body_without_prediction(self) -> None:
        predictor = Mock(return_value={"label": "nevus", "confidence": 0.90})
        app.dependency_overrides[predictions.get_predictor] = lambda: predictor
        with patch.object(settings, "max_upload_bytes", 8):
            response = await self.post_image()

        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["detail"], "The image exceeds the upload size limit.")
        predictor.assert_not_called()

    async def test_rejects_image_over_pixel_limit_without_prediction(self) -> None:
        predictor = Mock(return_value={"label": "nevus", "confidence": 0.90})
        app.dependency_overrides[predictions.get_predictor] = lambda: predictor
        with patch.object(settings, "max_image_pixels", 0):
            response = await self.post_image()

        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["detail"], "The image exceeds the pixel limit.")
        predictor.assert_not_called()

    async def test_model_failure_returns_stable_error_without_internal_detail(self) -> None:
        def unavailable(_: bytes):
            raise FileNotFoundError("C:/private/checkpoints/model.pt")

        app.dependency_overrides[predictions.get_predictor] = lambda: unavailable
        response = await self.post_image()

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "The experimental classifier is unavailable.")
        self.assertNotIn("private", response.text)
        self.assertEqual(response.headers["cache-control"], "no-store")

    async def test_health_is_live_and_readiness_does_not_load_a_model(self) -> None:
        health = await self.client.get("/health")
        with patch.object(settings, "model_path", "missing-test-checkpoint.pt"):
            ready = await self.client.get("/ready")

        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json(), {"status": "ok"})
        self.assertEqual(ready.status_code, 503)
        self.assertEqual(
            ready.json(),
            {"status": "not_ready", "model_checkpoint_present": False},
        )

    async def test_openapi_documents_raw_images_and_stateless_response(self) -> None:
        response = await self.client.get("/openapi.json")

        operation = response.json()["paths"]["/predictions/demo"]["post"]
        content = operation["requestBody"]["content"]
        schema_name = operation["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
        schemas = response.json()["components"]["schemas"]

        self.assertEqual(set(content), {"image/jpeg", "image/png"})
        self.assertTrue(schema_name.endswith("ExperimentalClassificationResponse"))
        self.assertNotIn("scan_id", str(schemas))
        self.assertNotIn("image_url", str(schemas))
        self.assertNotIn("stored", str(schemas))

    async def test_host_and_cors_defaults_are_narrow(self) -> None:
        untrusted_client = AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://untrusted.example",
        )
        try:
            untrusted = await untrusted_client.get("/health")
        finally:
            await untrusted_client.aclose()

        allowed = await self.client.options(
            "/predictions/demo",
            headers={
                "Origin": "http://localhost:8081",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )
        blocked = await self.client.options(
            "/predictions/demo",
            headers={
                "Origin": "https://untrusted.example",
                "Access-Control-Request-Method": "POST",
            },
        )

        self.assertEqual(untrusted.status_code, 400)
        self.assertEqual(allowed.headers["access-control-allow-origin"], "http://localhost:8081")
        self.assertNotIn("access-control-allow-origin", blocked.headers)


if __name__ == "__main__":
    unittest.main()
