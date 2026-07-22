import asyncio
import threading
import unittest

from backend.app.services.prediction_runtime import (
    PredictionBusyError,
    PredictionCapacity,
    PredictionTimeoutError,
)


class PredictionCapacityTests(unittest.IsolatedAsyncioTestCase):
    async def test_rejects_a_second_request_while_capacity_is_occupied(self) -> None:
        started = threading.Event()
        release = threading.Event()
        capacity = PredictionCapacity(1)

        def blocked_predictor(_: bytes) -> dict[str, object]:
            started.set()
            release.wait(timeout=1)
            return {"label": "nevus", "confidence": 0.9}

        first = asyncio.create_task(
            capacity.run(
                blocked_predictor,
                b"first",
                queue_timeout_seconds=0.1,
                prediction_timeout_seconds=1,
            )
        )
        self.assertTrue(await asyncio.to_thread(started.wait, 0.5))

        with self.assertRaises(PredictionBusyError):
            await capacity.run(
                blocked_predictor,
                b"second",
                queue_timeout_seconds=0.01,
                prediction_timeout_seconds=1,
            )

        release.set()
        await asyncio.sleep(0.05)
        await first

    async def test_timeout_keeps_capacity_reserved_until_worker_finishes(self) -> None:
        started = threading.Event()
        release = threading.Event()
        capacity = PredictionCapacity(1)

        def slow_predictor(_: bytes) -> dict[str, object]:
            started.set()
            release.wait(timeout=1)
            return {"label": "nevus", "confidence": 0.9}

        with self.assertRaises(PredictionTimeoutError):
            await capacity.run(
                slow_predictor,
                b"first",
                queue_timeout_seconds=0.1,
                prediction_timeout_seconds=0.01,
            )
        self.assertTrue(started.is_set())

        with self.assertRaises(PredictionBusyError):
            await capacity.run(
                slow_predictor,
                b"second",
                queue_timeout_seconds=0.01,
                prediction_timeout_seconds=1,
            )

        release.set()
        await asyncio.sleep(0.05)
        result = await capacity.run(
            lambda _: {"label": "nevus", "confidence": 0.9},
            b"third",
            queue_timeout_seconds=0.1,
            prediction_timeout_seconds=0.1,
        )
        self.assertEqual(result["label"], "nevus")

    async def test_cancellation_keeps_capacity_reserved_until_worker_finishes(self) -> None:
        started = threading.Event()
        release = threading.Event()
        capacity = PredictionCapacity(1)

        def blocked_predictor(_: bytes) -> dict[str, object]:
            started.set()
            release.wait(timeout=1)
            return {"label": "nevus", "confidence": 0.9}

        request = asyncio.create_task(
            capacity.run(
                blocked_predictor,
                b"first",
                queue_timeout_seconds=0.1,
                prediction_timeout_seconds=1,
            )
        )
        self.assertTrue(await asyncio.to_thread(started.wait, 0.5))
        request.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await request

        with self.assertRaises(PredictionBusyError):
            await capacity.run(
                blocked_predictor,
                b"second",
                queue_timeout_seconds=0.01,
                prediction_timeout_seconds=1,
            )

        release.set()
        await asyncio.sleep(0.05)


if __name__ == "__main__":
    unittest.main()
