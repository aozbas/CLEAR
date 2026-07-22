import asyncio
from collections.abc import Callable, Mapping
from typing import Any

PredictionCallable = Callable[[bytes], Mapping[str, Any]]


class PredictionBusyError(RuntimeError):
    """Raised when the bounded demo worker is already occupied."""


class PredictionTimeoutError(RuntimeError):
    """Raised when the predictor does not finish within the request deadline."""


class PredictionCapacity:
    def __init__(self, maximum: int) -> None:
        self._semaphore = asyncio.Semaphore(maximum)

    async def run(
        self,
        predictor: PredictionCallable,
        image_bytes: bytes,
        *,
        queue_timeout_seconds: float,
        prediction_timeout_seconds: float,
    ) -> Mapping[str, Any]:
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=queue_timeout_seconds)
        except TimeoutError as exc:
            raise PredictionBusyError("Prediction capacity is busy.") from exc

        release_on_return = True
        loop = asyncio.get_running_loop()
        try:
            prediction = loop.run_in_executor(None, predictor, image_bytes)
        except Exception:
            self._semaphore.release()
            raise
        try:
            done, _ = await asyncio.wait({prediction}, timeout=prediction_timeout_seconds)
            if not done:
                release_on_return = False
                prediction.add_done_callback(lambda _: self._semaphore.release())
                raise PredictionTimeoutError("Prediction exceeded its deadline.")
            return prediction.result()
        except asyncio.CancelledError:
            if not prediction.done():
                release_on_return = False
                prediction.add_done_callback(lambda _: self._semaphore.release())
            raise
        finally:
            if release_on_return:
                self._semaphore.release()
