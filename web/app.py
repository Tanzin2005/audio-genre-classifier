from contextlib import asynccontextmanager
import io
import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
import torch

from genre_cnn.predict import Predictor

ROOT = Path(__file__).resolve().parents[1]
MAX_UPLOAD = 40 * 1024 * 1024
logger = logging.getLogger(__name__)


class UploadLimit:
    """Bound the raw request body before multipart parsing or temporary storage."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] != "/api/predict":
            return await self.app(scope, receive, send)
        size = 0

        async def limited_receive():
            nonlocal size
            message = await receive()
            size += len(message.get("body", b""))
            if size > MAX_UPLOAD + 1024 * 1024:
                raise HTTPException(413, "Choose an audio file smaller than 40 MB.")
            return message

        await self.app(scope, limited_receive, send)


def create_app(checkpoint=None):
    @asynccontextmanager
    async def lifespan(app):
        torch.set_num_threads(2)
        app.state.predictor = None
        app.state.busy = False
        model_path = checkpoint or os.environ.get(
            "MODEL_PATH", str(ROOT / "models/cnn.pt")
        )
        try:
            app.state.predictor = Predictor(model_path)
        except (FileNotFoundError, ValueError, RuntimeError, KeyError, TypeError):
            logger.exception("Could not load the trained CNN")
        yield

    app = FastAPI(title="Audio Genre Classifier", lifespan=lifespan)
    app.add_middleware(UploadLimit)
    app.mount("/static", StaticFiles(directory=ROOT / "web/static"), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(ROOT / "web/static/index.html")

    @app.get("/api/status")
    def status(request: Request):
        predictor = request.app.state.predictor
        return {
            "ready": predictor is not None,
            "genres": predictor.classes if predictor else [],
            "message": "Ready to classify"
            if predictor
            else "The trained model is not available yet.",
        }

    @app.get("/health")
    def health(request: Request):
        if request.app.state.predictor is None:
            raise HTTPException(503, "Trained model unavailable")
        return {"status": "ok"}

    @app.post("/api/predict")
    async def predict(request: Request):
        if request.app.state.predictor is None:
            raise HTTPException(503, "The trained model is not available yet.")
        # This check/set has no await in between: one in-flight upload/inference per worker.
        if request.app.state.busy:
            raise HTTPException(
                429, "Another clip is being analyzed. Try again shortly."
            )
        request.app.state.busy = True
        try:
            async with request.form(max_files=1, max_fields=0) as form:
                upload = form.get("file")
                if upload is None or not hasattr(upload, "read"):
                    raise HTTPException(422, "Choose an audio file.")
                if Path(upload.filename or "").suffix.lower() not in {
                    ".wav",
                    ".flac",
                    ".ogg",
                    ".mp3",
                }:
                    raise HTTPException(415, "Use WAV, FLAC, OGG, or MP3 audio.")
                content = await upload.read(MAX_UPLOAD + 1)
                if len(content) > MAX_UPLOAD:
                    raise HTTPException(413, "Choose an audio file smaller than 40 MB.")
                try:
                    return await run_in_threadpool(
                        request.app.state.predictor.predict, io.BytesIO(content)
                    )
                except (ValueError, RuntimeError, OSError) as error:
                    raise HTTPException(
                        422,
                        "Could not analyze this audio. Use a valid, non-silent clip at least 3 seconds long.",
                    ) from error
        finally:
            request.app.state.busy = False

    return app


app = create_app()
