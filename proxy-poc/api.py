import asyncio
import io
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from pydantic import BaseModel
from PIL import Image
import os

from recognizers import NsfwRecognizer, create_deepfake_recognizer, create_flux_detector

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    backend = os.getenv("DEEPFAKE_MODEL_BACKEND", "auto")
    checkpoint_path = os.getenv("DEEPFAKE_PT_CHECKPOINT_PATH")
    threshold_raw = os.getenv("DEEPFAKE_THRESHOLD")
    threshold = float(threshold_raw) if threshold_raw is not None else None
    flux_threshold = float(os.getenv("FLUX_THRESHOLD", "0.5"))
    flux_device = int(os.getenv("FLUX_DEVICE", "0"))
    flux_model_name = os.getenv("FLUX_MODEL_NAME", "prithivMLmods/OpenSDI-Flux.1-SigLIP2")
    models['deepfake'] = create_deepfake_recognizer(
        backend=backend,
        checkpoint_path=checkpoint_path,
        threshold=threshold,
    )
    models['flux'] = create_flux_detector(
        model_name=flux_model_name,
        device=flux_device,
        threshold=flux_threshold,
    )
    models['nsfw'] = NsfwRecognizer()
    yield
    models.clear()

app = FastAPI(title="Content Moderation API", lifespan=lifespan)

RECOGNIZER_NAMES = {"deepfake", "nsfw", "flux"}

class DetectionResult(BaseModel):
    is_target: bool
    label: str
    score: float
    all_predictions: list[dict]

class MultiDetectionResult(BaseModel):
    results: dict[str, DetectionResult]

@app.post("/detect/image/deepfake", response_model=DetectionResult)
async def detect_image_deepfake(file: UploadFile = File(...)):
    """Analyze an image for Deepfakes."""
    return await _process_image(file, models['deepfake'])

@app.post("/detect/image/nsfw", response_model=DetectionResult)
async def detect_image_nsfw(file: UploadFile = File(...)):
    """Analyze an image for NSFW content."""
    return await _process_image(file, models['nsfw'])


@app.post("/detect/image/flux", response_model=DetectionResult)
async def detect_image_flux(file: UploadFile = File(...)):
    """Analyze an image for Flux.1-generated content."""
    return await _process_image(file, models['flux'])

@app.post("/detect/image", response_model=MultiDetectionResult)
async def detect_image_multi(
    file: Annotated[UploadFile, File(...)],
    recognizers: Annotated[str, Form(..., description="Comma-separated list of recognizers to run: deepfake, nsfw, flux")],
):
    """Run one or more recognizers on an image in parallel and return combined results."""
    requested = {name.strip().lower() for name in recognizers.split(",") if name.strip()}
    unknown = requested - RECOGNIZER_NAMES
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown recognizer(s): {', '.join(sorted(unknown))}. Valid options: {', '.join(sorted(RECOGNIZER_NAMES))}")
    if not requested:
        raise HTTPException(status_code=400, detail="At least one recognizer must be specified.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    contents = await file.read()

    async def _run_one(name: str) -> tuple[str, DetectionResult]:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        return name, await _process_image_from_bytes(image, models[name])

    pairs = await asyncio.gather(*[_run_one(name) for name in requested])
    return MultiDetectionResult(results=dict(pairs))


async def _process_image_from_bytes(image: Image.Image, recognizer) -> DetectionResult:
    """Run a recognizer against an already-opened PIL image."""
    try:
        decision = recognizer.evaluate(image)
        return DetectionResult(
            is_target=decision.should_change,
            label=decision.label,
            score=decision.score,
            all_predictions=decision.predictions,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


async def _process_image(file: UploadFile, recognizer) -> DetectionResult:
    """Helper function to read the image and run the model."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Run inference
        decision = recognizer.evaluate(image)
        
        return DetectionResult(
            is_target=decision.should_change,
            label=decision.label,
            score=decision.score,
            all_predictions=decision.predictions
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")