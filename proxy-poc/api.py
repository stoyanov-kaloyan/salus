import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from PIL import Image
import os

from recognizers import NsfwRecognizer, create_deepfake_recognizer

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    backend = os.getenv("DEEPFAKE_MODEL_BACKEND", "auto")
    checkpoint_path = os.getenv("DEEPFAKE_PT_CHECKPOINT_PATH")
    threshold_raw = os.getenv("DEEPFAKE_THRESHOLD")
    threshold = float(threshold_raw) if threshold_raw is not None else None
    models['deepfake'] = create_deepfake_recognizer(
        backend=backend,
        checkpoint_path=checkpoint_path,
        threshold=threshold,
    )
    models['nsfw'] = NsfwRecognizer()
    yield
    models.clear()

app = FastAPI(title="Content Moderation API", lifespan=lifespan)

class DetectionResult(BaseModel):
    is_target: bool
    label: str
    score: float
    all_predictions: list[dict]

@app.post("/detect/image/deepfake", response_model=DetectionResult)
async def detect_image_deepfake(file: UploadFile = File(...)):
    """Analyze an image for Deepfakes."""
    return await _process_image(file, models['deepfake'])

@app.post("/detect/image/nsfw", response_model=DetectionResult)
async def detect_image_nsfw(file: UploadFile = File(...)):
    """Analyze an image for NSFW content."""
    return await _process_image(file, models['nsfw'])

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