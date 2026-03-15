import asyncio
import hashlib
import logging
import io
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import os

import env_loader
from recognizers import NsfwRecognizer, create_deepfake_recognizer, create_flux_detector
import stats_service

logger = logging.getLogger(__name__)

# Ensure .env settings are available regardless of how uvicorn is launched.
env_loader.load_environment()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

models = {}


def _cors_origins() -> list[str]:
    configured = os.getenv("CORS_ALLOW_ORIGINS", "")
    if configured.strip():
        return [origin.strip() for origin in configured.split(",") if origin.strip()]
    return [
        "http://localhost:3000",
        "http://localhost:4173",
        "http://localhost:5173",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:4173",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080",
    ]

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
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_methods=["*"],
    allow_headers=["*"],
)

RECOGNIZER_NAMES = {"deepfake", "nsfw", "flux"}

class DetectionResult(BaseModel):
    is_target: bool
    label: str
    score: float
    all_predictions: list[dict]

class MultiDetectionResult(BaseModel):
    results: dict[str, DetectionResult]


@app.post("/detect/image/deepfake", response_model=DetectionResult)
async def detect_image_deepfake(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """Analyze an image for Deepfakes."""
    contents, image_hash = await _read_and_hash(file)
    result = await _process_image_from_bytes(Image.open(io.BytesIO(contents)).convert("RGB"), models['deepfake'])
    background_tasks.add_task(
        stats_service.log_detection_event,
        recognizer="deepfake",
        endpoint="/detect/image/deepfake",
        image_hash=image_hash,
        is_flagged=result.is_target,
        label=result.label,
        score=result.score,
        all_predictions=result.all_predictions,
    )
    return result

@app.post("/detect/image/nsfw", response_model=DetectionResult)
async def detect_image_nsfw(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """Analyze an image for NSFW content."""
    contents, image_hash = await _read_and_hash(file)
    result = await _process_image_from_bytes(Image.open(io.BytesIO(contents)).convert("RGB"), models['nsfw'])
    background_tasks.add_task(
        stats_service.log_detection_event,
        recognizer="nsfw",
        endpoint="/detect/image/nsfw",
        image_hash=image_hash,
        is_flagged=result.is_target,
        label=result.label,
        score=result.score,
        all_predictions=result.all_predictions,
    )
    return result


@app.post("/detect/image/flux", response_model=DetectionResult)
async def detect_image_flux(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """Analyze an image for Flux.1-generated content."""
    contents, image_hash = await _read_and_hash(file)
    result = await _process_image_from_bytes(Image.open(io.BytesIO(contents)).convert("RGB"), models['flux'])
    background_tasks.add_task(
        stats_service.log_detection_event,
        recognizer="flux",
        endpoint="/detect/image/flux",
        image_hash=image_hash,
        is_flagged=result.is_target,
        label=result.label,
        score=result.score,
        all_predictions=result.all_predictions,
    )
    return result

@app.post("/detect/image", response_model=MultiDetectionResult)
async def detect_image_multi(
    file: Annotated[UploadFile, File(...)],
    recognizers: Annotated[str, Form(..., description="Comma-separated list of recognizers to run: deepfake, nsfw, flux")],
    background_tasks: BackgroundTasks = BackgroundTasks(),
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
    image_hash = _sha256(contents)

    async def _run_one(name: str) -> tuple[str, DetectionResult]:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        return name, await _process_image_from_bytes(image, models[name])

    pairs = await asyncio.gather(*[_run_one(name) for name in requested])
    results = dict(pairs)
    background_tasks.add_task(
        stats_service.log_multi_detection_events,
        endpoint="/detect/image",
        image_hash=image_hash,
        results=results,
    )
    return MultiDetectionResult(results=results)


class RecognizerOverview(BaseModel):
    recognizer: str
    total_scans: int
    flagged_count: int
    clean_count: int
    flag_rate_pct: float | None
    avg_score: float | None
    avg_flagged_score: float | None
    avg_clean_score: float | None
    first_scan_at: str | None
    last_scan_at: str | None


class DailyStats(BaseModel):
    day: str
    recognizer: str
    total_scans: int
    flagged_count: int
    clean_count: int
    flag_rate_pct: float | None
    avg_score: float | None
    avg_flagged_score: float | None


class HourlyActivity(BaseModel):
    hour: str
    recognizer: str
    scans: int
    flagged: int


class ScoreBucket(BaseModel):
    recognizer: str
    bucket: int
    bucket_min: float
    bucket_max: float
    count: int


class HighRiskEvent(BaseModel):
    minute: str
    recognizer_hit_count: int
    recognizers_triggered: list[str]
    max_score: float
    min_score: float


class EventRecord(BaseModel):
    id: str
    created_at: str
    recognizer: str
    endpoint: str
    is_flagged: bool
    label: str
    score: float
    image_hash: str | None = None
    scan_count: int = 1
    times_flagged: int = 0


@app.get("/stats/summary", response_model=list[RecognizerOverview], tags=["stats"])
async def stats_summary():
    """All-time statistics per recognizer (total scans, flag rate, average score, …)."""
    return await asyncio.to_thread(stats_service.get_recognizer_overview)


@app.get("/stats/daily", response_model=list[DailyStats], tags=["stats"])
async def stats_daily(
    recognizer: str | None = Query(
        default=None,
        description="Filter to a single recognizer: deepfake | nsfw | flux",
    ),
    days: int = Query(
        default=30,
        ge=1,
        le=365,
        description="Number of past days to include (1–365).",
    ),
):
    """Daily scan and flag counts, optionally filtered by recognizer and time window."""
    return await asyncio.to_thread(
        stats_service.get_daily_summary,
        recognizer=recognizer,
        days=days,
    )


@app.get("/stats/recent", response_model=list[HourlyActivity], tags=["stats"])
async def stats_recent():
    """Hourly activity (scans + flags) for the last 7 days – useful for live dashboards."""
    return await asyncio.to_thread(stats_service.get_recent_hourly_activity)


@app.get("/stats/distribution", response_model=list[ScoreBucket], tags=["stats"])
async def stats_distribution(
    recognizer: str | None = Query(
        default=None,
        description="Filter to a single recognizer: deepfake | nsfw | flux",
    ),
):
    """Score distribution in 10 equal-width buckets (0–0.1, 0.1–0.2, …)."""
    return await asyncio.to_thread(
        stats_service.get_score_distribution,
        recognizer=recognizer,
    )


@app.get("/stats/high-risk", response_model=list[HighRiskEvent], tags=["stats"])
async def stats_high_risk(
    limit: int = Query(default=50, ge=1, le=200),
):
    """Time-windows where multiple recognizers flagged an image simultaneously."""
    return await asyncio.to_thread(stats_service.get_high_risk_events, limit)


@app.get("/stats/events", response_model=list[EventRecord], tags=["stats"])
async def stats_events(
    recognizer: str | None = Query(
        default=None,
        description="Filter to a single recognizer: deepfake | nsfw | flux",
    ),
    flagged_only: bool = Query(
        default=False,
        description="Return only flagged events.",
    ),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """Paginated raw detection event log (most-recent first)."""
    return await asyncio.to_thread(
        stats_service.get_events,
        recognizer=recognizer,
        flagged_only=flagged_only,
        limit=limit,
        offset=offset,
    )


async def _read_and_hash(file: UploadFile) -> tuple[bytes, str]:
    """Read an uploaded image file and return its raw bytes + SHA-256 hex hash."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    contents = await file.read()
    return contents, _sha256(contents)


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