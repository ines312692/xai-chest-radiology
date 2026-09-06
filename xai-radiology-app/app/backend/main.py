"""FastAPI service exposing the XAI chest radiology pipeline.

Endpoints:
  GET  /health   readiness and runtime information
  GET  /models   the classifiers the service can serve and their availability
  POST /analyze  multipart upload, analysed by one model, or by every model at once
  GET  /         the single page frontend
"""
import io
import logging

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from config import DEFAULT_MODEL, ENABLE_VLM, MAX_UPLOAD_MB, MODELS, VLM_MODEL_ID
from inference import DEVICE, analyze_image, available_models, load_classifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="XAI Chest Radiology API",
    description=("Classification, Grad-CAM visual explanation and text conditioned "
                 "VLM explanation for chest radiographs, served by several trained "
                 "models. Research prototype."),
    version="2.0.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


@app.on_event("startup")
def warm_up():
    """Load every available classifier at startup so the first request is fast."""
    for entry in available_models():
        if entry["available"]:
            try:
                load_classifier(entry["id"])
            except Exception as exc:
                logger.error("Model %s could not be loaded: %s", entry["id"], exc)
        else:
            logger.warning("Checkpoint missing for model %s, it will be offered as "
                           "unavailable", entry["id"])


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE.type, "default_model": DEFAULT_MODEL,
            "vlm_enabled": ENABLE_VLM, "vlm_model": VLM_MODEL_ID}


@app.get("/models")
def models():
    """List the classifiers, so the interface can build its model selector."""
    return {"models": available_models(), "default": DEFAULT_MODEL}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...),
                  model_id: str = Form(DEFAULT_MODEL),
                  compare: bool = Form(False)):
    """Analyse one uploaded chest radiograph.

    With compare set, every available model analyses the same image, the selected one
    producing the textual explanation and the others only their heatmap, so the user
    can see how the visual explanation depends on what each model learned to detect.
    """
    payload = await file.read()
    if len(payload) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413,
                            detail=f"File larger than {MAX_UPLOAD_MB} MB")
    if model_id not in MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model {model_id}")

    try:
        image = Image.open(io.BytesIO(payload))
        image.load()
    except UnidentifiedImageError:
        raise HTTPException(status_code=400,
                            detail="Unsupported image format, send png or jpeg")

    try:
        primary = analyze_image(image, model_id, with_text=True)
        others = []
        if compare:
            for entry in available_models():
                if entry["available"] and entry["id"] != model_id:
                    others.append(analyze_image(image, entry["id"], with_text=False))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Model checkpoint missing: {exc}")
    except Exception as exc:
        logger.exception("Analysis failed")
        raise HTTPException(status_code=500, detail=str(exc))

    primary["filename"] = file.filename
    primary["comparisons"] = others
    return primary


@app.get("/")
def index():
    """Serve the single page frontend."""
    return FileResponse("static/index.html")