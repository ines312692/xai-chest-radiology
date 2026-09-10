"""FastAPI service exposing the XAI chest radiology pipeline.

Endpoints:
  GET  /health   readiness and runtime information
  GET  /models   the classifiers the service can serve and their availability
  GET  /vlm      state of the remote report server, the Kaggle session holding MedGemma
  POST /vlm      register the tunnel address printed by that notebook
  POST /analyze  multipart upload, analysed by one model, or by every model at once
  GET  /         the single page frontend
"""
import io
import logging

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from config import (CAM_METHODS, CAM_METRICS_SOURCE, DEFAULT_CAM_METHOD, DEFAULT_MODEL,
                    ENABLE_LOCAL_VLM, MAX_UPLOAD_MB, MODELS, VLM_MODEL_ID,
                    save_remote_vlm)
from inference import (DEVICE, analyze_image, available_cam_methods, available_models,
                       comparable_cam_methods, load_classifier, remote_status)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="XAI Chest Radiology API",
    description=("Classification, Grad-CAM visual explanation and text conditioned "
                 "VLM explanation for chest radiographs, served by several trained "
                 "models. Research prototype."),
    version="2.1.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


class VlmSettings(BaseModel):
    """Address of the report server, and optionally the shared secret guarding it."""
    url: str
    secret: str | None = None


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
    state = remote_status()
    logger.info("Report server: %s", state["detail"])


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE.type, "default_model": DEFAULT_MODEL,
            "local_vlm_enabled": ENABLE_LOCAL_VLM, "vlm_model": VLM_MODEL_ID,
            "remote_vlm": remote_status()}


@app.get("/models")
def models():
    """List the classifiers, so the interface can build its model selector."""
    return {"models": available_models(), "default": DEFAULT_MODEL}


@app.get("/cams")
def cams():
    """List the saliency methods and their benchmark scores, for the method selector."""
    return {"methods": available_cam_methods(), "default": DEFAULT_CAM_METHOD,
            "metrics_source": CAM_METRICS_SOURCE}


@app.get("/vlm")
def vlm():
    """Report whether the Kaggle report server is configured and answering."""
    return remote_status()


@app.post("/vlm")
def set_vlm(settings: VlmSettings):
    """Register the tunnel address printed by the notebook, then probe it.

    The address changes at every restart of the Kaggle session, so it is entered from the
    interface rather than edited in the source. An empty address clears the setting and
    sends the service back to the structured writer.
    """
    url = settings.url.strip().rstrip("/")
    if url and not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400,
                            detail="The address must start with http:// or https://")
    save_remote_vlm(url, settings.secret)
    return remote_status()


@app.post("/analyze")
async def analyze(file: UploadFile = File(...),
                  model_id: str = Form(DEFAULT_MODEL),
                  cam_method: str = Form(DEFAULT_CAM_METHOD),
                  compare: str = Form("none")):
    """Analyse one uploaded chest radiograph.

    Three comparison modes. With none, one model and one saliency method. With
    backbones, every available model analyses the same image with the chosen method,
    the selected one producing the textual explanation and the others only their
    heatmap, so the user sees how the visual explanation depends on what each model
    learned to detect. With methods, one model produces one map per fast saliency
    method, which shows how much of the explanation is the method rather than the model.
    """
    payload = await file.read()
    if len(payload) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413,
                            detail=f"File larger than {MAX_UPLOAD_MB} MB")
    if model_id not in MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model {model_id}")
    if cam_method not in CAM_METHODS:
        raise HTTPException(status_code=400,
                            detail=f"Unknown saliency method {cam_method}")

    # The checkbox of the previous interface sent a boolean, which is still accepted
    mode = {"true": "backbones", "false": "none", "": "none"}.get(
        compare.strip().lower(), compare.strip().lower())
    if mode not in ("none", "backbones", "methods"):
        raise HTTPException(status_code=400, detail=f"Unknown comparison mode {compare}")
    extra_methods = comparable_cam_methods(cam_method) if mode == "methods" else ()

    try:
        image = Image.open(io.BytesIO(payload))
        image.load()
    except UnidentifiedImageError:
        raise HTTPException(status_code=400,
                            detail="Unsupported image format, send png or jpeg")

    try:
        primary = analyze_image(image, model_id, with_text=True, cam_method=cam_method,
                                extra_cam_methods=extra_methods)
        others = []
        if mode == "backbones":
            for entry in available_models():
                if entry["available"] and entry["id"] != model_id:
                    others.append(analyze_image(image, entry["id"], with_text=False,
                                                cam_method=cam_method))
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