"""Inference core of the service.

The module exposes one entry point, analyze_image, which runs the three stages of the
pipeline for a chosen model: classification, visual explanation with Grad-CAM, and
textual explanation.

The textual stage is tried in three steps, in this order. The remote report server is the
normal path, a Kaggle session holding MedGemma on a GPU behind a public tunnel. A local
MedGemma is used when the machine running this service has a GPU and the local mode was
enabled. The structured writer is the deterministic fallback, so a demonstration never
fails because a tunnel closed.

What travels to the report server is the clean radiograph and the facts extracted here,
the predicted class, its confidence and the anatomical zone of strongest activation. The
saliency overlay never travels, following the conditioning results of the project.
"""
import base64
import io
import logging
import os
from functools import lru_cache

import numpy as np
import requests
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torchvision import models as tv_models

from config import (ENABLE_LOCAL_VLM, IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE, MODELS,
                    REMOTE_VLM_MAX_SIDE, REMOTE_VLM_TIMEOUT, VLM_MODEL_ID,
                    remote_vlm_secret, remote_vlm_url)

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def available_models():
    """Describe every declared model and whether its checkpoint is present on disk."""
    return [{
        "id": model_id,
        "name": spec["name"],
        "classes": spec["classes"],
        "description": spec["description"],
        "available": os.path.exists(spec["checkpoint"]),
    } for model_id, spec in MODELS.items()]


def _build_architecture(arch, num_classes):
    """Instantiate one backbone with a fresh head of the right size.

    The head lives under a different attribute in each family, classifier for DenseNet
    and fc for ResNet, which is the only architectural difference the service handles.
    """
    if arch == "densenet121":
        model = tv_models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif arch == "resnet50":
        model = tv_models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported architecture {arch}")
    return model


def _target_layer(model, arch):
    """Return the deepest convolutional block, the layer Grad-CAM must hook.

    Selvaraju et al. ICCV 2017 recommend the last convolutional layer, which holds the
    highest level spatial semantics. It is features for DenseNet and layer4 for ResNet.
    """
    return model.features[-1] if arch == "densenet121" else model.layer4[-1]


@lru_cache(maxsize=4)
def load_classifier(model_id):
    """Load one classifier and keep it cached, so switching model costs no reloading."""
    spec = MODELS[model_id]
    model = _build_architecture(spec["arch"], len(spec["classes"]))
    model.load_state_dict(torch.load(spec["checkpoint"], map_location=DEVICE))
    model.to(DEVICE).eval()
    logger.info("Model %s (%s) loaded from %s on %s",
                model_id, spec["arch"], spec["checkpoint"], DEVICE)
    return model


@lru_cache(maxsize=1)
def load_vlm():
    """Load a local Vision Language Model, quantized when a GPU is available.

    Returns None when the local mode is disabled or the model cannot be loaded, in which
    case the service relies on the remote report server or on the structured writer
    rather than failing. MedGemma is gated on Hugging Face, so a token must be available
    in the environment for it to load.
    """
    if not ENABLE_LOCAL_VLM:
        return None
    try:
        from transformers import (AutoProcessor, AutoModelForImageTextToText,
                                  BitsAndBytesConfig)
        kwargs = {"device_map": "auto"}
        if DEVICE.type == "cuda":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
        model = AutoModelForImageTextToText.from_pretrained(VLM_MODEL_ID, **kwargs)
        model.eval()
        logger.info("Local VLM loaded: %s", VLM_MODEL_ID)
        return processor, model
    except Exception as exc:
        logger.warning("Local VLM unavailable (%s)", exc)
        return None


def _peak_zone(heatmap):
    """Map the heatmap maximum to a coarse anatomical zone.

    Image left is the patient right side, following the radiological convention.
    """
    y, x = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    height, width = heatmap.shape
    vertical = ("upper" if y < height / 3
                else "middle" if y < 2 * height / 3 else "lower")
    side = "right" if x < width / 2 else "left"
    return f"{side} {vertical} lung zone"


def _overlay_to_base64(overlay_array):
    """Encode an RGB numpy array as a base64 png string for the JSON response."""
    buffer = io.BytesIO()
    Image.fromarray(overlay_array).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def classify_and_localize(image, model_id):
    """Run one classifier and its Grad-CAM.

    The map always explains the pneumonia score, so a normal prediction shows which
    regions were inspected and found insufficient to raise that score, rather than a flat
    map that the min max normalisation of the library would stretch into pure noise.
    Activations below the seventieth percentile are cut, so only the regions that
    genuinely drive the score are painted.
    """
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    spec = MODELS[model_id]
    model = load_classifier(model_id)
    tensor = _transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    predicted_index = int(torch.argmax(probabilities))
    label = spec["classes"][predicted_index]
    confidence = float(probabilities[predicted_index])
    distribution = [{"label": name, "probability": round(float(p), 4)}
                    for name, p in zip(spec["classes"], probabilities)]

    positive_index = 1 - spec["classes"].index(spec["negative_class"])
    cam = GradCAM(model=model, target_layers=[_target_layer(model, spec["arch"])])
    heatmap = cam(input_tensor=tensor,
                  targets=[ClassifierOutputTarget(positive_index)])[0]

    threshold = np.percentile(heatmap, 70)
    heatmap = np.clip((heatmap - threshold) / (heatmap.max() - threshold + 1e-8), 0, 1)

    display = np.asarray(image.convert("RGB").resize((IMG_SIZE, IMG_SIZE)),
                         dtype=np.float32) / 255.0
    overlay = show_cam_on_image(display, heatmap, use_rgb=True, image_weight=0.65)
    return (label, confidence, overlay, _peak_zone(heatmap), distribution,
            spec["classes"][positive_index])


def _build_prompt(label, confidence, zone):
    """Build the text conditioned prompt selected by the project experiments.

    The explanation travels as language, never as pixels, and the instruction forbids
    meta commentary, which was the contamination observed when the model was free to
    discuss the overlay. The same wording is used by the remote server, so a report is
    identical whichever side generates it.
    """
    readable = str(label).replace("_", " ").lower()
    return (f"A diagnostic model classifies this chest radiograph as {readable} with "
            f"{confidence:.0%} confidence, and its attention is concentrated on the "
            f"{zone}. Examine that region and write the findings section of a radiology "
            f"report in three to five sentences, covering the lungs, the heart size and "
            f"the pleural spaces. State whether the image supports the suggested "
            f"finding. Describe only the anatomy. Never mention any diagnostic system, "
            f"model, heatmap, colours or confidence value.")


LIMITATION_SECTION = ("■ LIMITATIONS — This system reports pneumonia only and "
                      "does not exclude other pathology. It is a research prototype, not "
                      "a medical device, and every output requires review by a qualified "
                      "radiologist.")


def _wrap_vlm_report(text):
    """Give the generated paragraph the same section markers as the structured writer.

    The model returns a findings paragraph, so the marker is added here rather than asked
    for in the prompt, which keeps the prompt identical to the one that was evaluated.
    The limitation section is appended because it must appear whoever wrote the report.
    """
    body = text.strip()
    if not body.startswith("■"):
        body = f"■ FINDINGS — {body}"
    return f"{body}\n\n{LIMITATION_SECTION}"


def _encode_for_remote(image):
    """Downscale and encode the clean radiograph as a base64 png for the report server."""
    picture = image.convert("RGB")
    longest = max(picture.size)
    if longest > REMOTE_VLM_MAX_SIDE:
        scale = REMOTE_VLM_MAX_SIDE / longest
        picture = picture.resize((max(1, int(picture.width * scale)),
                                  max(1, int(picture.height * scale))),
                                 Image.LANCZOS)
    buffer = io.BytesIO()
    picture.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def remote_status():
    """Ask the report server whether it is alive, for the health and settings endpoints.

    The call is short and never raises, because the interface polls it while the user is
    pasting a new tunnel address and a closed tunnel is an ordinary situation here.
    """
    url = remote_vlm_url()
    if not url:
        return {"configured": False, "reachable": False, "url": None,
                "detail": "No report server configured, the structured writer is used."}
    try:
        response = requests.get(f"{url}/health", timeout=10)
        response.raise_for_status()
        payload = response.json()
        return {"configured": True, "reachable": True, "url": url,
                "model": payload.get("model"), "device": payload.get("device"),
                "detail": "Report server reachable."}
    except Exception as exc:
        return {"configured": True, "reachable": False, "url": url,
                "detail": f"Report server unreachable: {exc}"}


def remote_report(image, label, confidence, zone):
    """Ask the Kaggle report server for the findings, raising when it cannot answer."""
    url = remote_vlm_url()
    if not url:
        raise RuntimeError("no report server configured")
    response = requests.post(
        f"{url}/report",
        json={"image_b64": _encode_for_remote(image), "label": label,
              "confidence": float(confidence), "zone": zone},
        headers={"X-Auth": remote_vlm_secret()},
        timeout=REMOTE_VLM_TIMEOUT)
    if response.status_code == 401:
        raise RuntimeError("the shared secret of the service and of the notebook differ")
    response.raise_for_status()
    payload = response.json()
    if not payload.get("report"):
        raise RuntimeError(payload.get("error", "the server returned no report"))
    return payload["report"]


def local_report(image, label, confidence, zone):
    """Generate the findings with a MedGemma loaded in this process."""
    processor, model = load_vlm()
    messages = [
        {"role": "system",
         "content": [{"type": "text", "text": "You are an expert radiologist."}]},
        {"role": "user",
         "content": [{"type": "image", "image": image.convert("RGB")},
                     {"type": "text",
                      "text": _build_prompt(label, confidence, zone)}]},
    ]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt").to(model.device)
    length = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=220, do_sample=False)
    return processor.decode(output[0][length:], skip_special_tokens=True).strip()


def _structured_report(label, confidence, zone, distribution):
    """Deterministic writer used when no Vision Language Model can be reached.

    Sections are separated by a visible marker rather than by a newline alone, because a
    line break depends on how the client renders the field while a marker survives any
    rendering. The degraded mode stays explicit to the caller through the generator field
    of the response.
    """
    is_normal = str(label).upper() == "NORMAL"
    runner_up = min(p["probability"] for p in distribution)

    if is_normal:
        findings = (f"No focal airspace opacity is identified. The analysed regions, "
                    f"including the {zone} which carried the strongest activation, show "
                    f"no evidence of consolidation.")
        impression = (f"No radiographic evidence of pneumonia, with {confidence:.0%} "
                      f"confidence.")
    else:
        findings = (f"There is an area of increased opacity in the {zone}, compatible "
                    f"with an airspace consolidation. The remaining lung fields "
                    f"contributed less to the decision.")
        impression = (f"Appearances compatible with pneumonia, with {confidence:.0%} "
                      f"confidence.")

    sections = [
        f"■ FINDINGS — {findings} Cardiac silhouette, mediastinal contours and "
        f"pleural spaces are not assessed by this system and require review on the "
        f"original study.",
        f"■ IMPRESSION — {impression} The alternative class was scored at "
        f"{runner_up:.0%}.",
        LIMITATION_SECTION,
    ]
    return "\n\n".join(sections)


def generate_text(image, label, confidence, zone, distribution):
    """Produce the textual explanation and say which writer produced it.

    Returns the report, the identifier of the writer, and a warning when a Vision
    Language Model was expected but could not answer, so the interface can tell the user
    that the report on screen comes from the fallback.
    """
    warning = None

    if remote_vlm_url():
        try:
            return _wrap_vlm_report(
                remote_report(image, label, confidence, zone)), "vlm_remote", None
        except Exception as exc:
            warning = (f"The report server did not answer ({exc}), so the structured "
                       f"writer produced the text below.")
            logger.warning("Remote report failed: %s", exc)

    if load_vlm() is not None:
        try:
            return _wrap_vlm_report(
                local_report(image, label, confidence, zone)), "vlm_local", warning
        except Exception as exc:
            warning = (f"The local model failed ({exc}), so the structured writer "
                       f"produced the text below.")
            logger.warning("Local report failed: %s", exc)

    return (_structured_report(label, confidence, zone, distribution),
            "structured", warning)


def analyze_image(image, model_id, with_text=True):
    """Run the pipeline on one image with one model and return the response payload.

    The comparison mode calls this function once per model with with_text set to false
    for the secondary one, so both heatmaps are produced without paying twice the cost of
    text generation.
    """
    spec = MODELS[model_id]
    (label, confidence, overlay, zone, distribution,
     cam_label) = classify_and_localize(image, model_id)
    if with_text:
        text, generator, warning = generate_text(image, label, confidence, zone,
                                                 distribution)
    else:
        text, generator, warning = None, None, None
    return {
        "model_id": model_id,
        "model_name": spec["name"],
        "label": label,
        "confidence": round(confidence, 4),
        "probabilities": distribution,
        "negative_class": spec["negative_class"],
        "peak_zone": zone,
        "cam_label": cam_label,
        "explanation_text": text,
        "generator": generator,
        "generator_warning": warning,
        "heatmap_png_base64": _overlay_to_base64(overlay),
        "disclaimer": ("Research prototype produced for a Master thesis. "
                       "Not a medical device. Not for clinical use."),
    }