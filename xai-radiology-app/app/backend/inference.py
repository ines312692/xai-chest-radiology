"""Inference core of the service.

The module exposes one entry point, analyze_image, which runs the three stages of the
pipeline for a chosen model: classification, visual explanation with Grad-CAM, and
textual explanation.

Multi model note. Several classifiers can be served side by side. Each one is loaded
once and cached, so switching model in the interface costs no reloading. Comparing the
heatmaps of two models on the same radiograph shows how the explanation depends on what
each model was trained to detect, which is the point of the comparison mode.

Multi class note. Grad-CAM always explains the predicted class specifically, which makes
the visual explanation class discriminative, the founding property of the method in
Selvaraju et al. ICCV 2017.

Design note. The ablation study of the project measured that passing the Grad-CAM
overlay to the Vision Language Model degrades every generation metric and can induce
hallucinations, the model reading the colored blob as a mass. The service therefore
sends the clean radiograph to the VLM and transmits the explanation as text, namely the
predicted class, its confidence and the anatomical zone of strongest activation. The
overlay is produced for the user interface only.
"""
import base64
import io
import logging
import os
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torchvision import models as tv_models

from config import (ENABLE_VLM, IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE, MODELS,
                    VLM_MODEL_ID)

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def available_models():
    """Describe every declared model and whether its checkpoint is present on disk."""
    described = []
    for model_id, spec in MODELS.items():
        described.append({
            "id": model_id,
            "name": spec["name"],
            "classes": spec["classes"],
            "description": spec["description"],
            "available": os.path.exists(spec["checkpoint"]),
        })
    return described


@lru_cache(maxsize=4)
def load_classifier(model_id):
    """Load one DenseNet121 classifier and keep it cached.

    The head is sized from the declared class list, so the same function loads the
    binary and the four class checkpoints without any code change.
    """
    spec = MODELS[model_id]
    model = tv_models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, len(spec["classes"]))
    model.load_state_dict(torch.load(spec["checkpoint"], map_location=DEVICE))
    model.to(DEVICE).eval()
    logger.info("Model %s loaded from %s with %d classes on %s",
                model_id, spec["checkpoint"], len(spec["classes"]), DEVICE)
    return model


@lru_cache(maxsize=1)
def load_vlm():
    """Load the Vision Language Model, quantized when a GPU is available.

    Returns None when the model is disabled or cannot be loaded, in which case the
    service falls back to the rule based writer instead of failing.
    """
    if not ENABLE_VLM:
        logger.warning("VLM disabled by configuration, using the rule based writer")
        return None
    try:
        from transformers import (AutoProcessor, BitsAndBytesConfig,
                                  LlavaForConditionalGeneration)
        kwargs = {"device_map": "auto"}
        if DEVICE.type == "cuda":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
        model = LlavaForConditionalGeneration.from_pretrained(VLM_MODEL_ID, **kwargs)
        model.eval()
        logger.info("VLM loaded: %s", VLM_MODEL_ID)
        return processor, model
    except Exception as exc:
        logger.warning("VLM unavailable (%s), using the rule based writer", exc)
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

    Returns the predicted label, its confidence, the overlay, the peak anatomical zone
    and the full probability distribution over the classes of that model.
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

    cam = GradCAM(model=model, target_layers=[model.features[-1]])
    heatmap = cam(input_tensor=tensor,
                  targets=[ClassifierOutputTarget(predicted_index)])[0]

    display = np.asarray(image.convert("RGB").resize((IMG_SIZE, IMG_SIZE)),
                         dtype=np.float32) / 255.0
    overlay = show_cam_on_image(display, heatmap, use_rgb=True, image_weight=0.6)
    return label, confidence, overlay, _peak_zone(heatmap), distribution


def _build_prompt(label, confidence, zone):
    """Build the text conditioned prompt validated by the project ablation study.

    The explanation travels as language, and the instruction forbids meta commentary,
    which was the contamination observed when the model could discuss the overlay.
    """
    readable = label.replace("_", " ").lower()
    return (f"You are an expert radiologist. A diagnostic system classifies this chest "
            f"radiograph as {readable} with {confidence:.0%} confidence, the most "
            f"influential region being the {zone}. Write a short radiology style "
            f"description of the visible findings in two to four sentences, paying "
            f"attention to that region and covering the lungs, heart size and pleural "
            f"spaces. State only what is visible. Describe only the anatomy. Never "
            f"mention any diagnostic system, model, heatmap, colors or confidence "
            f"value.")


def _rule_based_report(label, zone):
    """Deterministic fallback used when the VLM is not available.

    It keeps the service usable on a laptop, and the degraded mode stays explicit to
    the caller through the generator field of the response.
    """
    templates = {
        "NORMAL": (f"No focal consolidation is identified, the analysed regions "
                   f"including the {zone} showing no evidence of pneumonia."),
        "Normal": (f"The lung fields appear clear, the analysed regions including the "
                   f"{zone} showing no evidence of focal consolidation."),
        "PNEUMONIA": (f"Findings suggest an area of increased opacity in the {zone}, "
                      f"compatible with a pulmonary consolidation."),
        "Lung_Opacity": (f"An area of increased opacity is suggested in the {zone}, "
                         f"compatible with a focal pulmonary process."),
        "COVID": (f"Findings suggest a pattern of ground glass and patchy opacities, "
                  f"most prominent in the {zone}, a distribution described in viral "
                  f"pneumonia of COVID type."),
        "Viral Pneumonia": (f"Findings suggest a diffuse interstitial pattern involving "
                            f"the {zone}, a distribution described in viral pneumonia."),
    }
    body = templates.get(label, f"The findings are centred on the {zone}.")
    return (body + " Cardiac silhouette and pleural spaces should be assessed on the "
                   "original study, and correlation with the clinical presentation is "
                   "recommended.")


def generate_text(image, label, confidence, zone):
    """Produce the textual explanation, with the rule based writer as fallback."""
    loaded = load_vlm()
    if loaded is None:
        return _rule_based_report(label, zone), "rule_based"

    processor, model = loaded
    conversation = [{"role": "user",
                     "content": [{"type": "image"},
                                 {"type": "text",
                                  "text": _build_prompt(label, confidence, zone)}]}]
    chat = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image.convert("RGB"), text=chat,
                       return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=120, do_sample=False)
    decoded = processor.decode(output[0], skip_special_tokens=True)
    return decoded.split("ASSISTANT:")[-1].strip(), "vlm"


def analyze_image(image, model_id, with_text=True):
    """Run the pipeline on one image with one model and return the response payload.

    The comparison mode calls this function once per model with with_text set to false
    for the secondary model, so the two heatmaps are produced without paying twice the
    cost of text generation.
    """
    spec = MODELS[model_id]
    label, confidence, overlay, zone, distribution = classify_and_localize(
        image, model_id)
    if with_text:
        text, generator = generate_text(image, label, confidence, zone)
    else:
        text, generator = None, None
    return {
        "model_id": model_id,
        "model_name": spec["name"],
        "label": label,
        "confidence": round(confidence, 4),
        "probabilities": distribution,
        "negative_class": spec["negative_class"],
        "peak_zone": zone,
        "explanation_text": text,
        "generator": generator,
        "heatmap_png_base64": _overlay_to_base64(overlay),
        "disclaimer": ("Research prototype produced for a Master thesis. "
                       "Not a medical device. Not for clinical use."),
    }