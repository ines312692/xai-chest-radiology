"""Configuration of the XAI chest radiology service.

The service is focused on pneumonia detection and serves the two backbones the project
trained on the RSNA Pneumonia Detection Challenge, under the same frozen patient level
split. Serving both makes the comparison mode meaningful: the same radiograph, the same
task, two architectures, two saliency maps.

The language step is not executed here. MedGemma runs in a Kaggle session with a GPU and
is reached over a public tunnel, so the laptop only carries the classifiers and Grad-CAM,
which are comfortable on CPU. The address of that tunnel changes every time the Kaggle
session is restarted, therefore it is stored in a small runtime file that the interface
can rewrite without touching this source file.
"""
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    "densenet": {
        "name": "DenseNet121",
        "arch": "densenet121",
        "checkpoint": os.getenv("DENSENET_CHECKPOINT",
                                "models/densenet121_rsna_best.pth"),
        "classes": ["NORMAL", "PNEUMONIA"],
        "negative_class": "NORMAL",
        "description": ("DenseNet121 trained on the RSNA Pneumonia Detection Challenge, "
                        "adult radiographs, patient level split. Test accuracy 0.942, "
                        "AUC 0.985. Seven million parameters."),
    },
    "resnet": {
        "name": "ResNet50",
        "arch": "resnet50",
        "checkpoint": os.getenv("RESNET_CHECKPOINT",
                                "models/resnet50_rsna_best.pth"),
        "classes": ["NORMAL", "PNEUMONIA"],
        "negative_class": "NORMAL",
        "description": ("ResNet50 trained on the same split and the same protocol. "
                        "Test accuracy 0.945, AUC 0.987. Twenty three million "
                        "parameters."),
    },
}

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "densenet")

# Hugging Face identifier of the Vision Language Model producing the textual explanation.
# MedGemma is the model validated by the project, it needs a GPU and an accepted licence.
VLM_MODEL_ID = os.getenv("VLM_MODEL_ID", "google/medgemma-4b-it")

# Address printed by the Kaggle notebook when the Cloudflare tunnel opens, for example
# https://florence-membership-duke-clothes.trycloudflare.com
# Leave it empty here and paste the address in the interface instead, since it changes at
# every restart of the notebook. The value written from the interface takes precedence.
REMOTE_VLM_URL = os.getenv("REMOTE_VLM_URL", "")

# Shared secret, it only has to match the SHARED_SECRET of the notebook
REMOTE_VLM_SECRET = os.getenv("REMOTE_VLM_SECRET", "thorax-demo-2026")

# A four bit MedGemma on a T4 writes a report in roughly ten to thirty seconds, and the
# very first request also pays the warm up, so the timeout is deliberately generous.
REMOTE_VLM_TIMEOUT = float(os.getenv("REMOTE_VLM_TIMEOUT", "180"))

# The radiograph is downscaled before it travels, because MedGemma resizes it to its own
# vision resolution anyway and a full size study would only make the upload slower.
REMOTE_VLM_MAX_SIDE = int(os.getenv("REMOTE_VLM_MAX_SIDE", "896"))

# Loading MedGemma inside this process is only meaningful on a machine with a GPU. It
# stays off by default, so the service never tries to download four billion parameters on
# a laptop. The remote server is the normal path and the structured writer is the fallback.
ENABLE_LOCAL_VLM = os.getenv("ENABLE_LOCAL_VLM", "false").lower() == "true"

# Input resolution and normalization imposed by the ImageNet pretrained backbone
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

MAX_UPLOAD_MB = 20

# ---------------------------------------------------------------------------
# Runtime settings of the remote report server
#
# The tunnel address is written next to this file rather than kept in memory, so that a
# restart of the service during a demonstration does not lose it. The file is created on
# the first save and can be deleted safely.
# ---------------------------------------------------------------------------
RUNTIME_FILE = os.path.join(BASE_DIR, "runtime_vlm.json")


def _runtime_settings():
    """Read the settings saved from the interface, an empty dict when there are none."""
    try:
        with open(RUNTIME_FILE, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def remote_vlm_url():
    """Return the address of the report server, the saved one first."""
    saved = _runtime_settings().get("url")
    return (saved or REMOTE_VLM_URL or "").strip().rstrip("/")


def remote_vlm_secret():
    """Return the shared secret, the saved one first."""
    return _runtime_settings().get("secret") or REMOTE_VLM_SECRET


def save_remote_vlm(url, secret=None):
    """Persist a new tunnel address, and the secret when one is given.

    An empty address clears the setting, which sends the service back to the structured
    writer without needing a restart.
    """
    settings = _runtime_settings()
    settings["url"] = (url or "").strip().rstrip("/")
    if secret:
        settings["secret"] = secret.strip()
    with open(RUNTIME_FILE, "w", encoding="utf-8") as handle:
        json.dump(settings, handle, indent=2)
    return {"url": settings["url"], "secret": settings.get("secret", REMOTE_VLM_SECRET)}