"""Configuration of the XAI chest radiology service.

The service can serve several trained classifiers at once. Each entry of MODELS
declares one checkpoint, its class order and a short description shown in the user
interface. The class order must match exactly the order used during training,
otherwise the predicted indices point to the wrong labels.
"""
import os

MODELS = {
    "rsna": {
        "name": "RSNA binary",
        "checkpoint": os.getenv("RSNA_CHECKPOINT",
                                "models/densenet121_rsna_best.pth"),
        "classes": ["NORMAL", "PNEUMONIA"],
        "negative_class": "NORMAL",
        "description": ("DenseNet121 trained on the RSNA Pneumonia Detection "
                        "Challenge, adult radiographs, patient level split. "
                        "Test AUC 0.985."),
    },
    "covid": {
        "name": "COVID four classes",
        "checkpoint": os.getenv("COVID_CHECKPOINT",
                                "models/densenet121_covid_best.pth"),
        "classes": ["Normal", "Lung_Opacity", "COVID", "Viral Pneumonia"],
        "negative_class": "Normal",
        "description": ("DenseNet121 trained on the COVID-19 Radiography Database, "
                        "four classes. Test accuracy 0.952, macro F1 0.953."),
    },
}

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "covid")

# Hugging Face identifier of the Vision Language Model producing the textual explanation
VLM_MODEL_ID = os.getenv("VLM_MODEL_ID", "llava-hf/llava-1.5-7b-hf")

# When false the service skips the VLM and uses the rule based writer instead,
# which keeps the demonstration runnable without a GPU.
ENABLE_VLM = os.getenv("ENABLE_VLM", "true").lower() == "true"

# Input resolution and normalization imposed by the ImageNet pretrained backbone
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

MAX_UPLOAD_MB = 20