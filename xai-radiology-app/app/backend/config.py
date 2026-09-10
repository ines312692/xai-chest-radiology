"""Configuration of the XAI chest radiology service.

The service is focused on pneumonia detection and serves the two backbones the project
trained on the RSNA Pneumonia Detection Challenge, under the same frozen patient level
split. Serving both makes the comparison mode meaningful: the same radiograph, the same
task, two architectures, two saliency maps.
"""
import os

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

# When false the service skips the VLM and uses the structured writer instead,
# which keeps the demonstration runnable without a GPU.
ENABLE_VLM = os.getenv("ENABLE_VLM", "true").lower() == "true"

# Input resolution and normalization imposed by the ImageNet pretrained backbone
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

MAX_UPLOAD_MB = 20