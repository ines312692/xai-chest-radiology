"""Reading of the clinical image format.

RSNA ships DICOM files, the clinical standard. Everything downstream expects an eight
bit grayscale array, so the conversion lives here alone and no other module knows what
a DICOM is.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pydicom
from PIL import Image


def read_dicom(filepath: str | Path) -> np.ndarray:
    """Decode one DICOM study and return an eight bit grayscale array.

    The pixel array is rescaled to the full range rather than clipped, because DICOM
    stores raw detector values whose range varies from one study to the next, and a
    fixed window would darken or saturate part of the dataset.
    """
    dicom = pydicom.dcmread(str(filepath))
    array = dicom.pixel_array.astype(np.float32)
    array = (array - array.min()) / (array.max() - array.min() + 1e-8) * 255.0
    return array.astype(np.uint8)


def load_study(filepath: str | Path, image_size: int) -> Tuple[Image.Image, np.ndarray,
                                                               Tuple[int, int]]:
    """Return the study as a three channel image, its display array and its true size.

    Three channels because the ImageNet pretrained backbones expect RGB input, the
    display array in the range zero to one for the saliency overlays, and the original
    height and width so the expert boxes can be rescaled to the network resolution.
    """
    array = read_dicom(filepath)
    image = Image.fromarray(array).convert("RGB")
    display = np.asarray(image.resize((image_size, image_size)),
                         dtype=np.float32) / 255.0
    return image, display, array.shape
