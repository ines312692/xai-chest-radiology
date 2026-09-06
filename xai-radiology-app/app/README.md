# XAI Chest Radiology, Web Service

Deployable version of the project pipeline: a chest radiograph goes in, a
classification, a Grad-CAM visual explanation and a generated textual explanation
come out.

## Architecture

```
frontend (single HTML file)
        |  POST /analyze  multipart image
        v
FastAPI service
        |
        +-- DenseNet121 classifier      label and confidence
        +-- Grad-CAM                    heatmap overlay, peak anatomical zone
        +-- Vision Language Model       textual explanation
```

### Why the VLM never receives the overlay

The ablation study of the project measured that passing the Grad-CAM overlay to the
Vision Language Model degrades every generation metric and can induce hallucinations,
the model describing the warm colored blob as a mass. The service therefore separates
the two explanation channels. The overlay is rendered for the user, while the model
receives the clean radiograph together with a textual summary of the explanation, that
is the predicted class, its confidence and the anatomical zone of strongest activation.
This is a measured design decision, not a default.

## Layout

```
app/
├── Dockerfile
├── README.md
└── backend/
    ├── main.py            FastAPI routes
    ├── inference.py       pipeline, model loading, prompt construction
    ├── config.py          environment driven configuration
    ├── requirements.txt
    ├── models/            place densenet121_rsna_best.pth here, not versioned
    └── static/
        └── index.html     single page frontend, no build step
```

## Run locally

```bash
cd backend
mkdir -p models
# copy densenet121_rsna_best.pth into models/
pip install -r requirements.txt
uvicorn main:app --reload
```

Open http://localhost:8000

## Run with Docker

```bash
docker build -t xai-radiology .
docker run -p 8000:8000 -v $(pwd)/backend/models:/app/models xai-radiology
```

## Degraded mode without a GPU

The Vision Language Model needs about six gigabytes of GPU memory. Without a GPU, or
with `ENABLE_VLM=false`, the service keeps the classifier and Grad-CAM and produces the
textual explanation with a deterministic rule based writer. The response field
`generator` reports which writer produced the text, so a demonstration on a laptop stays
honest about what is running.

```bash
ENABLE_VLM=false uvicorn main:app
```

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `CHECKPOINT_PATH` | `models/densenet121_rsna_best.pth` | classifier weights |
| `VLM_MODEL_ID` | `llava-hf/llava-1.5-7b-hf` | Hugging Face model identifier |
| `ENABLE_VLM` | `true` | set to false to force the rule based writer |
| `DECISION_THRESHOLD` | `0.5` | pneumonia probability threshold |

## API

`GET /health` returns the device, whether the VLM is enabled and which model is set.

`POST /analyze`, multipart field `file`, returns:

```json
{
  "label": "PNEUMONIA",
  "confidence": 0.9412,
  "peak_zone": "right lower lung zone",
  "explanation_text": "...",
  "generator": "vlm",
  "heatmap_png_base64": "iVBORw0...",
  "disclaimer": "Research prototype ... Not for clinical use."
}
```

Interactive documentation is served at `/docs`.

## Status

Research prototype produced for a Master thesis. Not a medical device, not validated
for clinical use, and trained on public research datasets whose distribution differs
from routine clinical acquisition.
