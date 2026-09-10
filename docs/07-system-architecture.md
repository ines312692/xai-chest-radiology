# System architecture

Three artefacts, three purposes. The notebooks are the experimental record, the `src/`
package is the library form of the same pipeline, and the application is the
demonstration that runs the whole chain on an uploaded radiograph.

## 1. The notebooks, `notebooks/`

Each notebook is self contained and reproduces one experiment end to end on Kaggle,
with its methodological justifications written in the markdown cells. They stay the
record of what was actually run and what it produced; the package does not replace
them.

| Notebook | What it does |
|---|---|
| `resnet50/rsna-resnet50-full-pipeline.ipynb` | Preprocessing, training, evaluation on RSNA |
| `densent121/rsna-densenet121-full-pipeline.ipynb` | Same protocol, backbone changed only, plus a Grad-CAM figure by confusion category |
| `resnet50/rsna-cam-evaluation.ipynb` | Five saliency variants, mIoU, hit rate, deletion, insertion, ranking |

## 2. The package, `src/`

The same pipeline as reusable modules, so a step can be read, tested and re run alone,
and so changing the backbone is a line in a configuration file rather than a copy of a
notebook.

```
src/
├── configs/          one YAML file per experiment
├── scripts/          preprocess, train, evaluate, explain, benchmark_cams
├── tests/            unit tests on synthetic arrays, no dataset needed
└── xai_chest/
    ├── config.py     what an experiment is, and where its files live
    ├── utils.py      seed, logging, device, dataset discovery
    ├── data/         DICOM decoding, split, dataset, loaders
    ├── models/       backbone factory and saliency target layer
    ├── training/     fine tuning loop and the metric set
    ├── explain/      saliency production and the two scoring protocols
    └── reporting/    the figures of the thesis
```

Two design rules keep the structure honest. A module never reads a configuration file
and never parses arguments, it receives values, which is what makes it testable. A
script never contains method logic, it wires modules together, which is what keeps two
experiments identical except for their configuration.

Run order and options are in `../src/README.md`.

## 3. The application, `xai-radiology-app/`

A FastAPI service with a single page interface, called Thorax AI. It runs the three
stages of the pipeline on an uploaded radiograph: classification, saliency map, written
report.

### What the interface offers

1. A backbone selector, DenseNet121 or ResNet50, both trained on RSNA under the same
   frozen split.
2. A saliency method selector, the five methods of the study, each shown with its
   benchmark scores from [06-results.md](06-results.md).
3. A comparison mode: none, across backbones with one method, or across methods with
   one backbone. The second answers a question a jury asks, how much of the explanation
   is the method rather than the model.
4. Three views of the study, radiograph, overlay with an opacity slider, and saliency
   alone.
5. The written report, with the name of whichever writer produced it.

### The three stage chain

```
upload ──> classifier (CPU, local)  ──> label, confidence, class scores
              │
              └──> saliency map (CPU, local) ──> overlay shown to the human
                        │
                        └──> peak anatomical zone, as text
                                  │
        clean image + class + confidence + zone ──> report writer
```

The report writer is tried in three steps: the remote MedGemma server, then a local
MedGemma if the machine has a GPU and the local mode is enabled, then a deterministic
structured writer. When a fallback is used, the response carries the reason and the
interface displays it above the report, so a structured text is never mistaken for a
generated one.

### Why the heatmap does not travel to the language model

Because measuring said so. A tinted radiograph collapses the sensitivity of the model,
while naming the region in words preserves it, so the explanation crosses the boundary
as language and never as pixels. The prompt also forbids any mention of the system, the
model, the heatmap, colours or confidence values, which removes the meta commentary
observed when the model was free to discuss the overlay instead of the anatomy.

### The remote report server

MedGemma 4B in four bit quantisation needs a GPU, which the demonstration laptop does
not have. It runs in a Kaggle session behind a Cloudflare tunnel, and the application
calls it over HTTP.

```
GET  {tunnel}/health    liveness, model, GPU
POST {tunnel}/report    {image_b64, label, confidence, zone}, header X-Auth
                        -> {report, model}
```

The tunnel address changes at every restart of the notebook, so it is entered from the
interface and persisted in `runtime_vlm.json` next to the code, rather than edited in
the source. The radiograph is downscaled to 896 pixels on its longest side before it
travels, since the model resizes it to its own vision resolution anyway.

Endpoints added to the service for this: `GET /cams`, `GET /vlm`, `POST /vlm`, and two
new form fields on `POST /analyze`, `cam_method` and `compare`.

## Known limitation of the current implementation

`explain/cam.peak_zone` divides the image into thirds with no lung mask, so a peak
falling on a shoulder is still named as a lung zone, and that name is what travels to
the language model. The label is therefore indicative and must be read together with
the map. Replacing the grid by a lung mask is the next improvement, and the case where
the peak falls outside the mask is itself a useful indicator for RQ1.
