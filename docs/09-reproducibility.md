# Reproducibility

## The guarantees

1. Seed 42 is applied to the python, numpy and torch generators in every notebook and
   in `src/xai_chest/utils.py`.
2. Splits are frozen to CSV before any training. The same seed and the same split code
   across backbones make every comparison controlled: both see exactly the same
   patients.
3. Metrics are saved as JSON and per image predictions as CSV, versioned per experiment
   under `results/`. The three RSNA experiments are committed there; the earlier
   COVID-19 and Kermany runs still have to be exported from Kaggle.
4. Each notebook resolves its inputs by file search and can be re run on Kaggle by
   attaching the listed datasets.
5. The `src/` package writes the resolved configuration, defaults included, next to the
   results of each run, so a run can be reproduced from its own output folder.
6. The unit tests verify that the split is leak free, deterministic and stratified,
   that boxes are rescaled correctly and that the saliency scores behave as their
   definition says.

## Running the notebooks on Kaggle

1. Create a Kaggle account and verify the phone number, which is required for GPU and
   internet access.
2. For RSNA, open the competition page once and accept the rules.
3. Create a notebook, then File, Import Notebook, and load the target `.ipynb`.
4. Add Input: attach the datasets listed at the top of the notebook, plus the results
   dataset when the notebook consumes previous outputs, as `rsna-cam-evaluation` does.
5. Settings: Accelerator GPU T4, Internet On.
6. Run All, then Save Version, Save and Run All to persist the outputs.
7. From the committed notebook Output tab, create or update a Kaggle dataset with the
   produced files so the next notebook can attach it.

## Running the package

Locally:

```bash
cd src
pip install -e .
python scripts/preprocess.py     --config configs/resnet50_rsna.yaml
python scripts/train.py          --config configs/resnet50_rsna.yaml
python scripts/evaluate.py       --config configs/resnet50_rsna.yaml
python scripts/explain.py        --config configs/resnet50_rsna.yaml
python scripts/benchmark_cams.py --config configs/resnet50_rsna.yaml
python -m pytest tests -q
```

On Kaggle, publish `src/` as a dataset and call the same scripts:

```python
!pip install grad-cam --quiet
!cp -r /kaggle/input/<your-code-dataset>/src /kaggle/working/src
%cd /kaggle/working/src
!python scripts/train.py --config configs/resnet50_rsna.yaml
```

The dataset root is resolved by searching for `stage_2_train_labels.csv`, first under
the Kaggle mount and then under the current directory, so the same command works in
both places. Pass `--data-root` to be explicit.

## Comparing the two backbones

Run the same five commands with `configs/densenet121_rsna.yaml`. With the same seed and
the same proportions the partition is identical, which the test suite verifies, so the
only difference between the two runs is the architecture.

## Running the application

```bash
cd xai-radiology-app/app/backend
pip install -r requirements.txt
uvicorn main:app --port 8010
```

Then open `http://localhost:8010`. For the generated report, start the MedGemma
notebook on Kaggle with GPU T4 and internet, copy the tunnel address it prints, and
paste it into the report server dialog of the interface. Without it the application
falls back to the structured writer and says so.

## What is not reproducible from this repository alone

Model weights and images are never committed. The `.pth` files present under
`results/` are mirrors of the Kaggle datasets used as inputs by the downstream
notebooks, and the datasets themselves are attached from Kaggle. Re running the
pipeline therefore requires a Kaggle account with the competition rules accepted.
