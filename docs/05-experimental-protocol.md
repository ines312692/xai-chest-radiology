# Experimental protocol

The protocol is frozen: it is identical across backbones and datasets, so any
difference in a result is attributable to the variable under study and not to the
setup. Every value below is implemented in exactly one place in the code, named in the
last column.

## Preprocessing

1. **Dynamic path resolution.** Dataset roots are located by searching for a known
   file, `stage_2_train_labels.csv` for RSNA, rather than by a hardcoded mount path.
2. **Clean class definition.** Ambiguous groups are excluded. For RSNA, the group No
   Lung Opacity Not Normal contains other pathologies without pneumonia and would teach
   the model that an effusion is normal. Ensemble-CAM [4] excludes it the same way.
3. **Patient level split** wherever a patient identifier exists, with disjointness
   verified by assertions rather than assumed. This corrects the limitation
   acknowledged by PLOS ONE 2024 [3]. For the COVID-19 Radiography Database, which has
   no patient identifier, the split is stratified at image level and this threat to
   validity is stated rather than hidden.
4. **Frozen splits.** Train, validation and test partitions are written to CSV before
   any training, with the same seed everywhere, so both backbones see exactly the same
   patients.
5. **Expert annotations preserved.** The RSNA boxes of the test split are exported at
   preprocessing time, so the explanation stage never touches the training data.

## Traceability of every choice

| Choice | Value | Source | Implemented in |
|---|---|---|---|
| Input resolution | 224 by 224 | ImageNet pretraining; PLOS ONE 2024 [3]; MDPI 2025 [5] | `src/xai_chest/config.py` |
| Normalisation | ImageNet channel statistics | Imposed by the pretrained weights | `src/xai_chest/config.py` |
| Augmentation | rotation 15 degrees, horizontal flip, zoom 0.8 to 1.2 | Exact parameters of MDPI Information 2025 [5] | `src/xai_chest/data/datasets.py` |
| Split | 70 10 20, patient level, stratified | Scientific Reports 2025 [2]; MedVAG [10]; corrects PLOS ONE [3] | `src/xai_chest/data/splits.py` |
| Loss | Cross entropy weighted by inverse frequency | Ensemble-CAM [4]; weighting motivated by Scientific Reports 2025 [2] | `src/xai_chest/training/engine.py` |
| Optimizer | Adam, learning rate 1e-4 | Fine tuning value of Nature Machine Intelligence 2022 [1] | `src/xai_chest/training/engine.py` |
| Scheduler | ReduceLROnPlateau on the validation AUC | PLOS ONE 2024 [3]; Ensemble-CAM [4] | `src/xai_chest/training/engine.py` |
| Early stopping | On the validation AUC | PLOS ONE 2024 [3]; Ensemble-CAM [4] | `src/xai_chest/training/engine.py` |
| Classification metrics | Accuracy, precision, recall, F1, AUC | Protocol of PLOS ONE 2024 [3] | `src/xai_chest/training/metrics.py` |
| CAM target layer | Deepest convolutional block, `layer4` for ResNet50 and `features[-1]` for DenseNet121 | Selvaraju et al. [14] | `src/xai_chest/models/factory.py` |
| CAM localisation | mIoU and hit rate read jointly, map binarised at half of its maximum | Nature Machine Intelligence 2022 [1] | `src/xai_chest/explain/evaluation.py` |
| CAM faithfulness | Deletion and insertion, 20 steps, blurred baseline | RISE, Petsiuk et al. [19] | `src/xai_chest/explain/evaluation.py` |
| Report metrics | BLEU-4, ROUGE-L, METEOR, BERTScore | ClinicalBLIP [9], MedVAG [10], ChestX-Transcribe [11] | Report stage, in progress |
| Factuality metrics | RadGraph F1, CheXbert F1, entity level CHAIR rate | LLaVA-Rad [7], Flamingo-CXR [8], MedVH [12] | Report stage, in progress |
| Seed | 42, applied to python, numpy and torch | Reproducibility deliverable | `src/xai_chest/utils.py` |

## Why two saliency protocols rather than one

Localisation asks whether the map agrees with a radiologist. Faithfulness asks whether
the highlighted pixels are the ones the network actually used. These are different
questions and a method can pass one and fail the other.

Nature Machine Intelligence 2022 [1] states the limitation that motivates the second
protocol: when a map scores poorly against expert boxes, it is impossible to know
whether the method is bad or the model looked at the wrong place. Deletion and
insertion answer that, because they compare the map to the behaviour of the model
itself and need no annotation.

The measured outcome justifies the effort. On our RSNA test set the two protocols
disagree: Score-CAM localises best while Grad-CAM is the most faithful. Reporting only
mIoU would have selected a different method for the report generation stage.

## Metric directions, to avoid a reading error

| Metric | Direction | Meaning |
|---|---|---|
| mIoU | higher is better | Overlap between the binarised map and the union of the expert boxes |
| Hit rate | higher is better | The single most activated pixel falls inside an expert box |
| Deletion AUC | **lower** is better | Removing the highlighted pixels first makes the probability collapse fast |
| Insertion AUC | higher is better | Restoring the highlighted pixels first makes the probability recover fast |
| Insertion minus deletion | higher is better | One number summarising faithfulness |
| CHAIR rate | **lower** is better | Share of mentioned pathologies absent from the ground truth |
