# Results

Bold marks the best value of a comparison.

Provenance. The RSNA numbers come from the files committed under `../results/`,
`rsna_resnet50/`, `rsna_densenet121/` and `rsna_cam_evaluation/`, each holding its
metrics JSON, its per image predictions, its frozen splits and its figures. The
COVID-19 Radiography and Kermany numbers come from earlier Kaggle runs of the same
protocol whose result folders are not committed in this repository; they are reported
here for the cross dataset comparison and should be re exported from Kaggle before the
thesis is submitted, so that every table in this document is backed by a file.

## 1. Classification, cross backbone comparison on identical frozen splits

| Dataset | Metric | ResNet50 | DenseNet121 | Winner |
|---|---|---|---|---|
| RSNA, binary, adult, patient level split | Accuracy | **0.9448** | 0.9418 | ResNet50, marginal |
| | F1 | **0.9320** | 0.9262 | |
| | AUC | **0.9868** | 0.9853 | |
| COVID-19 Radiography, 4 classes | Accuracy | 0.9313 | **0.9521** | DenseNet121, clear |
| | Macro F1 | 0.9297 | **0.9534** | |
| | Macro AUC one versus rest | 0.9913 | **0.9945** | |
| Kermany, binary, pediatric | Accuracy | 0.7740 | **0.9215** | DenseNet121, clear |
| | F1 | 0.8469 | **0.9400** | |
| | AUC | 0.9726 | **0.9764** | |

**Main finding.** No backbone dominates across settings. DenseNet121 wins clearly on
the pediatric dataset and on the multi class multi source one, while the two are
statistically indistinguishable on the adult binary task, where they differ mainly in
their precision recall trade off: DenseNet121 precision 0.952 and recall 0.902,
ResNet50 precision 0.930 and recall 0.934. Backbone choice should therefore be
validated per task rather than inherited from the literature.

The recall difference matters clinically more than the accuracy difference, since a
false negative is a missed pneumonia. On that criterion ResNet50 is preferable on RSNA
despite the near tie on accuracy.

## 2. Four backbone study on Kermany, identical protocol

| Backbone | Parameters, millions | Accuracy | F1 | AUC |
|---|---|---|---|---|
| DenseNet121 | 7.0 | **0.9215** | **0.9400** | **0.9764** |
| ResNet50 | 23.5 | 0.7740 | 0.8469 | 0.9726 |
| EfficientNet B0 | 4.0 | 0.8798 | 0.9117 | 0.9692 |
| ViT B16 | 85.8 | 0.8926 | 0.9161 | 0.9635 |

The Vision Transformer underperforms the best convolutional network despite twelve
times more parameters, which is consistent with the known data hunger of transformers
at this dataset scale. DenseNet121 wins with the smallest parameter count, which is
also why CheXNet [22] chose that family.

## 3. Per class detail, COVID-19 Radiography test set, DenseNet121

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Normal | 0.9342 | 0.9750 | 0.9542 | 2039 |
| Lung Opacity | 0.9572 | 0.9102 | 0.9331 | 1203 |
| COVID | 0.9846 | 0.9723 | 0.9784 | 723 |
| Viral Pneumonia | 0.9879 | 0.9108 | 0.9478 | 269 |

DenseNet121 corrects the two weaknesses of the ResNet50 run on the same split: Lung
Opacity recall rises from 0.867 to 0.910, and Viral Pneumonia precision from 0.868 to
0.988.

## 4. RQ1 answered. Five saliency methods on RSNA, 50 true positive images

| Method | mIoU | Hit rate | Deletion, lower better | Insertion, higher better | Faithfulness, insertion minus deletion |
|---|---|---|---|---|---|
| Score-CAM | **0.2853** | 0.72 | 0.7258 | 0.9811 | 0.2554 |
| **Grad-CAM** | 0.2683 | 0.70 | **0.6874** | **0.9876** | **0.3002** |
| Grad-CAM++ | 0.2671 | **0.74** | 0.7001 | 0.9855 | 0.2855 |
| LayerCAM | 0.2660 | **0.74** | 0.7057 | 0.9844 | 0.2787 |
| Eigen-CAM | 0.2310 | 0.72 | 0.7637 | 0.9775 | 0.2138 |

**Conclusion.** Four of the five variants are indistinguishable in localisation, mIoU
0.27 plus or minus 0.01 and hit rate 0.70 to 0.74, with Eigen-CAM clearly behind, which
is consistent with its class agnostic formulation: it takes the first principal
component of the activations and uses no gradient, so it cannot be class
discriminative.

Faithfulness separates the field. Grad-CAM achieves both the best deletion and the best
insertion area, meaning its highlighted pixels genuinely drive the decision.
**Grad-CAM is therefore retained to condition the report generation stage.** This
ranking independently replicates the conclusion of Nature Machine Intelligence 2022 [1]
on a different dataset, a different backbone and a different split.

Score-CAM is the counter example that justifies reading both protocols: it localises
best, gains only 1.7 mIoU points over Grad-CAM, loses on faithfulness, and costs about
thirty times more compute. Its cost is not justified here.

## 5. Positioning against the reviewed corpus

1. MDPI Information 2025 [5], ResNet50 on Kermany: accuracy 0.90, AUC 0.93. Our RSNA
   ResNet50 reaches 0.945 and 0.987 on a harder adult dataset with a stricter patient
   level split.
2. Ensemble-CAM 2024 [4]: accuracy 0.93 to 0.98 with an ensemble of three CNNs on
   COVID-19 Radiography. Our single DenseNet121 reaches accuracy 0.952 and macro F1
   0.953 on the same data.
3. Ensemble-CAM 2024 [4] reports IoU 0.17 to 0.34 for pneumonia localisation. Our
   0.23 to 0.29 falls inside that range, on a stricter protocol.
4. Nature Machine Intelligence 2022 [1] reports that the best method stays 24 percent
   of mIoU below radiologists. Our absolute values are of the same order and support
   the same reading.

A direct numerical comparison must stay careful: the datasets, the splits and the
patient populations differ.

## 6. Qualitative findings on explainability

1. Class discriminative Grad-CAM confirms that predicted classes rely on intra
   pulmonary evidence, and that the model ignores the medical devices present in the
   images.
2. Low scoring counterfactual class heatmaps occasionally activate extra anatomical
   regions, which suggests residual acquisition related shortcuts in the multi source
   COVID-19 dataset. This is exactly the audit value of saliency inspection, and it
   echoes pitfall 4 of [04-datasets.md](04-datasets.md) on view position.
3. On RSNA error cases the heatmaps expose two distinct failure modes: a false negative
   where all five methods focus near but not on the expert box, right region and wrong
   decision, and a false positive where the model is distracted by medical devices and
   image borders.
4. The moderate absolute mIoU, about 0.27, reflects a structural ceiling of CAM
   methods, smooth blobs measured against tight rectangular boxes, consistent with the
   human gap reported in [1].

## 7. Result of the report conditioning experiment

Giving the tinted radiograph to the vision language model as an image collapses its
sensitivity, while naming the region in words preserves it. The pipeline therefore
sends the clean image plus a textual statement of the predicted class, its confidence
and the anatomical zone of strongest activation, and the heatmap is shown to the human
only.

This is the design rule implemented in the application, and it is a partial answer to
RQ2: the conditioning helps, but only in the modality the model can use.

## 8. What is not measured yet

BLEU-4, ROUGE-L, METEOR and BERTScore on IU-Xray, RadGraph F1 and CheXbert F1, and the
entity level CHAIR hallucination rate. These are the remaining deliverables of RQ2,
RQ3 and RQ4, tracked in [08-roadmap.md](08-roadmap.md).
