# State of the art

Thirteen papers were reviewed, each characterised on twenty four dimensions in
`../sota/SOTA_Table_XAI_Multimodal_Radiologie.xlsx`, from preprocessing and augmentation
to loss, evaluation protocol, reproducibility, stated limitations and future work. This
document is the readable synthesis, organised by the three axes of the project. Numbers
in brackets refer to [03-references.md](03-references.md).

Papers are cited here for what they contribute to our protocol, and for the weakness
that our protocol corrects. A published number is quoted only when it is a reference
point we can be compared against.

---

## Axis 1. Saliency methods and their evaluation

### The benchmark that defines the protocol

**Benchmarking saliency methods for chest X-ray interpretation**, Nature Machine
Intelligence 2022 [1]. Seven methods on CheXpert and CheXlocalize, with DenseNet121 as
reference backbone. Grad-CAM localises best of the seven, and every method stays far
below the human benchmark: minus 24.0 percent mIoU and minus 29.4 percent hit rate on
average, with the widest gap on lung lesion at minus 76.2 percent. Localisation
degrades on small and complex shaped pathologies, and model confidence correlates only
weakly with localisation quality, Spearman 0.285.

What we take from it: the whole localisation protocol, mIoU as primary metric and hit
rate as secondary, read jointly, plus the reference point that even the best method is
far from a radiologist. Its own stated limitation, that a poor score cannot be
attributed to the model or to the method, is one reason we add a second protocol that
does not depend on annotation.

### The methods compared

**Comparative evaluation of CAM methods for enhancing explainability in veterinary
radiography**, Scientific Reports 2025 [2]. Eleven CAM methods on canine radiographs,
scored by human raters. The result is negative and useful: no method improves
diagnostic confidence, all means fall between two and three out of five, and two
ranking procedures disagree about which method wins. Inter annotator agreement is weak,
Kendall tau between 0.12 and 0.38.

What we take from it: the choice to rank methods on measurable criteria rather than on
preference, and the 70 10 20 split proportions.

**Ensemble-CAM**, Frontiers in Big Data 2024 [4]. Grad-CAM++ over an ensemble of three
CNNs on RSNA and ChestX-ray14. Classification is strong, 0.93 to 0.98 accuracy, and
localisation is weak, IoU 0.17 to 0.34 on pneumonia. The instructive part is that
DenseNet121 alone reaches 0.53 to 0.70, so the ensemble is worse than its best member.

What we take from it: RSNA as a saliency evaluation dataset, the exclusion rule for the
ambiguous class, and a caution against assuming that combining models improves
explanations.

**Interpretable Deep Learning for Pneumonia Detection**, Information MDPI 2025 [5].
ResNet50 on the Kermany pediatric dataset with four interpretability techniques.
Baseline accuracy 0.90 and AUC 0.93, which is the number our own ResNet50 runs are
positioned against. The authors note that Grad-CAM is limited by its dependence on the
last convolutional layer.

What we take from it: the ResNet50 baseline for comparison, and the exact augmentation
parameters, rotation fifteen degrees, horizontal flip, zoom twenty percent.

**Advancing AI Interpretability in Medical Imaging**, Machine Learning and Knowledge
Extraction 2025 [6]. Compares post hoc Grad-CAM with an intrinsic pixel level method on
a COVID-19 dataset, reporting IoU 0.74 against 0.57 in favour of the intrinsic method.

What we take from it: a reminder that post hoc methods are one family among others, and
that the ceiling we observe is partly structural.

### The human side

**Evaluating XAI techniques in chest radiology imaging through a human centered lens**,
PLOS ONE 2024 [3]. Twenty six clinicians compare Grad-CAM and LIME. Grad-CAM is
preferred nineteen to six, thirteen of twenty six report readability problems with the
colour scheme, and fifteen of twenty six ask for textual explanations in addition to
the map.

What we take from it: the metric set for classification, the evidence that a visual
explanation alone is not what clinicians want, which is the premise of this project,
and the limitation that the paper acknowledges about its own missing patient level
split, which our protocol corrects.

---

## Axis 2. Report generation with vision language models

**LLaVA-Rad and CheXprompt**, Nature Communications 2025 [7]. A small multimodal model
trained on 697K pairs, beating much larger models, together with an automatic metric
that agrees with radiologists better than ROUGE-L or RadGraph F1, Kendall tau-b above
0.75 against below 0.57. The sobering number is that only 2.58 percent of generated
reports are fully error free on MIMIC-CXR, and the most frequent error is a clinically
significant false negative.

**Flamingo-CXR**, Nature Medicine 2025 [8]. Report generation evaluated by a panel of
radiologists on MIMIC-CXR and on a private Indian dataset. Clinically significant
errors appear in 22.8 percent of AI reports against 14.0 percent of human reports, and
the panel is unanimous in only 27.4 percent of cases, which says as much about the
evaluation as about the model.

**ClinicalBLIP**, JMIR 2024 [9]. InstructBLIP with LoRA on IU-Xray, METEOR 0.570 and
ROUGE-L 0.534. Its ablation is the interesting part: removing prior information drops
METEOR from 0.570 to 0.339.

**MedVAG**, Scientific Reports 2025 [10]. ViT plus DenseNet121 tags plus GPT-2 on
IU-Xray, BLEU-4 0.595 and ROUGE-L 0.762. Grad-CAM is used qualitatively and is guided
by tags, but it is never evaluated and never fed to the model.

**ChestX-Transcribe**, Frontiers in Digital Health 2025 [11]. Swin Transformer on
IU-Xray, BLEU-4 0.472 and ROUGE-L 0.698.

What we take from this axis: IU-Xray as the comparison dataset, the metric set BLEU-4,
ROUGE-L, METEOR and BERTScore, and the reference values above as the targets of RQ4.
Two observations shape our own design. First, the published numbers are not directly
comparable across papers because the splits differ, so ours must be documented
explicitly. Second, none of these systems feeds a saliency map to the language model,
which is precisely the gap RQ2 addresses.

---

## Axis 3. Hallucination and factuality

**MedVH**, Advanced Intelligent Systems 2025 [12]. A systematic hallucination benchmark
for medical vision language models. The headline finding is counterintuitive and
central to this project: medically fine tuned models hallucinate more than general
models despite better performance on standard tasks. Instance level CHAIR on report
generation ranges from 0.461 for the best model to 0.938 for the worst, with LLaVA-Med
at 0.737.

What we take from it: the CHAIR protocol at entity level for RQ3, and the reference
values to position our own rate.

**Comparative evaluation of generative AI models for chest radiograph report generation
in the emergency department**, European Radiology 2026 [13]. Seven systems evaluated by
radiologists on 478 emergency patients. One commercial system beats radiologists on
RADPEER and acceptability. The others show hallucination rates from 5.4 to 17.4 percent,
no model detects miliary nodules or pneumoperitoneum, and language clarity is rated
lowest for MedGemma, which is the model our application uses.

What we take from it: a realistic expectation of what an open model produces in a
clinical setting, and a reason to report hallucination rates rather than only overlap
metrics.

---

## What the review establishes for this project

1. **Grad-CAM is the reference, and it is not good enough.** Every benchmark that ranks
   methods puts Grad-CAM at or near the top, and every one of them reports a large gap
   to human localisation. Reproducing that ranking on a new dataset is a contribution;
   assuming it would have been a mistake.
2. **Localisation and faithfulness are different questions.** A method can agree with a
   radiologist and not reflect what the network used. Nature Machine Intelligence 2022
   recommends reading two metrics jointly, and we add a second protocol that needs no
   annotation for the same reason.
3. **Clinicians want text with the map.** This is measured, not assumed, in PLOS ONE
   2024, and it is the premise of the multimodal pipeline.
4. **Generated reports are fluent and often wrong.** Across three independent
   evaluations the error rate is high and the most frequent error is a missed finding,
   which is the dangerous direction in radiology.
5. **No published system conditions the report on the saliency map.** That is the gap
   this project occupies, and our own experiments show that how the conditioning is
   done matters more than whether it is done.
