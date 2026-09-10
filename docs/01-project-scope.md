# Project scope and research questions

## The problem

Deep learning models applied to medical imaging reach expert level performance on many
diagnostic tasks and remain largely opaque. That opacity is a barrier to clinical
adoption, to regulatory approval and to physician trust. Two families of answers exist
and are usually studied separately: visual explanations, which show where a model
looked, and textual explanations, which say what it saw.

Neither is sufficient alone. A heatmap tells a radiologist where, not what, and the
PLOS ONE 2024 user study found that fifteen of twenty six clinicians asked for textual
explanations to accompany the map [3]. A generated report says what without showing
where, and can state findings the image does not support: the MedVH benchmark measures
that even the best evaluated model hallucinates close to one mentioned pathology out of
two [12].

## What this project does

It designs, implements and evaluates a complete pipeline that produces both, and it
measures whether they agree with the image, with the model and with each other.

1. **Visual explanation.** A convolutional classifier is fine tuned on chest
   radiographs, and five class activation mapping variants are produced and scored
   against expert bounding boxes and against the behaviour of the model itself.
2. **Textual explanation.** A vision language model writes the findings section of a
   report, conditioned on the image and on the facts extracted from the visual stage.
3. **Faithfulness analysis.** Both outputs are measured rather than shown: localisation
   against expert annotation, faithfulness by perturbation, and hallucination at the
   level of clinical entities.

## Research questions

| Id | Question | Status |
|---|---|---|
| RQ1 | Which CAM variant produces the most clinically relevant and quantitatively superior attention maps on chest X-rays? | **Answered.** Grad-CAM retained, see [06-results.md](06-results.md) |
| RQ2 | Does conditioning the vision language model on the visual attention map produce more faithful and accurate reports than using the image alone? | In progress |
| RQ3 | What is the hallucination rate of the generated reports, and which metrics detect them most reliably in radiology? | In progress |
| RQ4 | How does the full pipeline compare with state of the art report generation systems on BLEU-4, ROUGE-L and BERTScore? | In progress |

RQ2 has already produced one result that shaped the system: giving the tinted
radiograph to the model as an image collapses its sensitivity, while naming the region
in words preserves it. The application therefore conditions the report on the clean
image plus a textual statement of the predicted class, its confidence and the
anatomical zone of strongest activation, and the heatmap is shown to the human only.

## Deliverables

1. A reproducible classification and explanation pipeline, delivered as Kaggle
   notebooks and as the `src/` package.
2. A quantitative comparison of five saliency methods on two protocols, which answers
   RQ1.
3. A report generation stage evaluated on natural language metrics and on factuality.
4. A demonstration application that runs the whole chain on an uploaded radiograph.
5. The thesis and its defence.

## Boundaries

The system reports pneumonia only and does not exclude other pathology. It is a
research prototype, not a medical device, and every output requires review by a
qualified radiologist. This wording appears in the application itself, on every report
it produces, whichever component wrote the text.
