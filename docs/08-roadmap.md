# Roadmap and status

## Milestones

| Milestone | Status |
|---|---|
| Systematic literature review, thirteen papers, twenty four dimensions each | Done |
| Dataset selection study, weighted grid, verified pitfalls | Done |
| Experimental configuration frozen, datasets and metrics | Done |
| Preprocessing with frozen patient level splits | Done |
| Four backbone comparison on Kermany | Done |
| RSNA binary pipelines, ResNet50 and DenseNet121, identical split, Grad-CAM | Done |
| COVID-19 multi class pipelines, both backbones, class discriminative Grad-CAM | Done |
| **RQ1. Five saliency variants scored on two protocols. Grad-CAM retained** | **Done** |
| Code restructured as the `src/` package, tested | Done |
| Demonstration application, three stage chain, method and backbone selectors | Done |
| MedGemma report generation wired through the remote GPU server | Done |
| RQ2, RQ4. Report generation on IU-Xray, image alone versus image plus conditioning, BLEU-4, ROUGE-L, METEOR, BERTScore | In progress |
| RQ3. Hallucination quantification, entity level CHAIR rate on 30 to 50 reports per condition | In progress |
| Thesis writing and defence presentation | In progress |

## Next steps, in order

1. **Lung mask for the anatomical zone.** The current thirds grid names a shoulder as a
   lung zone and that name reaches the language model. A coarse mask fixes the label,
   and the rate of peaks falling outside the mask becomes a reportable indicator for
   RQ1.
2. **The report generation campaign on IU-Xray.** Two conditions, image alone and image
   plus textual conditioning, on the same studies, scored with BLEU-4, ROUGE-L, METEOR
   and BERTScore. The comparison targets are ClinicalBLIP ROUGE-L 0.534,
   ChestX-Transcribe BLEU-4 0.472 and MedVAG BLEU-4 0.595, all on IU-Xray but with
   different splits, so ours must be documented explicitly.
3. **Factuality and hallucination.** RadGraph F1 and CheXbert F1, plus an entity level
   CHAIR rate following MedVH, whose reference value for LLaVA-Med is 0.737.
4. **Thesis figures.** The evaluation figure, the saliency comparison grid and the
   metric barplots are already produced by `scripts/benchmark_cams.py` and by
   `xai_chest/reporting/figures.py`, regenerable from the saved CSV files.

## Risks and how they are handled

| Risk | Handling |
|---|---|
| The Kaggle session holding MedGemma stops during the defence | The application falls back to the structured writer and states it. Restarting the notebook and pasting the new tunnel address takes about a minute |
| IU-Xray splits differ between published papers, making BLEU-4 incomparable | Our split is frozen to CSV and documented; the comparison is presented as indicative rather than as a ranking |
| Kaggle weekly GPU quota | Early stopping on the validation AUC, and `--skip-faithfulness` on the saliency benchmark when the budget is short |
| Manual hallucination annotation is slow | The protocol is fixed at 30 to 50 reports per condition, which is the sample size MedVH uses at comparable scale |
