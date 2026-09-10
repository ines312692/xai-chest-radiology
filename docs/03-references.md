# References

Numbering used across the documentation. Entries 1 to 13 are the reviewed corpus, held
in full in `../sota/SOTA_Table_XAI_Multimodal_Radiologie.xlsx`. Entries 14 and beyond
are the method and resource papers the implementation depends on.

BibTeX for all of them: [references.bib](references.bib).

## Reviewed corpus

| # | Reference | Venue | Rank | Link |
|---|---|---|---|---|
| 1 | Benchmarking saliency methods for chest X-ray interpretation | Nature Machine Intelligence, 2022 | Q1 | https://www.nature.com/articles/s42256-022-00536-x |
| 2 | Comparative evaluation of CAM methods for enhancing explainability in veterinary radiography | Scientific Reports, 2025 | Q1 | https://www.nature.com/articles/s41598-025-14060-6 |
| 3 | Evaluating Explainable Artificial Intelligence techniques in chest radiology imaging through a human-centered lens | PLOS ONE, 2024 | Q1 | https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0308758 |
| 4 | Toward explainable AI in radiology: Ensemble-CAM for effective thoracic disease localization in chest X-ray images using weak supervised learning | Frontiers in Big Data, 2024 | Q2 | https://www.frontiersin.org/journals/big-data/articles/10.3389/fdata.2024.1366415/full |
| 5 | Interpretable Deep Learning for Pneumonia Detection Using Chest X-Ray Images | Information (MDPI), 2025 | Q2 | https://www.mdpi.com/2078-2489/16/1/53 |
| 6 | Advancing AI Interpretability in Medical Imaging: A Comparative Analysis of Pixel-Level Interpretability and Grad-CAM Models | Machine Learning and Knowledge Extraction, 2025 | Q1 | https://www.mdpi.com/2504-4990/7/1/12 |
| 7 | A clinically accessible small multimodal radiology model and evaluation metric for chest X-ray findings (LLaVA-Rad, CheXprompt) | Nature Communications, 2025 | Q1 | https://www.nature.com/articles/s41467-025-58344-x |
| 8 | Collaboration between clinicians and vision-language models in radiology report generation (Flamingo-CXR) | Nature Medicine, 2025 | Q1 | https://www.nature.com/articles/s41591-024-03302-1 |
| 9 | Vision-Language Model for Generating Textual Descriptions From Clinical Images (ClinicalBLIP) | JMIR, 2024 | Q1 | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10884898/ |
| 10 | A vision attention driven language framework for medical report generation (MedVAG) | Scientific Reports, 2025 | Q1 | https://www.nature.com/articles/s41598-025-95666-8 |
| 11 | ChestX-Transcribe: a multimodal transformer for automated radiology report generation from chest x-rays | Frontiers in Digital Health, 2025 | Q1 | https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2025.1535168/full |
| 12 | MedVH: Toward Systematic Evaluation of Hallucination for Large Vision Language Models in the Medical Context | Advanced Intelligent Systems, 2025 | Q1 | https://advanced.onlinelibrary.wiley.com/doi/10.1002/aisy.202500255 |
| 13 | Comparative evaluation of generative AI models for chest radiograph report generation in the emergency department | European Radiology, 2026 | Q1 | https://link.springer.com/article/10.1007/s00330-026-12648-8 |

## Saliency methods implemented

| # | Reference | Venue |
|---|---|---|
| 14 | Selvaraju et al. Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization | ICCV 2017, IJCV 2020 |
| 15 | Chattopadhay et al. Grad-CAM++: Generalized Gradient-based Visual Explanations for Deep Convolutional Networks | WACV 2018 |
| 16 | Jiang et al. LayerCAM: Exploring Hierarchical Class Activation Maps for Localization | IEEE TIP 2021 |
| 17 | Muhammad and Yeasin. Eigen-CAM: Class Activation Map using Principal Components | IJCNN 2020 |
| 18 | Wang et al. Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks | CVPR Workshops 2020 |
| 19 | Petsiuk et al. RISE: Randomized Input Sampling for Explanation of Black-box Models | BMVC 2018 |

Reference 19 defines the deletion and insertion curves used as the faithfulness
protocol. References 14 to 18 are the five methods served by the pipeline and by the
application, all through the `grad-cam` library.

## Backbones and models

| # | Reference | Venue |
|---|---|---|
| 20 | He et al. Deep Residual Learning for Image Recognition (ResNet) | CVPR 2016 |
| 21 | Huang et al. Densely Connected Convolutional Networks (DenseNet) | CVPR 2017 |
| 22 | Rajpurkar et al. CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning | arXiv 2017 |
| 23 | Google. MedGemma, a medical vision language model | Model card, 2025, https://huggingface.co/google/medgemma-4b-it |

## Datasets

| # | Reference | Venue |
|---|---|---|
| 24 | RSNA Pneumonia Detection Challenge | Kaggle, 2018, https://www.kaggle.com/c/rsna-pneumonia-detection-challenge |
| 25 | Kermany et al. Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning | Cell 2018 |
| 26 | Wang et al. ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks | CVPR 2017 |
| 27 | Irvin et al. CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison | AAAI 2019 |
| 28 | Demner-Fushman et al. Preparing a collection of radiology examinations for distribution and retrieval (Indiana University, Open-i) | JAMIA 2016 |
| 29 | Chowdhury et al. Can AI Help in Screening Viral and COVID-19 Pneumonia? (COVID-19 Radiography Database) | IEEE Access 2020 |
| 30 | Rahman et al. Exploring the Effect of Image Enhancement Techniques on COVID-19 Detection Using Chest X-ray Images | Computers in Biology and Medicine 2021 |

## Note on completeness

Author lists for entries 1 to 13 are not stored in the review workbook, which indexes
papers by title, venue and link. In `references.bib` those entries carry a `TODO`
author field to be completed from the paper page before submission. Every other field,
title, venue, year and link, was taken from the workbook and is verified.
