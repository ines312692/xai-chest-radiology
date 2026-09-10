# Datasets

Images and model weights are never stored in this repository. Every notebook runs on
Kaggle with the datasets attached as inputs, and resolves its mount path by file search
rather than by a hardcoded path, so it survives a change of mount layout.

## What is used, and for what

| Dataset | Role in the project | Size | Why it was selected | Link |
|---|---|---|---|---|
| RSNA Pneumonia Detection Challenge | Main binary task, adult patients, and the saliency evaluation | 26,684 DICOM images | In the reviewed corpus through Ensemble-CAM [4]. The only selected dataset that carries both a patient identifier, which allows a leak free split, and expert bounding boxes, which allow mIoU and hit rate to be computed | https://www.kaggle.com/c/rsna-pneumonia-detection-challenge |
| COVID-19 Radiography Database | Multi class task, four classes, class discriminative Grad-CAM | 21,165 PNG images | In the reviewed corpus through Ensemble-CAM [4]. Official dataset of Chowdhury et al. [29] and Rahman et al. [30]. Multi source, which makes it useful for detecting acquisition shortcuts | https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database |
| Chest X-Ray Images (Pneumonia), Kermany | Comparison point with the literature, pediatric | 5,863 JPEG images | Benchmark of PLOS ONE 2024 [3] and MDPI Information 2025 [5], which allows a direct numerical comparison with published accuracy 0.90 and AUC 0.93 | https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia |
| IU-Xray, Indiana University Open-i | Report generation, RQ2 and RQ4 | 7,470 images, 3,955 reports | The benchmark on which ClinicalBLIP [9], MedVAG [10] and ChestX-Transcribe [11] publish, therefore the only place where our BLEU-4 and ROUGE-L are comparable | https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university |

## The selection study

`../sota/Choix_base_de_donnees_CXR.xlsx` holds the full comparison, a weighted
selection grid and a decision sheet. Its founding observation is worth keeping in the
thesis as written:

> No public dataset simultaneously offers a sufficient training volume, pixel level
> localisation ground truth and paired reports. Looking for the right dataset is
> therefore a dead end. Four are assembled instead, each for what it does well.

The workbook explores a multi source configuration built on NIH ChestX-ray14 then
CheXpert for training, CheXlocalize for pixel level localisation, CheXmask for the
anatomical constraint and IU-Xray for reports. The configuration actually implemented
is centred on RSNA instead, because RSNA carries patient identifiers and expert boxes
in a single dataset that is already mounted on Kaggle, which removes both the access
delay of Stanford AIMI and the label noise of the NIH text extractor. The workbook
analysis stays valid and is the fallback if the localisation evaluation is later
extended beyond pneumonia, since CheXlocalize is the only resource that offers a human
benchmark to compare against.

MIMIC-CXR and PadChest were explicitly ruled out and the reason should appear in the
thesis: 4.7 TB with a long credentialing process for the first, 1 TB with Spanish
reports and no localisation for the second, both incompatible with a Kaggle workspace
and a six month schedule. Both are cited in the state of the art, neither is used.

## Verified pitfalls

Collected on 24 August 2026 from the official pages and the original papers, and
recorded here because each one silently corrupts a result rather than raising an error.

| # | Concerns | The trap | Consequence if ignored |
|---|---|---|---|
| 1 | NIH BBox_List_2017 | The CVPR 2017 paper announces 983 images and about 1,600 boxes; the distributed file holds 984 boxes over 880 unique images, very unevenly spread | A wrong figure in the thesis, inherited from the paper rather than from the data |
| 2 | NIH class names | The box file says Infiltrate, the label file says Infiltration | The join between the two files silently loses 123 boxes |
| 3 | NIH RGBA images | Some PNG files carry four channels instead of grayscale | The data loader crashes mid epoch. Remedy: always `convert('L')` |
| 4 | NIH view position | AP views are bedside acquisitions on sicker patients, and the pathology rate differs strongly between AP and PA | The model learns the patient position rather than the pathology, and the attention map localises that confounder |
| 5 | NIH ages | The age column contains values up to 411 years | A sign that the metadata is machine generated. Do not use age as a variable |
| 6 | Every dataset with patients | A NIH patient has 3.6 images on average and sometimes more than a hundred | An image level split puts the same patient in train and test, the AUROC rises and the result is worthless |
| 7 | CheXpert uncertain labels | Labels have three states including uncertain, coded minus one | Treating minus one as zero without saying so distorts the comparison. It is an ablation, U-Ones, U-Zeros or U-Ignore |
| 8 | CheXlocalize RLE masks | Masks are COCO run length encoded, column major | A transposed decoding gives plausible masks rotated by ninety degrees, and every mIoU becomes wrong with no visible error |
| 9 | CheXmask quality | Masks are generated automatically and the authors recommend filtering on mean RCA-DSC above 0.7 | Raw masks introduce a false anatomical constraint on part of the images |
| 10 | IU-Xray split | Published papers use different splits, 7:1:2 or the official one | Comparing a BLEU-4 across two different splits means nothing. Document yours |
| 11 | Shenzhen masks | The 566 widely used lung masks come from Stirenko et al. 2018, not from the NLM | Citing the NLM as the source of the masks is an attribution error |

## Points that could not be verified

To be quoted only with an explicit caveat, or re verified before submission.

1. CheXpert sizes, 11 GB for the small version and 439 GB for the full one, come from
   third party sources; no official Stanford page was readable.
2. The exact resolution of CheXpert-v1.0-small, often quoted as about 320 pixels on the
   short side, is unconfirmed.
3. The exact licence terms of CheXlocalize, the Redivis page was not readable.
4. The explicit licence and native resolution of IU-Xray, no formal licence was found
   on Open-i.
5. The total number of boxes in MS-CXR; only the figure of 1,162 image sentence pairs
   is confirmed.
6. MIMIC-CXR sizes, 4.7 TB and 570 GB, are common usage and unconfirmed on PhysioNet.
7. The licence of RSNA: the frequently quoted CC BY-NC-SA is contradicted by the
   official RSNA terms.
8. The exact licence text on the SIIM-ACR Kaggle page, which is rendered in JavaScript.

Method: a fact is recorded as verified only when it was read on an official page, NIH
Box, Stanford AIMI, PhysioNet, Open-i, NLM, JSRT, or in the original article. A third
party source is not accepted as verification.
