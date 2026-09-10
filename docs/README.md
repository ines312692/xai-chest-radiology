# Documentation

Reference material of the project *Multimodal Explainable AI for Chest Radiology*,
Master 2 ISI. The repository root README is the short front page. This folder holds the
detail: the research background, the datasets, the frozen protocol, the results and the
architecture of the code.

## Contents

| File | What it answers |
|---|---|
| [01-project-scope.md](01-project-scope.md) | What the project studies, its four research questions, what is delivered |
| [02-state-of-the-art.md](02-state-of-the-art.md) | The thirteen reviewed papers, grouped by axis, and what each one contributes to our protocol |
| [03-references.md](03-references.md) | Every reference with its link, numbered for citation |
| [references.bib](references.bib) | The same references as BibTeX, for the thesis |
| [04-datasets.md](04-datasets.md) | The datasets used, why they were selected, and the verified pitfalls of each |
| [05-experimental-protocol.md](05-experimental-protocol.md) | Every experimental choice and the source that justifies it |
| [06-results.md](06-results.md) | All measured results, including the answer to research question one |
| [07-system-architecture.md](07-system-architecture.md) | The code package, the demonstration application and the remote report server |
| [08-roadmap.md](08-roadmap.md) | What is done, what is running, what remains |
| [09-reproducibility.md](09-reproducibility.md) | How to re run everything, on Kaggle and locally |

## Conventions used in these documents

A claim about the literature carries the reference that supports it. A number reported
here comes from a file under `results/`, never from memory. A statement that could not
be verified against a primary source is marked as unverified rather than dropped, since
knowing what remains unchecked is part of the research record.

## Source material

The raw review lives in `../sota/`. `SOTA_Table_XAI_Multimodal_Radiologie.xlsx` holds
the thirteen papers with twenty four columns each, from preprocessing to future work.
`Choix_base_de_donnees_CXR.xlsx` holds the dataset selection study, its weighted grid
and its list of verified pitfalls. The documents in this folder are the readable
synthesis of those two files, not a replacement for them.
