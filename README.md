# Auditing Label Reliability and Data Leakage in Auto-Labelled Social-Media Sentiment Corpora

This repository contains the code, annotation data and results for the paper:

> **Auditing label reliability and data leakage in auto-labelled social-media sentiment corpora: A case study on FIFA World Cup tweets**
> Md. Maruf Bangabashi, Md. Mahabubur Rahman, Tahmid Hasan, Golam Mahmud, Mostofa Kamal Nasir
> Submitted to *PLOS ONE* (under review).

With these files you can reproduce the label-quality audit, the duplicate-leakage test and the leakage-controlled model benchmark reported in the paper.

## Summary of the study

We audit a widely used FIFA World Cup 2022 tweet sentiment dataset from Kaggle. Its labels were assigned automatically by a RoBERTa-based model (`cardiffnlp/twitter-roberta-base-sentiment-latest`), not by people.

- **Label quality (main result).** Five annotators labelled a stratified sample of 500 tweets. They agree with each other moderately (Fleiss' κ = 0.535, 95% CI [0.481, 0.585]). Their majority label agrees with the dataset's automatic label only slightly (Cohen's κ = 0.126, 95% CI [0.073, 0.179]; 43.8% raw agreement).
- **Duplicate leakage.** 1,243 exact duplicate tweets (5.5%) were removed before splitting. Leaving them in raises decision tree and random forest test accuracy by about 1 percentage point.
- **Benchmark.** On the deduplicated corpus (21,259 tweets; 17,007 train / 4,252 test), RoBERTa is the most accurate model (87.16% accuracy, macro-F1 0.872). The best classical model (logistic regression with TF-IDF, 72.20%) does not differ significantly from a Conv1D–BiLSTM model (72.15%; McNemar's p = 0.97).

## Repository contents

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── code/
│   ├── nlp-final.ipynb            # full pipeline: cleaning, deduplication, all 18 models, McNemar tests, figures
│   ├── kappa_analysis.ipynb       # agreement statistics with bootstrap 95% CIs
│   └── compute_kappa_ci.py        # script version of the agreement analysis (single-sheet input, see below)
├── data/
│   ├── FIFA_annotation_EASY_filled.xlsx   # 500 tweets x 5 annotators, one sheet per annotator
│   └── audit_auto_labels.csv              # for the same 500 tweets: dataset auto-label and human-majority label
├── docs/
│   ├── Online_Resource_1_Annotation_Guideline.pdf   # annotation guideline (S1 File of the paper)
│   └── HOWTO_model_vs_human_gold.md                 # optional extension: score models against the human labels
└── results/
    ├── model_comparison.csv       # test accuracy and macro-F1 for all 18 model configurations
    ├── results_summary.json       # all metrics, per-model test predictions, McNemar results, confusion matrices
    └── figs/                      # figures used in the paper
```

### Data files

- **`FIFA_annotation_EASY_filled.xlsx`**: one sheet per annotator (`Annotator_1` to `Annotator_5`), each with the columns `id`, `tweet` and `sentiment` (negative / neutral / positive). Annotators are anonymous.
- **`audit_auto_labels.csv`**: one row per audited tweet, with the columns `id` (matches the spreadsheet), `auto_label` (the dataset's CardiffNLP label) and `human_majority` (majority vote of the five annotators). Seven tweets had a 2–2–1 tie; ties were broken toward the lower class in the order negative < neutral < positive.

## How to reproduce the results

### 1. Set up the environment

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Agreement statistics (CPU only, about 1 minute)

The notebook looks for its input files in its own folder, so copy them into `code/` first:

```bash
cd code
cp ../data/FIFA_annotation_EASY_filled.xlsx .
cp ../data/audit_auto_labels.csv fifa_auto_labels.csv
jupyter nbconvert --to notebook --execute --inplace kappa_analysis.ipynb
```

Expected output:

| Statistic | κ | 95% CI |
|---|---|---|
| Fleiss' κ (5 annotators) | 0.535 | [0.481, 0.585] |
| Mean pairwise Cohen's κ | 0.535 | [0.482, 0.585] |
| Pairwise range | 0.485–0.576 | |
| Human majority vs. auto-label | 0.126 | [0.073, 0.179] |

Confidence intervals use 10,000 bootstrap resamples over tweets with seed 42.

`compute_kappa_ci.py` does the same analysis but expects a single sheet that holds all five annotator columns and an auto-label column. Use the notebook above with the files as provided.

### 3. Full benchmark (GPU recommended)

`code/nlp-final.ipynb` was written for Kaggle notebooks with a single NVIDIA T4 GPU.

1. Create a Kaggle notebook and attach the dataset [FIFA World Cup 2022 Tweets](https://www.kaggle.com/datasets/tirendazacademy/fifa-world-cup-2022-tweets). The notebook finds the CSV under `/kaggle/input` automatically.
2. For the agreement section, also upload `data/FIFA_annotation_EASY_filled.xlsx` and `data/audit_auto_labels.csv`, renaming the latter to `kappa_master.csv`, which is the name the notebook expects.
3. Run all cells. The notebook removes duplicates, makes the stratified 80/20 split, trains the 7 classical classifiers with TF-IDF and count features, the Conv1D–BiLSTM model and the three transformers (BERT, DistilBERT, RoBERTa), runs the McNemar tests, and writes the metrics and figures that are saved in `results/`.

Transformer results can vary slightly between GPU types and library versions.

### 4. Optional: score the models against the human labels

The paper does not include this analysis. `docs/HOWTO_model_vs_human_gold.md` explains how to add a few cells to `nlp-final.ipynb` that save each model's predictions on the 500 audited tweets and compare them with the human-majority labels.

## Settings used in the paper

| Setting | Value |
|---|---|
| Random seed | 42 (splits, vectorisers, model initialisation, bootstrap) |
| Deduplication | 1,243 exact duplicates removed before splitting |
| Train/test split | stratified 80/20 after deduplication (17,007 / 4,252) |
| Classical models | stratified 5-fold CV on the training set; vectorisers fitted on training folds only |
| Features | TF-IDF or counts, 1–3-grams, min. document frequency 10, max. 8,000 terms |
| Transformers | AdamW, LR 2e-5 (BERT, DistilBERT) or 1e-5 (RoBERTa), batch 16, 3 epochs, max. length 128 |
| Environment | Python 3.10, scikit-learn 1.2, TensorFlow 2.x, Hugging Face Transformers, spaCy 3.x |

## Data source and terms of use

All tweets come from the public Kaggle dataset [FIFA World Cup 2022 Tweets](https://www.kaggle.com/datasets/tirendazacademy/fifa-world-cup-2022-tweets) (Tirendaz Academy). Its creators collected the tweets with Snscrape on 20 November 2022 and labelled them with the CardiffNLP model. We did not collect any new data. Tweet content remains subject to the terms of the original Kaggle dataset and of X (Twitter). The annotation files contain no personal information about the annotators, none of whom is an author of the paper.

## Citation

If you use this repository, please cite:

```bibtex
@article{bangabashi2026audit,
  title   = {Auditing label reliability and data leakage in auto-labelled
             social-media sentiment corpora: A case study on {FIFA} {World} {Cup} tweets},
  author  = {Bangabashi, Md. Maruf and Rahman, Md. Mahabubur and Hasan, Tahmid
             and Mahmud, Golam and Nasir, Mostofa Kamal},
  journal = {PLOS ONE},
  note    = {Under review},
  year    = {2026}
}
```

## License

The code is released under the MIT License (see `LICENSE`). Tweet text is not covered by this license and remains subject to the original dataset's terms and those of X (Twitter).

## Contact

Md. Maruf Bangabashi, marufbangabashi@gmail.com
