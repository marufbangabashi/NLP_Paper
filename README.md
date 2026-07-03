# Auditing Label Reliability and Data Leakage in Auto-Labelled Social-Media Sentiment Corpora

Reproducibility package for the paper:

> **Auditing Label Reliability and Data Leakage in Auto-Labelled Social-Media Sentiment Corpora: A Case Study on FIFA World Cup Tweets**
> Md. Maruf Bangabashi, Md. Mahabubur Rahman, Md. Naeem Ahmed Talukder, Mostofa Kamal Nasir.
> Submitted to *Progress in Artificial Intelligence* (Springer).

This repository contains everything needed to reproduce the label-quality audit,
the leakage demonstration, and the leakage-controlled benchmark reported in the paper.

## What this study does

Rather than another accuracy-focused benchmark, we audit a widely reused FIFA World
Cup 2022 tweet sentiment dataset whose labels were generated automatically by a
RoBERTa-based classifier (CardiffNLP). Key findings:

- **Label quality (primary):** five independent annotators agree with each other
  moderately (Fleiss' κ = 0.535, 95% CI [0.481, 0.585]) but agree with the dataset's
  automatic labels only slightly (Cohen's κ = 0.126, 95% CI [0.074, 0.179]).
- **Leakage (supporting):** exact-duplicate tweets modestly inflate tree-based
  classifier accuracy (~1 pp) under a controlled, duplicate-aware pipeline.
- **Corrected benchmark:** on the deduplicated corpus (21,259 tweets), RoBERTa is best
  (87.16% accuracy, macro-F1 0.872); the best classical model (LR+TF-IDF, 72.20%) is
  statistically indistinguishable from a Conv1D–BiLSTM (McNemar's p = 0.97).

## Repository layout

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── code/
│   ├── nlp-final.ipynb            # full pipeline: preprocessing, ML/DL/transformers, McNemar, figures
│   ├── kappa_analysis.ipynb       # inter-annotator agreement + bootstrap 95% CIs (runs on the xlsx alone)
│   └── compute_kappa_ci.py        # same analysis as a script; also builds the model-vs-human-gold table
├── data/
│   └── FIFA_annotation_EASY_filled.xlsx   # 5 annotators × 500 tweets (de-identified)
├── docs/
│   ├── Online_Resource_1_Annotation_Guideline.pdf
│   └── HOWTO_model_vs_human_gold.md
└── results/
    ├── model_comparison.csv       # accuracy / macro-F1 for all 18 model configurations
    ├── results_summary.json       # full metrics, per-model test predictions, McNemar table
    └── figs/                      # all figures used in the paper
```

## Reproducing the results

**1. Environment**
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**2. Inter-annotator agreement + confidence intervals** (fast, CPU-only)
```bash
cd code
jupyter nbconvert --to notebook --execute kappa_analysis.ipynb
# or: python compute_kappa_ci.py --annotations ../data/FIFA_annotation_EASY_filled.xlsx
```
Reproduces Fleiss' κ, mean pairwise Cohen's κ, and human-vs-auto κ, each with a
10,000-sample bootstrap 95% CI, and prints LaTeX-ready table rows.

**3. Full benchmark** (GPU recommended for the transformers)

Run `code/nlp-final.ipynb` top to bottom. It downloads the public dataset labels,
applies the documented preprocessing, trains the seven classical models, the
Conv1D–BiLSTM, and the three transformers, and writes metrics + figures to `results/`.

**4. (Optional) Model-vs-human-gold table**

Follow `docs/HOWTO_model_vs_human_gold.md` to add four small cells to `nlp-final.ipynb`;
one run exports `model_preds_on_500.csv`, then `compute_kappa_ci.py --predictions` prints
each model's accuracy and Cohen's κ against the human-majority label.

## Reproducibility notes

- Global random seed: **42** (splits, vectorisers, model init).
- Train/test split: stratified 80/20 applied **after** removing 1,243 exact duplicates.
- Vectorisers are fit on training folds only (no test-set fitting).
- Environment: Python 3.10, scikit-learn 1.2, TensorFlow 2.x, Hugging Face Transformers,
  spaCy 3.x; transformers trained on a single NVIDIA T4 GPU (Kaggle).

## Data

The underlying tweets are from the public Kaggle dataset
[FIFA World Cup 2022 Tweets](https://www.kaggle.com/datasets/tirendazacademy/fifa-world-cup-2022-tweets).
`data/FIFA_annotation_EASY_filled.xlsx` contains only tweet text and the five annotators'
sentiment labels — no annotator identities.

## Citation

```bibtex
@article{bangabashi2026audit,
  title   = {Auditing Label Reliability and Data Leakage in Auto-Labelled
             Social-Media Sentiment Corpora: A Case Study on FIFA World Cup Tweets},
  author  = {Bangabashi, Md. Maruf and Rahman, Md. Mahabubur and
             Talukder, Md. Naeem Ahmed and Nasir, Mostofa Kamal},
  journal = {Progress in Artificial Intelligence (under review)},
  year    = {2026}
}
```

## License

Code is released under the MIT License (see `LICENSE`). The tweet content is subject to
the terms of the original Kaggle dataset and X/Twitter.
