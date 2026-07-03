#!/usr/bin/env python3
"""
compute_kappa_ci.py
===================
Reviewer-response analysis for the FIFA label-audit paper.

Computes, with 95% bootstrap confidence intervals:
  * Fleiss' kappa across the 5 annotators
  * mean pairwise Cohen's kappa (human-human)
  * Cohen's kappa: human-majority label vs. CardiffNLP automatic label
  * (optional) a MODEL-vs-HUMAN-GOLD table: each model's accuracy and
    Cohen's kappa against the human-majority label on the 500-tweet subset.

------------------------------------------------------------------
INPUTS
------------------------------------------------------------------
1) ANNOTATION FILE  (required)  -- your FIFA_annotation_EASY_filled.xlsx
   A spreadsheet with one row per audited tweet and columns:
     - 5 annotator columns holding labels in {negative, neutral, positive}
       (or {0,1,2}); auto-detected by name (annot*, rater*, a1..a5, etc.)
     - one column with the dataset's automatic/CardiffNLP label
       (name containing 'auto', 'cardiff', 'dataset', or 'machine')
   Edit COLUMN HINTS below if auto-detection misses your headers.

2) MODEL PREDICTIONS (optional) -- to build the model-vs-gold table.
   A CSV with one row per audited tweet and one column per model holding
   that model's predicted label on the same 500 tweets, e.g.:
     tweet_id, LR_TFIDF, Conv1D_BiLSTM, BERT, DistilBERT, RoBERTa
   Row order must match the annotation file, or share a 'tweet_id'/'id' key.

------------------------------------------------------------------
USAGE
------------------------------------------------------------------
  pip install pandas numpy openpyxl scikit-learn
  python compute_kappa_ci.py --annotations FIFA_annotation_EASY_filled.xlsx
  python compute_kappa_ci.py --annotations FIFA_annotation_EASY_filled.xlsx \
                             --predictions model_preds_on_500.csv
Outputs printed to console + LaTeX-ready rows written to kappa_ci_out.txt.
"""
import argparse, itertools, re, sys
import numpy as np
import pandas as pd

LABELS = ["negative", "neutral", "positive"]
LABEL_MAP = {"neg":0,"negative":0,"0":0,"-1":0,
             "neu":1,"neutral":1,"1":1,
             "pos":2,"positive":2,"2":2}
RNG = np.random.default_rng(42)
B = 10000  # bootstrap resamples

def norm_label(x):
    s = str(x).strip().lower()
    if s in LABEL_MAP: return LABEL_MAP[s]
    for k,v in LABEL_MAP.items():
        if s.startswith(k): return v
    return np.nan

def cohen_kappa(a, b):
    a = np.asarray(a); b = np.asarray(b)
    n = len(a); cats = [0,1,2]
    po = np.mean(a == b)
    pe = sum((np.mean(a==c))*(np.mean(b==c)) for c in cats)
    return (po - pe)/(1 - pe) if (1-pe) > 0 else np.nan

def fleiss_kappa(mat):
    # mat: n_items x n_categories counts (rows sum to n_raters)
    n, k = mat.shape
    N = mat.sum(axis=1)[0]
    p = mat.sum(axis=0) / (n * N)
    P = (np.sum(mat**2, axis=1) - N) / (N*(N-1))
    Pbar = P.mean(); Pe = np.sum(p**2)
    return (Pbar - Pe)/(1 - Pe) if (1-Pe) > 0 else np.nan

def counts_matrix(ann):  # ann: n_items x n_raters (ints 0..2)
    n = ann.shape[0]; mat = np.zeros((n,3))
    for i in range(n):
        for r in ann[i]:
            if not np.isnan(r): mat[i, int(r)] += 1
    return mat

def majority(ann_row):
    vals = [int(v) for v in ann_row if not np.isnan(v)]
    if not vals: return np.nan
    counts = [vals.count(0), vals.count(1), vals.count(2)]
    m = max(counts)
    # tie-break toward lower polarity: negative < neutral < positive
    for c in [0,1,2]:
        if counts[c] == m: return c

def boot_ci(stat_fn, *arrays, n_boot=B):
    n = len(arrays[0]); idx = np.arange(n); vals = []
    for _ in range(n_boot):
        s = RNG.choice(idx, size=n, replace=True)
        vals.append(stat_fn(*[np.asarray(a)[s] for a in arrays]))
    vals = np.array([v for v in vals if not np.isnan(v)])
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)

def detect_cols(df):
    ann_cols, auto_col = [], None
    for c in df.columns:
        cl = str(c).lower()
        if any(t in cl for t in ["auto","cardiff","dataset","machine"]):
            auto_col = c
        elif re.search(r"(annot|rater|coder|human|a\d|ann\d|label\d)", cl):
            ann_cols.append(c)
    return ann_cols, auto_col

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--predictions", default=None)
    ap.add_argument("--ann-cols", nargs="*", help="override annotator column names")
    ap.add_argument("--auto-col", help="override automatic-label column name")
    args = ap.parse_args()

    df = pd.read_excel(args.annotations) if args.annotations.lower().endswith(("xlsx","xls")) \
         else pd.read_csv(args.annotations)
    ann_cols = args.ann_cols or None
    auto_col = args.auto_col
    if not ann_cols or not auto_col:
        d_ann, d_auto = detect_cols(df)
        ann_cols = ann_cols or d_ann
        auto_col = auto_col or d_auto
    print(f"Annotator columns: {ann_cols}\nAutomatic-label column: {auto_col}\n")
    if len(ann_cols) < 2 or auto_col is None:
        sys.exit("Could not identify columns. Re-run with --ann-cols and --auto-col.")

    ann = np.column_stack([df[c].map(norm_label).values for c in ann_cols])
    auto = df[auto_col].map(norm_label).values
    n = ann.shape[0]

    # human-human
    fk = fleiss_kappa(counts_matrix(ann))
    pairwise = [cohen_kappa(ann[:,i], ann[:,j])
                for i,j in itertools.combinations(range(ann.shape[1]),2)]
    mean_pair = np.mean(pairwise)
    maj = np.array([majority(r) for r in ann])

    # human vs auto
    k_ha = cohen_kappa(maj, auto)
    raw = np.mean(maj == auto)

    # bootstrap CIs
    fk_ci   = boot_ci(lambda A: fleiss_kappa(counts_matrix(A)), ann)
    mp_ci   = boot_ci(lambda A: np.mean([cohen_kappa(A[:,i],A[:,j])
                       for i,j in itertools.combinations(range(A.shape[1]),2)]), ann)
    kha_ci  = boot_ci(cohen_kappa, maj, auto)

    lines = []
    lines.append(f"n = {n} tweets, {ann.shape[1]} annotators, B = {B} bootstrap resamples\n")
    lines.append(f"Fleiss' kappa (5 annotators)        : {fk:.3f}  95% CI [{fk_ci[0]:.3f}, {fk_ci[1]:.3f}]")
    lines.append(f"Mean pairwise Cohen's kappa (h-h)   : {mean_pair:.3f}  95% CI [{mp_ci[0]:.3f}, {mp_ci[1]:.3f}]")
    lines.append(f"  pairwise range                    : {min(pairwise):.3f}-{max(pairwise):.3f}")
    lines.append(f"Human-majority vs. automatic label  : {k_ha:.3f}  95% CI [{kha_ci[0]:.3f}, {kha_ci[1]:.3f}]")
    lines.append(f"  raw agreement                     : {raw*100:.1f}%\n")

    # LaTeX rows for Table (tab:kappa) with CI column
    lines.append("% --- LaTeX rows for Table tab:kappa (add a '95\\% CI' column) ---")
    lines.append(f"Fleiss' $\\kappa$ (5 annotators) & {fk:.3f} & [{fk_ci[0]:.3f}, {fk_ci[1]:.3f}] & Moderate \\\\")
    lines.append(f"Mean pairwise Cohen's $\\kappa$ & {mean_pair:.3f} & [{mp_ci[0]:.3f}, {mp_ci[1]:.3f}] & Moderate \\\\")
    lines.append(f"Human majority vs.\\ auto-label & {k_ha:.3f} & [{kha_ci[0]:.3f}, {kha_ci[1]:.3f}] & Slight \\\\")

    # -------- optional model-vs-gold table --------
    if args.predictions:
        pred = pd.read_csv(args.predictions)
        key = next((c for c in pred.columns if str(c).lower() in ("tweet_id","id","tweetid")), None)
        if key and key in df.columns:
            pred = df[[key]].merge(pred, on=key, how="left")
        model_cols = [c for c in pred.columns if c != key]
        lines.append("\n% --- Model vs. human-gold (500-tweet subset) ---")
        lines.append("Model & Acc. vs human & Cohen's $\\kappa$ vs human \\\\")
        rows = []
        for m in model_cols:
            mp = pred[m].map(norm_label).values
            ok = ~np.isnan(mp) & ~np.isnan(maj)
            acc = np.mean(mp[ok] == maj[ok]); kk = cohen_kappa(maj[ok], mp[ok])
            kci = boot_ci(cohen_kappa, maj[ok], mp[ok])
            rows.append((m, acc, kk, kci))
            lines.append(f"{m} & {acc*100:.1f}\\% & {kk:.3f} [{kci[0]:.3f}, {kci[1]:.3f}] \\\\")
        best = max(rows, key=lambda r: r[2])
        lines.append(f"% Best human-agreement model: {best[0]} (kappa={best[2]:.3f})")

    out = "\n".join(lines)
    print(out)
    with open("kappa_ci_out.txt","w") as f: f.write(out + "\n")
    print("\nWritten to kappa_ci_out.txt")

if __name__ == "__main__":
    main()
