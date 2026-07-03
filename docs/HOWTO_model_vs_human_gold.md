# How to produce the model-vs-human-gold table (one notebook run)

**Why this isn't already done:** in `nlp-final.ipynb`, the transformers are trained
in a loop and then deleted (`del mdl`, `save_strategy="no"`), and no model ever
predicts on the 500 audited tweets (cell 20 is only a placeholder note). So the
table can't be reconstructed from saved files — the models must predict on the 500
tweets *during* a run. The cells below do exactly that, using your existing
variable names (`df`/`train_df`/`test_df`, `Xtr_text`, `ytr`, `make_vec`, `tok`,
`MAXLEN`, `model`, `DS`, `SEED`). Add them, run once, then run `compute_kappa_ci.py`
or the last cell of `kappa_analysis.ipynb`.

---

### Cell A — build the 500-tweet audit frame
*Add after `test_df` is created, before the transformer loop (cell 16).*

```python
# build the 500-tweet audit frame + export per-tweet auto labels
import pandas as pd, numpy as np
_ann  = pd.read_excel("FIFA_annotation_EASY_filled.xlsx", sheet_name="Annotator_1").sort_values("id")
_corp = pd.concat([train_df, test_df], ignore_index=True)
aud = (_ann[["id","tweet"]]
       .merge(_corp[["text_raw","text_clean","text_light","y"]].drop_duplicates("text_raw"),
              left_on="tweet", right_on="text_raw", how="left"))
print("audit tweets matched to corpus:", int(aud["text_light"].notna().sum()), "/", len(aud))
AUD_PREDS = {"tweet_id": aud["id"].values}
# bonus: real per-tweet auto labels for the kappa notebook
aud.assign(auto_label=aud["y"].map({0:"negative",1:"neutral",2:"positive"}))[["id","auto_label"]] \
   .to_csv("fifa_auto_labels.csv", index=False)
```

If the "matched" count is well below 500, the annotation `tweet` text differs
slightly from `text_raw`; tell me and I'll switch the match to a normalised key.

---

### Cell B — classical + Conv1D-BiLSTM predictions on the 500
*Add after the BiLSTM cell (cell 15), so `model`, `tok`, `MAXLEN` exist.*

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
_pipe = Pipeline([("vec", make_vec("tfidf")),
                  ("clf", LogisticRegression(max_iter=1000, C=1.0))]).fit(Xtr_text, ytr)
AUD_PREDS["LR_TFIDF"] = _pipe.predict(aud["text_clean"].fillna(""))

_seq = pad_sequences(tok.texts_to_sequences(aud["text_clean"].fillna("")), maxlen=MAXLEN)
AUD_PREDS["Conv1D_BiLSTM"] = model.predict(_seq).argmax(1)
```

---

### Cell C — one line inside the transformer loop (cell 16)
*Add immediately after `yp = logits.argmax(-1)` and before `del mdl`:*

```python
    # predict on the 500 audit tweets before this model is deleted
    AUD_PREDS[name] = tr.predict(DS(aud["text_light"].fillna("").tolist(),
                                    [0]*len(aud), tk)).predictions.argmax(-1)
```

---

### Cell D — save the predictions
*Add after the transformer loop finishes.*

```python
import pandas as pd
pd.DataFrame(AUD_PREDS).to_csv("model_preds_on_500.csv", index=False)
print("saved model_preds_on_500.csv", pd.DataFrame(AUD_PREDS).shape)
```

---

### Finish
Run either:

```bash
python compute_kappa_ci.py --annotations FIFA_annotation_EASY_filled.xlsx \
                           --predictions model_preds_on_500.csv
```

or the last cell of `kappa_analysis.ipynb` (it auto-detects `model_preds_on_500.csv`).

You'll get a ready-to-paste table of each model's **accuracy and Cohen's κ vs the
human-majority label**, with 95% CIs — i.e. *which model best matches humans*.
The κ values in the paper (Fleiss 0.535, human-vs-auto 0.126) will also now come
from your real per-tweet auto labels via `fifa_auto_labels.csv`, matching exactly.
```
