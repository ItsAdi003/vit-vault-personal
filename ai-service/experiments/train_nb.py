"""
AML experiment 2: learn the Naive Bayes parameters from the gallery's own labelled data.

Ground truth = the event category (INSTITUTIONAL / COMMITTEE) that the uploader
chose for each folder; features = the CLIP tags of every approved photo.

  * 5-fold stratified cross-validation of the learned model
  * the hand-crafted table of the live app evaluated on exactly the same photos
  * a final model fitted on all data and saved to ml/artifacts/nb_learned.json
    (the AI Lab page and /api/lab/nb pick it up automatically)

Run (needs the Supabase project to be reachable):  python experiments/train_nb.py
"""
import contextlib
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _report import ARTIFACT_DIR, print_table, save_report  # noqa: E402

from config import CANDIDATE_LABELS  # noqa: E402
from ml.nb_model import BernoulliNaiveBayes, classification_report  # noqa: E402

CLASSES = ["INSTITUTIONAL", "COMMITTEE"]


def load_records():
    from config import supabase
    rows = (supabase.table("media").select("ai_tags, events(type)").eq("status", "APPROVED").execute().data) or []
    recs = []
    for r in rows:
        etype = (r.get("events") or {}).get("type")
        tags = r.get("ai_tags") or []
        if etype in CLASSES and tags and "duplicate" not in tags:
            recs.append((tags, etype))
    return recs


def handcrafted_predict(tag_sets):
    from services.classifier import classify_category
    preds, proba = [], []
    for tags in tag_sets:
        with contextlib.redirect_stdout(io.StringIO()):
            r = classify_category(tags)
        preds.append(r["predicted_category"])
        proba.append([r["posteriors"][c] for c in CLASSES])
    return preds, np.array(proba)


def run(records, folds=5, alpha=1.0, seed=0, save_model=True):
    from sklearn.model_selection import StratifiedKFold

    tag_sets = [r[0] for r in records]
    y = np.array([r[1] for r in records])
    counts = {c: int((y == c).sum()) for c in CLASSES}
    if min(counts.values()) < 2:
        raise SystemExit(f"Need at least 2 photos per class, found {counts}. Upload more labelled events first.")
    folds = max(2, min(folds, min(counts.values())))

    pred = np.empty(len(y), dtype=object)
    proba = np.zeros((len(y), 2))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(tag_sets, y):
        m = BernoulliNaiveBayes(CANDIDATE_LABELS, CLASSES, alpha).fit([tag_sets[i] for i in tr], y[tr])
        p = m.predict_proba([tag_sets[i] for i in te])
        proba[te] = p
        pred[te] = [CLASSES[i] for i in p.argmax(1)]
    learned = classification_report(list(y), list(pred), proba, CLASSES)

    h_pred, h_proba = handcrafted_predict(tag_sets)
    hand = classification_report(list(y), h_pred, h_proba, CLASSES)

    majority = max(counts.values()) / len(y)
    rows = [
        ["majority class", majority, None, None],
        ["hand-crafted table (live app)", hand["accuracy"], hand["macro_f1"], hand.get("log_loss")],
        [f"learned Bernoulli NB ({folds}-fold CV)", learned["accuracy"], learned["macro_f1"], learned.get("log_loss")],
    ]
    cols = ["model", "accuracy", "macro F1", "log-loss"]

    final = BernoulliNaiveBayes(CANDIDATE_LABELS, CLASSES, alpha).fit(tag_sets, y)
    if save_model:
        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        final.save(os.path.join(ARTIFACT_DIR, "nb_learned.json"))

    report = {
        "title": "Naive Bayes learned from the gallery's labelled events",
        "description": "Labels are the INSTITUTIONAL/COMMITTEE type chosen by uploaders; features are CLIP tags. "
                       "Learned numbers are out-of-fold (5-fold stratified CV); the hand-crafted table needs no "
                       "training so it is simply scored on the same photos.",
        "meta": {"photos": len(y), **counts, "alpha": alpha, "folds": folds},
        "table": {"columns": cols, "rows": rows},
        "confusion": {"classes": CLASSES, "matrix": learned["confusion_matrix"]},
        "notes": "Photos from the same folder are strongly correlated; because folds are not grouped by folder the "
                 "cross-validated accuracy is optimistic. Report it with that caveat.",
    }
    return report, final


def main():
    records = load_records()
    print(f"{len(records)} labelled photos")
    report, _ = run(records)
    print_table(report["table"]["columns"], report["table"]["rows"])
    print("saved", save_report("nb_learned_eval", report))
    print("saved model to", os.path.join(ARTIFACT_DIR, "nb_learned.json"))


if __name__ == "__main__":
    main()
