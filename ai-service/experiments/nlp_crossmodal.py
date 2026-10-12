"""
NLP experiment 2: text -> image retrieval and sensitivity to query wording.

2 000 CIFAR-10 test images are the "gallery". For every class we write several
ways of asking for it; each query is embedded with CLIP's text tower and the
gallery is ranked by cosine similarity. A retrieved image is relevant if it is
of that class (200 relevant images per query).

Run:  python experiments/nlp_crossmodal.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _cifar import CLASSES, clip_embeddings  # noqa: E402
from _report import print_table, save_report  # noqa: E402

from ml import retrieval as R  # noqa: E402
from ml import zeroshot  # noqa: E402

STYLES = {
    "bare label": lambda c, para: c,
    "template: a photo of a ...": lambda c, para: f"a photo of a {c}",
    "natural paraphrase": lambda c, para: para,
}

PARAPHRASES = {
    "airplane": "a jet flying in the sky", "automobile": "a car parked on the road",
    "bird": "a small bird perched outdoors", "cat": "a pet kitten indoors",
    "deer": "a wild stag standing in a forest", "dog": "a puppy playing outside",
    "frog": "a green amphibian near water", "horse": "a stallion standing in a field",
    "ship": "a large vessel on the ocean", "truck": "a heavy lorry carrying cargo",
}


def main():
    E, y, _ = clip_embeddings("test", 200)
    rows = []
    per_style = {}
    for style, make in STYLES.items():
        ap, p10, p50 = [], [], []
        for c, name in enumerate(CLASSES):
            q = zeroshot.encode_texts([make(name, PARAPHRASES[name])])[0]
            scores = R.cosine_scores(q, E)
            ranked = R.ranking(scores)
            rel = set(np.flatnonzero(y == c).tolist())
            ap.append(R.average_precision(ranked, rel))
            p10.append(R.precision_at_k(ranked, rel, 10))
            p50.append(R.precision_at_k(ranked, rel, 50))
        per_style[style] = ap
        rows.append([style, float(np.mean(ap)), float(np.mean(p10)), float(np.mean(p50))])

    # per-class difficulty under the natural-paraphrase style (shows which concepts are hard)
    hardest = sorted(zip(CLASSES, per_style["natural paraphrase"]), key=lambda t: t[1])[:3]
    cols = ["query style", "mAP", "P@10", "P@50"]
    print(f"gallery: {len(E)} images, 200 relevant per query, 10 queries per style")
    print_table(cols, rows)
    print("lowest-AP classes with natural paraphrases:", [(n, round(a, 3)) for n, a in hardest])

    save_report("nlp_crossmodal", {
        "title": "Text-to-image retrieval: effect of query wording",
        "description": "Rank 2 000 CIFAR-10 test images by cosine similarity to the CLIP embedding of the query text; "
                       "an image is relevant if it shows the queried class.",
        "meta": {"gallery images": len(E), "queries per style": 10, "relevant per query": 200,
                 "hardest classes": ", ".join(f"{n} ({a:.2f})" for n, a in hardest)},
        "table": {"columns": cols, "rows": rows},
        "notes": "A prompt template adds context words the text encoder saw during training ('a photo of ...'), "
                 "which usually helps; free-form paraphrases are closer to what users really type but are noisier.",
    })


if __name__ == "__main__":
    main()
