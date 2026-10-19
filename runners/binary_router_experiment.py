# -*- coding: utf-8 -*-
"""
PHASE 2.1: Binary tool-router experiment.

Two-stage agent architecture test: a tiny classifier decides "does this query need a tool?"
before the LLM is invoked. Trained ONLY on the canonical 20-case suite (15 pos + 5 neg),
evaluated on the 30-case challenge suite (20 pos + 10 neg) — a clean, pre-existing split.

Models compared:
  * keyword heuristic baseline
  * char n-gram (2-4) hashed TF + logistic regression (numpy, deterministic full-batch GD)

Router input = concatenated user turns of the conversation (realistic: router sees the dialog).
Success criterion (from the paper plan): false-positive rate on negatives < 20%.
"""
import json
import re
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

CANON = r"C:\ai_models\test_suite_standardized.json"
HELD = r"C:\ai_models\heldout_test_suite_30.json"
OUT = r"C:\ai_models\paper_artifacts\binary_router_experiment.json"
OUT_DESKTOP = r"C:\Users\Рам\Desktop\BitNet_Research_Reports\binary_router_experiment.json"

TOOL_PAT = re.compile(
    r"(погод|температур|погода|weather|temperature|поиск|поищи|найди|новост|интернет|search|news|"
    r"python|скрипт|fibonacci|фибоначчи|простых чисел|посчитай|вычисли|calculate|compute|"
    r"прочитай|открой|файл|file|/etc|\.json|\.log|\.txt|read file|"
    r"sql|запрос к|базе данных|таблиц|database|select|заказов|пользовател|"
    r"кто (стал|победил|выиграл)|чемпион)",  # rule derived from canonical TRAIN case 4 ("Кто стал чемпионом мира по футболу в 2022")
    re.IGNORECASE,
)


def router_input(case):
    return " ".join(m["content"] for m in case["messages"] if m["role"] == "user")


def keyword_router(text):
    return bool(TOOL_PAT.search(text))


def ngrams(text, lo=2, hi=4):
    t = " " + text.lower().strip() + " "
    for n in range(lo, hi + 1):
        for i in range(len(t) - n + 1):
            yield t[i:i + n]


DIM = 2 ** 16
MASK = DIM - 1


def featurize(texts):
    X = np.zeros((len(texts), DIM), dtype=np.float32)
    for r, t in enumerate(texts):
        for g in ngrams(t):
            X[r, hash(g) & MASK] += 1.0
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


class LogReg:
    """Deterministic full-batch L2-regularized logistic regression."""

    def __init__(self, l2=1e-3, lr=0.5, iters=3000):
        self.l2, self.lr, self.iters = l2, lr, iters
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d, dtype=np.float32)
        self.b = 0.0
        for _ in range(self.iters):
            z = X @ self.w + self.b
            p = 1.0 / (1.0 + np.exp(-z))
            gw = X.T @ (p - y) / n + self.l2 * self.w
            gb = float(np.mean(p - y))
            self.w -= self.lr * gw
            self.b -= self.lr * gb
        return self

    def predict(self, X):
        return (X @ self.w + self.b) >= 0.0


def metrics(y_true, y_pred, n_pos, n_neg):
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum((~y_true) & y_pred))
    fn = int(np.sum(y_true & (~y_pred)))
    tn = int(np.sum((~y_true) & (~y_pred)))
    return {
        "positives": n_pos, "negatives": n_neg,
        "FN (tool missed)": fn, "FN rate": round(fn / n_pos * 100, 1) if n_pos else None,
        "FP (false trigger)": fp, "FP rate %": round(fp / n_neg * 100, 1) if n_neg else None,
        "TN (correct direct)": tn,
        "accuracy %": round((tp + tn) / (n_pos + n_neg) * 100, 1),
    }


def main():
    canon = json.load(open(CANON, encoding="utf-8"))
    held = json.load(open(HELD, encoding="utf-8"))

    tr_texts = [router_input(c) for c in canon]
    tr_y = np.array([bool(c["needs_tool"]) for c in canon])
    te_texts = [router_input(c) for c in held]
    te_y = np.array([bool(c["needs_tool"]) for c in held])

    print(f"train: {len(tr_texts)} ({int(tr_y.sum())} pos / {int((~tr_y).sum())} neg)")
    print(f"test : {len(te_texts)} ({int(te_y.sum())} pos / {int((~te_y).sum())} neg)")

    # 1) keyword baseline
    kw_tr = np.array([keyword_router(t) for t in tr_texts])
    kw_te = np.array([keyword_router(t) for t in te_texts])

    # 2) n-gram LR
    Xtr = featurize(tr_texts)
    Xte = featurize(te_texts)
    lr = LogReg().fit(Xtr, tr_y.astype(np.float32))
    lr_tr = lr.predict(Xtr)
    lr_te = lr.predict(Xte)

    # agreement between baseline and LR
    agree = int(np.mean(kw_te == lr_te) * 100)

    results = {
        "title": "Binary tool-router: keyword heuristic vs char-ngram logistic regression",
        "protocol": {
            "train_set": CANON + " (15 pos / 5 neg)",
            "test_set": HELD + " (20 pos / 10 neg)",
            "features": "char n-grams 2-4, hashed to 2^16, L2-normalized",
            "classifier": "logistic regression, full-batch GD, L2=1e-3, deterministic",
            "input": "concatenated user turns of the conversation",
        },
        "keyword_baseline": {
            "train": metrics(tr_y, kw_tr, int(tr_y.sum()), int((~tr_y).sum())),
            "test": metrics(te_y, kw_te, int(te_y.sum()), int((~te_y).sum())),
        },
        "ngram_logreg": {
            "train": metrics(tr_y, lr_tr, int(tr_y.sum()), int((~tr_y).sum())),
            "test": metrics(te_y, lr_te, int(te_y.sum()), int((~te_y).sum())),
        },
        "baseline_vs_lr_agreement_test_pct": agree,
        "notes": [
            "Router is a pre-LLM gate: FP -> false tool invocation avoided only if router is correct;",
            "when router says 'no tool', stage 2 = generation WITHOUT tool schema (laziness protocol showed BitNet answers directly);",
            "when router says 'tool', stage 2 = existing tool-calling path (C++ ReLU2+HF: S2 18/20, exec 2/8).",
        ],
    }

    print("\nKEYWORD baseline:")
    print(json.dumps(results["keyword_baseline"]["test"], ensure_ascii=False, indent=1))
    print("\nN-GRAM LR:")
    print(json.dumps(results["ngram_logreg"]["test"], ensure_ascii=False, indent=1))
    print(f"\ntrain ngram LR: {json.dumps(results['ngram_logreg']['train'], ensure_ascii=False)}")
    print(f"agreement baseline vs LR on test: {agree}%")

    # per-case errors on test
    errors = []
    for i, c in enumerate(held):
        pred = bool(lr_te[i])
        if pred != bool(te_y[i]):
            errors.append({"id": c["id"], "category": c["category"],
                           "needs_tool": bool(te_y[i]), "router_pred": pred,
                           "query": router_input(c)[:90]})
    results["ngram_logreg"]["test_errors"] = errors
    print("\ntest errors:")
    for e in errors:
        print("  ", e)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    try:
        import shutil
        shutil.copyfile(OUT, OUT_DESKTOP)
    except Exception as e:
        print(f"[warn] desktop copy failed: {e}")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
