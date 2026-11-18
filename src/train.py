"""Train LightGBM, XGBoost and CatBoost with stratified 5-fold CV and blend them.

Usage:
    python src/train.py            # all models
    python src/train.py lgb xgb    # a subset
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from features import CAT_COLS, ID, TARGET, build, load, target_encode  # noqa: E402

SEED = 42
N_FOLDS = 5
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)


def fit_lgb(Xtr, ytr, Xva, yva):
    m = LGBMClassifier(
        n_estimators=5000, learning_rate=0.02, num_leaves=31, min_child_samples=100,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5.0,
        max_cat_to_onehot=16, random_state=SEED, verbose=-1,
    )
    m.fit(Xtr, ytr, eval_X=(Xva,), eval_y=(yva,), eval_metric="auc",
          callbacks=[early_stopping(200, verbose=False), log_evaluation(0)])
    return m, lambda X: m.predict_proba(X)[:, 1]


def fit_xgb(Xtr, ytr, Xva, yva):
    m = XGBClassifier(
        n_estimators=5000, learning_rate=0.02, max_depth=6, min_child_weight=10,
        subsample=0.8, colsample_bytree=0.5, reg_lambda=5.0, tree_method="hist",
        enable_categorical=True, max_cat_to_onehot=16, eval_metric="auc",
        early_stopping_rounds=200, random_state=SEED, n_jobs=-1,
    )
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m, lambda X: m.predict_proba(X)[:, 1]


def fit_cat(Xtr, ytr, Xva, yva):
    m = CatBoostClassifier(
        iterations=4000, learning_rate=0.05, depth=6, l2_leaf_reg=5,
        eval_metric="AUC", od_type="Iter", od_wait=200, random_seed=SEED,
        cat_features=CAT_COLS, verbose=0, thread_count=-1,
    )
    to_str = lambda X: X.astype({c: str for c in CAT_COLS})  # noqa: E731
    m.fit(to_str(Xtr), ytr, eval_set=(to_str(Xva), yva))
    return m, lambda X: m.predict_proba(to_str(X))[:, 1]


MODELS = {"lgb": fit_lgb, "xgb": fit_xgb, "cat": fit_cat}


def run_cv(name, X, y, X_test, X_orig, y_orig):
    """Stratified K-fold. The original dataset is appended to every training fold only,
    so validation scores are measured purely on competition data."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    importances = []
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        t = time.time()
        Xtr = pd.concat([X.iloc[tr], X_orig], ignore_index=True)
        ytr = np.concatenate([y[tr], y_orig])
        # Target encoding is learned on the training fold only (nested K-fold inside it).
        Xtr, (Xva, Xte) = target_encode(Xtr, ytr, [X.iloc[va].reset_index(drop=True), X_test])
        model, predict = MODELS[name](Xtr, ytr, Xva, y[va])
        oof[va] = predict(Xva)
        test_pred += predict(Xte) / N_FOLDS
        if hasattr(model, "feature_importances_"):
            importances.append(model.feature_importances_)
        print(f"  [{name}] fold {fold}: AUC={roc_auc_score(y[va], oof[va]):.5f} ({time.time() - t:.0f}s)")
    auc = roc_auc_score(y, oof)
    print(f"[{name}] OOF AUC = {auc:.5f}")
    np.save(OUT / f"oof_{name}.npy", oof)
    np.save(OUT / f"test_{name}.npy", test_pred)
    if importances:
        pd.Series(np.mean(importances, axis=0), index=Xtr.columns).sort_values(ascending=False) \
            .to_csv(OUT / f"importance_{name}.csv", header=["importance"])
    return auc


def blend(y, names):
    """Find non-negative weights on rank-normalised OOF predictions that maximise AUC."""
    oofs = np.column_stack([rankdata(np.load(OUT / f"oof_{n}.npy")) / len(y) for n in names])
    tests = [np.load(OUT / f"test_{n}.npy") for n in names]
    tests = np.column_stack([rankdata(t) / len(t) for t in tests])

    def loss(w):
        w = np.abs(w) / np.abs(w).sum()
        return -roc_auc_score(y, oofs @ w)

    res = minimize(loss, np.ones(len(names)) / len(names), method="Nelder-Mead")
    w = np.abs(res.x) / np.abs(res.x).sum()
    return dict(zip(names, w.round(4).tolist())), -res.fun, tests @ w


def main():
    names = sys.argv[1:] or list(MODELS)
    train, test, orig = load()
    X, y, X_test, X_orig, y_orig = build(train, test, orig)
    print(f"train {X.shape}, test {X_test.shape}, original {X_orig.shape}, fraud rate {y.mean():.4f}")

    for n in names:
        run_cv(n, X, y, X_test, X_orig, y_orig)

    available = [n for n in MODELS if (OUT / f"oof_{n}.npy").exists()]
    weights, blend_auc, pred = blend(y, available)
    print(f"Blend weights {weights} -> OOF AUC = {blend_auc:.5f}")

    sub = pd.DataFrame({ID: test[ID], TARGET: pred})
    sub.to_csv(OUT / "submission.csv", index=False)
    all_scores = {n: roc_auc_score(y, np.load(OUT / f"oof_{n}.npy")) for n in available}
    json.dump({"oof_auc": all_scores, "blend_weights": weights, "blend_oof_auc": blend_auc},
              open(OUT / "cv_scores.json", "w"), indent=2)
    print("wrote outputs/submission.csv")


if __name__ == "__main__":
    main()
