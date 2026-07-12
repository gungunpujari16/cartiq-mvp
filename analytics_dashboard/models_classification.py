"""
Shared 4-model classification suite (KNN / Decision Tree / Random Forest /
Gradient Boosting) used identically by Tab 4 (shopper conversion) and Tab 6
(brand adoption) -- same methodology, different data, so this lives in one
place rather than being duplicated per tab.

Every model is evaluated the same way: 5-fold stratified cross-validation
(not a single train/test split) for the headline metrics, plus out-of-fold
predictions (cross_val_predict) across the whole dataset for a single,
statistically honest confusion matrix -- avoids both "lucky split" bias and
wasting data on a held-out set that's never reused.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

MODEL_METHODOLOGY = {
    "KNN": {
        "why": "A distance-based baseline with no assumptions about feature relationships -- useful as a "
               "floor to compare the other three against.",
        "why_not": "Expected to underperform here: one-hot encoding several categorical variables creates a "
                   "high-dimensional sparse space where Euclidean distance becomes less meaningful (the "
                   "curse of dimensionality) -- exactly what the confusion matrix below tends to show.",
        "calculates": "Classifies each row by majority vote among its 15 nearest neighbors in scaled feature space.",
    },
    "Decision Tree": {
        "why": "Interpretable, handles mixed categorical/numerical data natively -- useful as an "
               "interpretability baseline against the two ensembles below.",
        "why_not": "A single tree is prone to overfitting/high variance; expect the ensembles to beat it on "
                   "held-out performance even though its logic is easier to read.",
        "calculates": "A single tree of rules (max depth 6), splitting on whichever feature most reduces class impurity at each step.",
    },
    "Random Forest": {
        "why": "Averaging many decision trees should reduce the single tree's overfitting/variance, and it "
               "gives feature importance for free -- useful for tying results back to the diagnostic tab's narrative.",
        "why_not": "More opaque than a single tree (200 trees voting, not one readable rule set); usually "
                   "beaten on raw accuracy by boosting on tabular data of this size.",
        "calculates": "Majority vote across 200 independently-trained decision trees, each on a random subset of rows and features.",
    },
    "Gradient Boosting (XGBoost)": {
        "why": "Sequentially corrects the previous trees' errors -- typically the strongest raw accuracy on "
               "tabular data of this size, and it's the algorithm actually used in CartIQ's production scoring engine.",
        "why_not": "With only ~1,000-1,400 rows and default-ish settings it risks overfitting without careful "
                   "cross-validation -- which is exactly why every model here is evaluated with 5-fold CV, "
                   "not a single train/test split.",
        "calculates": "200 boosted trees, each trained to predict the residual error of the ensemble so far.",
    },
}


def _build_models(y: pd.Series) -> dict:
    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    return {
        "KNN": KNeighborsClassifier(n_neighbors=15),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42),
        "Gradient Boosting (XGBoost)": xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.08,
            scale_pos_weight=pos_weight, eval_metric="logloss", random_state=42,
        ),
    }


def run_classification_suite(X: pd.DataFrame, y: pd.Series) -> dict:
    """Returns {model_name: {cv_accuracy, cv_precision, cv_recall, cv_f1, cv_roc_auc,
    confusion_matrix, importances (Series or None)}}"""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    results = {}
    for name, model in _build_models(y).items():
        Xin = X_scaled if name == "KNN" else X
        cv = cross_validate(model, Xin, y, cv=skf, scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])
        proba = cross_val_predict(model, Xin, y, cv=skf, method="predict_proba")[:, 1]
        preds = (proba >= 0.5).astype(int)
        cm = confusion_matrix(y, preds)

        model.fit(Xin, y)  # full-data fit, only used for feature importance display
        importances = None
        if hasattr(model, "feature_importances_"):
            importances = (
                pd.Series(model.feature_importances_, index=X.columns)
                .sort_values(ascending=False).head(10)
            )

        results[name] = {
            "cv_accuracy": float(cv["test_accuracy"].mean()),
            "cv_precision": float(cv["test_precision"].mean()),
            "cv_recall": float(cv["test_recall"].mean()),
            "cv_f1": float(cv["test_f1"].mean()),
            "cv_roc_auc": float(cv["test_roc_auc"].mean()),
            "confusion_matrix": cm,
            "importances": importances,
        }
    return results


def naive_baseline_accuracy(y: pd.Series) -> float:
    """Accuracy of always predicting the majority class -- the floor every
    model needs to beat to be worth anything (per the reference template's
    own point: a model predicting one class for everyone still scores ~55%+
    'accuracy' trivially on an imbalanced target)."""
    return float(max(y.mean(), 1 - y.mean()))


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data_utils import clean_ecommerce, encode_shopper_features

    df, _ = clean_ecommerce()
    X, y = encode_shopper_features(df)
    print("Naive baseline accuracy (always predict majority class):", round(naive_baseline_accuracy(y), 3))
    results = run_classification_suite(X, y)
    for name, r in results.items():
        print(f"\n{name}:")
        print(f"  CV accuracy={r['cv_accuracy']:.3f} precision={r['cv_precision']:.3f} "
              f"recall={r['cv_recall']:.3f} f1={r['cv_f1']:.3f} roc_auc={r['cv_roc_auc']:.3f}")
        print(f"  Confusion matrix:\n{r['confusion_matrix']}")
        if r["importances"] is not None:
            print(f"  Top features: {r['importances'].head(3).to_dict()}")
    print("\nAll classification suite checks ran without error.")
