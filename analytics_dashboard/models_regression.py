"""
Tab 5: predicting willingness_to_pay_usd. VIF is computed FIRST -- it's the
actual justification for reaching for regularization at all, not a checklist
item run alongside the models. gmv_tier_enc / monthly_orders_enc /
team_size_enc are correlated by construction in generate_brand_survey.py
(bigger GMV -> more orders -> bigger team, each with noise), so real
multicollinearity is expected here, not manufactured.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """VIF > 5 is the usual rule-of-thumb flag for problematic multicollinearity;
    > 10 is severe. Computed on numeric features only (VIF on one-hot dummies
    is not meaningful in the same way)."""
    numeric_cols = [c for c in X.columns if X[c].nunique() > 2]  # skip binary/one-hot dummies
    X_num = X[numeric_cols].copy()
    X_c = add_constant(X_num)
    rows = []
    for i, col in enumerate(X_c.columns):
        if col == "const":
            continue
        vif = variance_inflation_factor(X_c.values, i)
        rows.append({"feature": col, "VIF": round(float(vif), 2)})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)


def fit_regression_suite(X: pd.DataFrame, y: pd.Series, alpha: float = 1.0, l1_ratio: float = 0.5,
                          test_size: float = 0.25, random_state: int = 42) -> dict:
    """Fits OLS + Ridge + Lasso + ElasticNet at the given alpha (the UI slider
    drives this on every rerun -- Streamlit reruns the script on interaction,
    so this is naturally live). Target y is expected to already be log1p-
    transformed; RMSE is also reported back-transformed to dollars."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    models = {
        "OLS (no regularization)": LinearRegression(),
        "Ridge": Ridge(alpha=alpha),
        "Lasso": Lasso(alpha=alpha, max_iter=8000),
        "Elastic Net": ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=8000),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)
        coefs = pd.Series(model.coef_, index=X.columns)
        results[name] = {
            "r2": float(r2_score(y_test, pred)),
            "rmse_usd": float(root_mean_squared_error(np.expm1(y_test), np.expm1(pred))),
            "coefficients": coefs.sort_values(key=abs, ascending=False),
            "n_zero_coefs": int((coefs.abs() < 1e-6).sum()),
            "coef_l2_norm": float(np.sqrt((coefs ** 2).sum())),
        }
    return results


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data_utils import load_brand_survey, encode_wtp_features

    df = load_brand_survey()
    X, y = encode_wtp_features(df)

    vif = compute_vif(X)
    print("VIF (numeric features only):")
    print(vif.to_string(index=False))

    print("\nRegression suite at alpha=1.0:")
    results = fit_regression_suite(X, y, alpha=1.0)
    for name, r in results.items():
        print(f"  {name}: R2={r['r2']:.3f} RMSE=${r['rmse_usd']:.0f} "
              f"zero_coefs={r['n_zero_coefs']} coef_l2_norm={r['coef_l2_norm']:.2f}")

    print("\nCoefficient shrinkage check (OLS vs Ridge coef L2 norm across alphas):")
    for a in [0.001, 0.1, 1.0, 10.0, 100.0]:
        r = fit_regression_suite(X, y, alpha=a)
        print(f"  alpha={a:<8} Ridge coef_l2_norm={r['Ridge']['coef_l2_norm']:.2f} "
              f"Lasso zero_coefs={r['Lasso']['n_zero_coefs']}")
    print("\nAll regression suite checks ran without error.")
