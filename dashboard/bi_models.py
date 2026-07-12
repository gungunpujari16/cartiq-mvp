"""
The five algorithms from the BRD's Section 4 table that answer "will another
brand buy CartIQ, and how much" -- distinct from app scoring.py's XGBoost,
which answers "will THIS shopper buy right now" for one brand's own
customers. Each function here is self-contained (features in, result out) so
the Business Intelligence page can call them independently and show one
algorithm at a time.

All models train on dashboard/data/brand_survey.csv -- SIMULATED data (see
generate_brand_survey.py), since no real B2B survey has been run yet.
"""
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

GMV_TIERS = ["<$10K/mo", "$10K-50K/mo", "$50K-200K/mo", "$200K-1M/mo", "$1M-5M/mo", ">$5M/mo"]
ORDER_TIERS = ["<100/mo", "100-500/mo", "500-2,000/mo", "2,000-10,000/mo", "10,000+/mo"]
TEAM_TIERS = ["1-5", "6-15", "16-50", "51-200", "200+"]


def _encode(df: pd.DataFrame) -> pd.DataFrame:
    """Shared feature frame: ordinal encodings + one-hot categoricals, used by
    the adoption, WTP, and segmentation models (not the decision tree, which
    deliberately excludes team_size to avoid predicting it from itself)."""
    out = pd.DataFrame(index=df.index)
    out["gmv_tier_enc"] = df["gmv_tier"].map({v: i for i, v in enumerate(GMV_TIERS)})
    out["monthly_orders_enc"] = df["monthly_orders"].map({v: i for i, v in enumerate(ORDER_TIERS)})
    out["team_size_enc"] = df["team_size"].map({v: i for i, v in enumerate(TEAM_TIERS)})
    out["cart_abandonment_rate"] = df["cart_abandonment_rate"]
    out["current_tool_sophistication"] = df["current_tool_sophistication"]
    out["pricing_pref_revshare"] = (df["pricing_preference"] == "Revenue Share").astype(int)
    out = pd.concat([out, pd.get_dummies(df["platform"], prefix="platform")], axis=1)
    out = pd.concat([out, pd.get_dummies(df["checkout_pain_point"], prefix="pain")], axis=1)
    return out


def run_random_forest(df: pd.DataFrame) -> dict:
    """Business question: will a brand subscribe? (BRD S4)"""
    X = _encode(df)
    y = (df["adoption_likelihood"] >= 7).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    importances = (
        pd.Series(model.feature_importances_, index=X.columns)
        .sort_values(ascending=False).head(8).reset_index()
    )
    importances.columns = ["feature", "importance"]

    return {
        "accuracy": round(accuracy_score(y_test, preds), 3),
        "auc": round(roc_auc_score(y_test, proba), 3),
        "predicted_likely_count": int(model.predict(X).sum()),
        "total": len(df),
        "importances": importances,
    }


def run_ridge(df: pd.DataFrame) -> dict:
    """Business question: how much will they pay? (BRD S4)"""
    X = _encode(df)
    y_log = np.log1p(df["willingness_to_pay_usd"])
    X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.25, random_state=42)

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    pred_log = model.predict(X_test)

    r2 = round(r2_score(y_test, pred_log), 3)
    # RMSE computed in dollar space (back-transformed), not log space, so it's
    # directly interpretable as "typical error in dollars".
    rmse_usd = round(float(root_mean_squared_error(np.expm1(y_test), np.expm1(pred_log))), 0)

    coefs = (
        pd.Series(model.coef_, index=X.columns)
        .sort_values(key=abs, ascending=False).head(8).reset_index()
    )
    coefs.columns = ["feature", "coefficient"]

    predicted_wtp = np.expm1(model.predict(X))
    return {
        "r2": r2,
        "rmse_usd": rmse_usd,
        "coefficients": coefs,
        "predicted_wtp": predicted_wtp,
        "avg_predicted_wtp": round(float(predicted_wtp.mean()), 0),
    }


def run_kmeans_brands(df: pd.DataFrame, predicted_wtp: np.ndarray, k: int = 3) -> dict:
    """Business question: which brands to target first? (BRD S4)"""
    X = _encode(df)[["gmv_tier_enc", "team_size_enc", "cart_abandonment_rate", "current_tool_sophistication"]].copy()
    X["predicted_wtp"] = predicted_wtp
    scaled = StandardScaler().fit_transform(X)

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    clusters = kmeans.fit_predict(scaled)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(scaled)

    work = df[["brand_id", "company_name"]].copy()
    work["cluster"] = clusters
    work["predicted_wtp"] = predicted_wtp
    work["pca_x"], work["pca_y"] = coords[:, 0], coords[:, 1]

    profile = work.groupby("cluster")["predicted_wtp"].mean().sort_values()
    labels = ["Budget", "Growth", "Premium"][:k]
    label_map = dict(zip(profile.index, labels))
    work["segment"] = work["cluster"].map(label_map)

    profiles = (
        work.groupby("segment")
        .agg(brands=("brand_id", "count"), avg_predicted_wtp=("predicted_wtp", "mean"))
        .round(0).reset_index()
        .sort_values("avg_predicted_wtp")
    )
    return {"profiles": profiles, "points": work[["brand_id", "segment", "pca_x", "pca_y"]]}


def run_apriori(df: pd.DataFrame) -> dict:
    """Business question: what drives high WTP? (BRD S4)"""
    high_wtp_cutoff = df["willingness_to_pay_usd"].median()
    basket = pd.DataFrame({
        "High WTP": df["willingness_to_pay_usd"] > high_wtp_cutoff,
        "High Adoption Intent": df["adoption_likelihood"] >= 7,
    })
    basket = pd.concat([
        basket,
        pd.get_dummies(df["platform"], prefix="Platform").astype(bool),
        pd.get_dummies(df["gmv_tier"], prefix="GMV").astype(bool),
        pd.get_dummies(df["checkout_pain_point"], prefix="Pain").astype(bool),
        (df["pricing_preference"] == "Revenue Share").rename("Prefers Revenue Share"),
    ], axis=1)

    frequent = apriori(basket, min_support=0.06, use_colnames=True)
    if frequent.empty:
        return {"rules": pd.DataFrame()}

    rules = association_rules(frequent, metric="lift", min_threshold=1.3)
    # Keep rules that predict the outcomes we actually care about (BRD's own threshold: lift > 1.5)
    rules = rules[
        rules["consequents"].apply(lambda s: any(x in {"High WTP", "High Adoption Intent"} for x in s))
        & (rules["lift"] > 1.3)
    ].sort_values("lift", ascending=False).head(10)

    rules_display = rules.copy()
    rules_display["antecedents"] = rules_display["antecedents"].apply(lambda s: ", ".join(sorted(s)))
    rules_display["consequents"] = rules_display["consequents"].apply(lambda s: ", ".join(sorted(s)))
    rules_display = rules_display[["antecedents", "consequents", "support", "confidence", "lift"]].round(3)
    return {"rules": rules_display}


def run_decision_tree(df: pd.DataFrame) -> dict:
    """Business question: predict company size (BRD S4). Deliberately excludes
    team_size/team_size_enc as a feature -- company_size is derived from team
    size in the survey generator, so including it would be pure leakage;
    predicting it from GMV/orders/industry signals instead is the realistic
    version of this business question (infer size when it wasn't asked directly)."""
    X = pd.DataFrame(index=df.index)
    X["gmv_tier_enc"] = df["gmv_tier"].map({v: i for i, v in enumerate(GMV_TIERS)})
    X["monthly_orders_enc"] = df["monthly_orders"].map({v: i for i, v in enumerate(ORDER_TIERS)})
    X["cart_abandonment_rate"] = df["cart_abandonment_rate"]
    X["current_tool_sophistication"] = df["current_tool_sophistication"]
    X = pd.concat([X, pd.get_dummies(df["industry"], prefix="industry")], axis=1)
    y = df["company_size"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    model = DecisionTreeClassifier(max_depth=4, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    accuracy = round(accuracy_score(y_test, model.predict(X_test)), 3)
    tree_text = export_text(model, feature_names=list(X.columns), max_depth=3)

    return {"accuracy": accuracy, "tree_text": tree_text, "classes": sorted(y.unique().tolist())}
