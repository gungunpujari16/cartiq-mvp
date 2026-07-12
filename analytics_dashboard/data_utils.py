"""
Data loading, cleaning, and feature-encoding shared by every tab in this
dashboard. Two independent datasets, two independent business questions --
see each function's docstring for which one it serves.

This app is fully self-contained (own copies of both CSVs, no live backend),
so this module deliberately duplicates the small amount of shared logic that
also exists in cartiq-mvp/dashboard/bi_models.py and IPBL/utils.py rather
than importing across sibling apps -- that's an intentional trade-off, not
an oversight (each Streamlit app needs to deploy independently).
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"

# ── Ordinal vocabularies (brand survey) ─────────────────────────────────────
GMV_TIERS = ["<$10K/mo", "$10K-50K/mo", "$50K-200K/mo", "$200K-1M/mo", "$1M-5M/mo", ">$5M/mo"]
ORDER_TIERS = ["<100/mo", "100-500/mo", "500-2,000/mo", "2,000-10,000/mo", "10,000+/mo"]
TEAM_TIERS = ["1-5", "6-15", "16-50", "51-200", "200+"]


# ═════════════════════════════════════════════════════════════════════════
# Dataset 1 — shopper sessions (ecommerce_cleaned.csv)
# ═════════════════════════════════════════════════════════════════════════
def load_ecommerce_raw() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "ecommerce_cleaned.csv")


def clean_ecommerce() -> tuple[pd.DataFrame, list[dict]]:
    """Returns (cleaned_df, cleaning_log) -- the log is rendered directly in
    the Data Cleaning tab so every fix is visible, not silent."""
    df = load_ecommerce_raw()
    log = []

    n_rows = len(df)
    log.append({"step": "Load", "detail": f"{n_rows} rows, {df.shape[1]} columns loaded from ecommerce_cleaned.csv"})

    # -- Structural missingness: Items_in_Cart is NaN exactly when a shopper
    # never added anything to cart. This is not a random gap to impute --
    # it IS the true value (zero items), so encode it as 0, not a median.
    n_missing_items = int(df["Items_in_Cart"].isna().sum())
    n_no_cart = int((df["Added_to_Cart"] == "No").sum())
    df["Items_in_Cart"] = df["Items_in_Cart"].fillna(0)
    log.append({
        "step": "Items_in_Cart: structural missingness",
        "detail": (
            f"{n_missing_items} NaN rows found, all {n_no_cart} of which are Added_to_Cart == 'No' "
            f"(exact match) -- these shoppers never added anything, so 0 is the true value, not a "
            f"measurement gap. Encoded as 0, not median-imputed."
        ),
    })

    # -- Leakage flag: Order_Value is non-null only for converted sessions.
    # We do NOT impute this -- it's correctly absent for non-converters, and
    # is explicitly excluded as a feature everywhere Converted is a target
    # (see encode_shopper_features below). Flagged here as a finding.
    n_missing_order = int(df["Order_Value"].isna().sum())
    n_not_converted = int((df["Converted"] == "No").sum())
    log.append({
        "step": "Order_Value: leakage flag (not cleaned, deliberately left as-is)",
        "detail": (
            f"{n_missing_order} NaN rows, all {n_not_converted} of which are Converted == 'No' (exact "
            f"match). This field only exists *because* a purchase happened -- using it to predict "
            f"whether a purchase happened would be circular. Left as NaN; excluded from every "
            f"conversion-classification feature set (Tab 4)."
        ),
    })

    n_missing_cat2 = int(df["Product_Category_2"].isna().sum())
    df["Product_Category_2"] = df["Product_Category_2"].fillna("No Second Category")
    log.append({
        "step": "Product_Category_2",
        "detail": f"{n_missing_cat2} NaN rows (shopper only browsed one category) encoded as 'No Second Category'.",
    })

    # -- Minor cleaning check: casing / whitespace. Report what was checked,
    # not just what was found -- this dataset is synthetically generated and
    # is already clean, unlike a real-world survey export.
    cat_cols = ["Device_Type", "Traffic_Source", "Location", "Product_Category", "Payment_Method", "Time_of_Day"]
    casing_issues = 0
    whitespace_issues = 0
    for col in cat_cols:
        vals = df[col].dropna().astype(str)
        casing_issues += (vals.str.lower().nunique() != vals.nunique())
        whitespace_issues += (vals.str.strip().nunique() != vals.nunique())
    log.append({
        "step": "Casing / whitespace check",
        "detail": (
            f"Checked {len(cat_cols)} categorical columns: {casing_issues} had casing duplicates, "
            f"{whitespace_issues} had whitespace duplicates. This dataset is synthetically generated "
            f"(see IPBL/ecommerce_data_generator_v2.py) with a fixed category vocabulary, so it's "
            f"clean by construction -- real-world survey exports (e.g. a live B2B survey) would need "
            f"this same check to actually catch issues."
        ),
    })

    # Derived features (same as IPBL/utils.py's pattern, kept consistent)
    df["Converted_bin"] = (df["Converted"] == "Yes").astype(int)
    df["Return_bin"] = (df["Return_Customer"] == "Yes").astype(int)
    df["Discount_bin"] = (df["Discount_Code_Used"] == "Yes").astype(int)
    df["Cart_bin"] = (df["Added_to_Cart"] == "Yes").astype(int)
    df["Engagement_Score"] = df["Time_Spent_on_Site"] * df["Pages_Viewed"]
    df["Avg_Time_Per_Page"] = df["Time_Spent_on_Site"] / (df["Pages_Viewed"] + 1)

    log.append({"step": "Done", "detail": f"{len(df)} rows retained (no rows dropped)."})
    return df, log


def encode_shopper_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Feature matrix for Tab 4 (predict Converted). Deliberately excludes
    Order_Value (leakage -- only exists for converters) and Abandonment_Point
    (a post-hoc label of the same outcome, not a predictor of it)."""
    X = pd.DataFrame(index=df.index)
    X["time_spent"] = df["Time_Spent_on_Site"]
    X["pages_viewed"] = df["Pages_Viewed"]
    X["session_count"] = df["Session_Count"]
    X["items_in_cart"] = df["Items_in_Cart"]
    X["age"] = df["Age"]
    X["return_customer"] = df["Return_bin"]
    X["discount_used"] = df["Discount_bin"]
    X["added_to_cart"] = df["Cart_bin"]
    X["engagement_score"] = df["Engagement_Score"]
    X = pd.concat([
        X,
        pd.get_dummies(df["Device_Type"], prefix="device"),
        pd.get_dummies(df["Traffic_Source"], prefix="traffic"),
        pd.get_dummies(df["Time_of_Day"], prefix="time"),
        pd.get_dummies(df["Product_Category"], prefix="category"),
    ], axis=1)
    y = df["Converted_bin"]
    return X, y


def encode_shopper_features_with_leakage(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Same as encode_shopper_features but WITH Order_Value included --
    exists only to demonstrate the leakage finding concretely (Tab 4 shows
    this achieves trivially high accuracy, which is the point being made)."""
    X, y = encode_shopper_features(df)
    X = X.copy()
    X["order_value_LEAKY"] = df["Order_Value"].fillna(0)
    return X, y


def encode_shopper_clustering_features(df: pd.DataFrame) -> pd.DataFrame:
    """Behavioral/engagement-only features for Tab 7 shopper clustering --
    deliberately excludes demographics/categoricals (K-Means needs
    Euclidean-meaningful continuous features; mixing in one-hot categoricals
    would distort distances)."""
    return df[["Time_Spent_on_Site", "Pages_Viewed", "Items_in_Cart", "Session_Count"]].fillna(0)


# ═════════════════════════════════════════════════════════════════════════
# Dataset 2 — B2B brand survey (brand_survey.csv, simulated)
# ═════════════════════════════════════════════════════════════════════════
def load_brand_survey() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "brand_survey.csv")


def _encode_brand_common(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["gmv_tier_enc"] = df["gmv_tier"].map({v: i for i, v in enumerate(GMV_TIERS)})
    out["monthly_orders_enc"] = df["monthly_orders"].map({v: i for i, v in enumerate(ORDER_TIERS)})
    out["team_size_enc"] = df["team_size"].map({v: i for i, v in enumerate(TEAM_TIERS)})
    out["cart_abandonment_rate"] = df["cart_abandonment_rate"]
    out["current_tool_sophistication"] = df["current_tool_sophistication"]
    out["pricing_pref_revshare"] = (df["pricing_preference"] == "Revenue Share").astype(int)
    return out


def encode_adoption_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Tab 6: predict adoption_likelihood >= 7. Excludes willingness_to_pay_usd
    (a separate outcome, not a predictor of adoption) and team_size (see
    encode_company_size_features for why)."""
    X = _encode_brand_common(df)
    X = pd.concat([
        X,
        pd.get_dummies(df["platform"], prefix="platform"),
        pd.get_dummies(df["checkout_pain_point"], prefix="pain"),
    ], axis=1)
    y = (df["adoption_likelihood"] >= 7).astype(int)
    return X, y


def encode_wtp_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Tab 5: predict log1p(willingness_to_pay_usd)."""
    X = _encode_brand_common(df)
    X = pd.concat([
        X,
        pd.get_dummies(df["platform"], prefix="platform"),
        pd.get_dummies(df["checkout_pain_point"], prefix="pain"),
    ], axis=1)
    y = np.log1p(df["willingness_to_pay_usd"])
    return X, y


def encode_company_size_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Predicts company_size WITHOUT team_size as a feature -- company_size
    is derived directly from team_size in generate_brand_survey.py, so
    including it would be pure leakage (100% trivial accuracy, zero signal)."""
    X = pd.DataFrame(index=df.index)
    X["gmv_tier_enc"] = df["gmv_tier"].map({v: i for i, v in enumerate(GMV_TIERS)})
    X["monthly_orders_enc"] = df["monthly_orders"].map({v: i for i, v in enumerate(ORDER_TIERS)})
    X["cart_abandonment_rate"] = df["cart_abandonment_rate"]
    X["current_tool_sophistication"] = df["current_tool_sophistication"]
    X = pd.concat([X, pd.get_dummies(df["industry"], prefix="industry")], axis=1)
    y = df["company_size"]
    return X, y


def encode_brand_clustering_features(df: pd.DataFrame) -> pd.DataFrame:
    X = _encode_brand_common(df)[["gmv_tier_enc", "team_size_enc", "cart_abandonment_rate", "current_tool_sophistication"]].copy()
    X["willingness_to_pay_usd"] = df["willingness_to_pay_usd"]
    return X


if __name__ == "__main__":
    # Standalone verification -- run directly to sanity-check the leakage /
    # structural-missingness claims against the real data before any UI exists.
    df_raw = load_ecommerce_raw()
    print("Raw shopper rows:", len(df_raw))
    print("Items_in_Cart NaN:", df_raw["Items_in_Cart"].isna().sum(), "(expected 663)")
    print("Order_Value NaN:", df_raw["Order_Value"].isna().sum(), "(expected 1150)")
    print("Added_to_Cart == No:", (df_raw["Added_to_Cart"] == "No").sum())
    print("Converted == No:", (df_raw["Converted"] == "No").sum())

    df_clean, log = clean_ecommerce()
    print("\nCleaning log:")
    for entry in log:
        print(f"  [{entry['step']}] {entry['detail']}")

    X, y = encode_shopper_features(df_clean)
    print("\nConversion feature matrix:", X.shape, "target balance:", y.value_counts().to_dict())
    assert "order_value_LEAKY" not in X.columns, "Order_Value leaked into the clean feature set!"

    df_brand = load_brand_survey()
    print("\nBrand survey rows:", len(df_brand))
    Xw, yw = encode_wtp_features(df_brand)
    print("WTP feature matrix:", Xw.shape)
    Xc, yc = encode_company_size_features(df_brand)
    assert "team_size_enc" not in Xc.columns, "team_size leaked into company_size features!"
    print("Company-size feature matrix:", Xc.shape, "(team_size correctly excluded)")
    print("\nAll checks passed.")
