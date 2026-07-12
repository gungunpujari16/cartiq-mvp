"""
Tab 8: association rule mining, shared by both the shopper-behavior view and
the brand-survey view. Apriori (not FP-Growth) is used deliberately for both:
with only a handful of categorical items per basket and ~200-1,400
transactions, Apriori's candidate-generation cost is manageable and its
rules are the most directly interpretable for a business audience --
FP-Growth would matter more at a much larger item-set scale than either
dataset here has.
"""
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


def build_shopper_basket(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([
        pd.get_dummies(df["Device_Type"], prefix="Device"),
        pd.get_dummies(df["Traffic_Source"], prefix="Traffic"),
        pd.get_dummies(df["Product_Category"], prefix="Category"),
        pd.get_dummies(df["Time_of_Day"], prefix="Time"),
        (df["Converted"] == "Yes").rename("Converted"),
    ], axis=1).astype(bool)


def build_brand_basket(df: pd.DataFrame) -> pd.DataFrame:
    high_wtp = df["willingness_to_pay_usd"] > df["willingness_to_pay_usd"].median()
    high_adoption = df["adoption_likelihood"] >= 7
    return pd.concat([
        pd.get_dummies(df["platform"], prefix="Platform"),
        pd.get_dummies(df["checkout_pain_point"], prefix="Pain"),
        pd.get_dummies(df["gmv_tier"], prefix="GMV"),
        (df["pricing_preference"] == "Revenue Share").rename("Prefers Revenue Share"),
        high_wtp.rename("High WTP"),
        high_adoption.rename("High Adoption Intent"),
    ], axis=1).astype(bool)


def mine_rules(basket: pd.DataFrame, min_support: float = 0.05, min_confidence: float = 0.4,
               min_lift: float = 1.2, consequent_filter: list[str] | None = None,
               max_rules: int = 25) -> pd.DataFrame:
    """consequent_filter: if given, only keep rules whose consequent is one
    of these items (e.g. ['Converted'] or ['High WTP', 'High Adoption Intent'])
    -- otherwise Apriori returns rules for every possible combination, most
    of which aren't the business question being asked."""
    frequent = apriori(basket, min_support=min_support, use_colnames=True)
    if frequent.empty:
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

    rules = association_rules(frequent, metric="confidence", min_threshold=min_confidence)
    rules = rules[rules["lift"] >= min_lift]
    if consequent_filter:
        rules = rules[rules["consequents"].apply(lambda s: any(x in consequent_filter for x in s))]
    rules = rules.sort_values("lift", ascending=False).head(max_rules).copy()
    rules["antecedents"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
    rules["consequents"] = rules["consequents"].apply(lambda s: ", ".join(sorted(s)))
    return rules[["antecedents", "consequents", "support", "confidence", "lift"]].round(3)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data_utils import clean_ecommerce, load_brand_survey

    df, _ = clean_ecommerce()
    basket = build_shopper_basket(df)
    rules = mine_rules(basket, min_support=0.04, min_confidence=0.3, min_lift=1.1, consequent_filter=["Converted"])
    print(f"Shopper rules found: {len(rules)}")
    print(rules.head(5).to_string(index=False))

    df_b = load_brand_survey()
    basket_b = build_brand_basket(df_b)
    rules_b = mine_rules(basket_b, min_support=0.05, min_confidence=0.3, min_lift=1.2,
                          consequent_filter=["High WTP", "High Adoption Intent"])
    print(f"\nBrand rules found: {len(rules_b)}")
    print(rules_b.head(5).to_string(index=False))
    print("\nAll association-rule checks ran without error.")
