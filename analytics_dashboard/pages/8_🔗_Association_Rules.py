import plotly.express as px
import streamlit as st

from data_utils import clean_ecommerce, load_brand_survey
from models_association import build_brand_basket, build_shopper_basket, mine_rules
from style import inject_css, insight, methodology, style_layout

st.set_page_config(page_title="Association Rules", page_icon="🔗", layout="wide")
inject_css()

st.markdown('<div class="dash-title">🔗 Association Rule Mining</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="dash-sub">Apriori, not FP-Growth: with only a handful of categorical items per basket and '
    '~200&ndash;1,400 transactions, candidate-generation cost stays manageable and the rules read directly as '
    'business statements &mdash; FP-Growth would only start to matter at a much larger item-set scale than '
    'either dataset here has.</div>',
    unsafe_allow_html=True,
)

view = st.radio("Which basket?", ["🛍️ Shopper Behavior → Converted", "🧠 Brand Traits → WTP / Adoption"], horizontal=True)

if view.startswith("🛍️"):
    df, _ = clean_ecommerce()
    basket = build_shopper_basket(df)
    consequents = ["Converted"]
    default_support, default_confidence, default_lift = 0.03, 0.15, 1.05
    st.caption("Items: Device Type, Traffic Source, Product Category, Time of Day, and Converted. Shopper "
               "behavior is a genuinely weaker signal than brand traits below — individual browsing habits "
               "don't determine conversion nearly as cleanly as firmographics determine WTP, so the default "
               "thresholds here are deliberately looser to surface any rules at all.")
else:
    df = load_brand_survey()
    basket = build_brand_basket(df)
    consequents = ["High WTP", "High Adoption Intent"]
    default_support, default_confidence, default_lift = 0.05, 0.30, 1.20
    st.markdown('<span class="sim-badge">⚠️ Simulated survey data (N=220)</span>', unsafe_allow_html=True)
    st.caption("Items: Platform, Checkout Pain Point, GMV Tier, Pricing Preference, High WTP (top 50% of "
               "willingness-to-pay), and High Adoption Intent (score ≥ 7).")

st.markdown("#### Thresholds")
c1, c2, c3 = st.columns(3)
min_support = c1.slider("Min support", 0.01, 0.30, default_support, 0.01,
                         help="Fraction of all rows containing both the antecedent and consequent.")
min_confidence = c2.slider("Min confidence", 0.05, 0.90, default_confidence, 0.05,
                            help="P(consequent | antecedent) — how often the rule holds when the antecedent is true.")
min_lift = c3.slider("Min lift", 1.0, 3.0, default_lift, 0.05,
                      help="How much more likely the consequent is given the antecedent, vs. its base rate. Lift=1 means no association.")

rules = mine_rules(basket, min_support=min_support, min_confidence=min_confidence,
                    min_lift=min_lift, consequent_filter=consequents, max_rules=30)

if rules.empty:
    st.warning("No rules clear these thresholds. This itself is a finding, not an error — try lowering "
               "min confidence or min lift with the sliders above.")
else:
    st.markdown(f"#### {len(rules)} rules found")
    sort_col = st.selectbox("Sort by", ["lift", "confidence", "support"], index=0)
    st.dataframe(rules.sort_values(sort_col, ascending=False), use_container_width=True, hide_index=True)

    fig = px.scatter(
        rules, x="support", y="confidence", size="lift", color="lift",
        color_continuous_scale="Blues", size_max=40,
        hover_data=["antecedents", "consequents"],
    )
    st.plotly_chart(style_layout(fig, "Support vs. Confidence (bubble size & color = lift)", height=420),
                     use_container_width=True)

    top = rules.sort_values("lift", ascending=False).iloc[0]
    insight(f"Strongest rule: <b>{top['antecedents']} → {top['consequents']}</b> "
            f"(support={top['support']:.3f}, confidence={top['confidence']:.3f}, lift={top['lift']:.2f}). "
            f"A lift of {top['lift']:.2f} means this combination is {top['lift']:.2f}x more likely to lead to "
            f"the outcome than the outcome's base rate alone — the kind of non-linear combination effect a "
            f"regression on individual features would miss entirely.")

methodology(
    "Apriori / Association Rule Mining",
    why="Both baskets are sets of categorical items (device, traffic source, platform, pain point, etc.) — "
        "Apriori finds which combinations co-occur far more than chance, something a classifier's per-feature "
        "importances can't express (it can't tell you 'Mobile AND Social Media together' matters more than "
        "either alone).",
    why_not="FP-Growth avoids Apriori's repeated candidate-generation passes, but that only pays off with far "
            "more items/transactions than either dataset here has — added complexity with no real benefit at "
            "this scale.",
    calculates="For every itemset combination clearing min support: confidence = P(consequent | antecedent), "
               "lift = confidence ÷ P(consequent) — how much the antecedent changes the odds versus the base rate.",
)
