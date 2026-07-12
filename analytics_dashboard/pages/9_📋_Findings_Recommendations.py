import streamlit as st

from data_utils import clean_ecommerce, load_brand_survey
from style import diagnostic, inject_css, insight

st.set_page_config(page_title="Findings & Recommendations", page_icon="📋", layout="wide")
inject_css()

st.markdown('<div class="dash-title">📋 Findings & Recommendations</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="dash-sub">Synthesis across both business questions, prescriptive actions, and an honest '
    'statement of what this analysis can and cannot claim.</div>',
    unsafe_allow_html=True,
)

df, _ = clean_ecommerce()
brand = load_brand_survey()
added_pct = (df["Added_to_Cart"] == "Yes").mean() * 100
converted_pct = (df["Converted"] == "Yes").mean() * 100
avg_wtp = brand["willingness_to_pay_usd"].mean()
adopters = int((brand["adoption_likelihood"] >= 7).sum())
corr_adopt_abandon = brand["adoption_likelihood"].corr(brand["cart_abandonment_rate"])

st.subheader("1. What drives each outcome")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**🛍️ Shopper Conversion (Tab 4)**")
    st.markdown(
        "- Tree-based models (Decision Tree, Random Forest, XGBoost) all rank `Items_in_Cart` and "
        "`Engagement_Score`/`Time_Spent_on_Site` as the top drivers — cart-level commitment and active "
        "engagement, not demographics or traffic source.\n"
        "- KNN underperforms the naive baseline on recall — a concrete example of why model choice matters, "
        "not just accuracy in isolation (Tab 4).\n"
        "- Association rules (Tab 8) surface specific Device × Traffic × Category combinations that convert "
        "meaningfully above the shopper base rate, evidence conversion isn't reducible to single-feature effects."
    )
with c2:
    st.markdown("**🧠 Brand Adoption & WTP (Tabs 5-6)**")
    st.markdown(
        f"- `cart_abandonment_rate` is the top feature for adoption classification (Tab 6) — brands already "
        f"losing the most revenue want CartIQ the most (correlation with adoption likelihood: r={corr_adopt_abandon:.2f}).\n"
        "- GMV tier, order volume, and team size are correlated by construction (VIF>5, Tab 5) — this is why "
        "Ridge, not plain OLS or Lasso, is the regression of record for WTP.\n"
        "- Association rules (Tab 8) tie specific GMV tiers and pain points to the 'High WTP' / 'High Adoption "
        "Intent' flags directly."
    )

st.divider()
st.subheader("2. Where the two questions overlap")
insight(f"<b>Cart abandonment / checkout friction is the thread connecting both sides.</b> On the shopper "
        f"side, only {converted_pct:.1f}% of sessions convert despite {added_pct:.1f}% adding something to "
        f"cart — a {added_pct-converted_pct:.1f}-point intent-to-completion gap. On the brand side, the "
        f"brands reporting the highest abandonment rates are also the ones most likely to adopt CartIQ "
        f"(r={corr_adopt_abandon:.2f}). The same friction problem that CartIQ's shopper-scoring product is "
        f"built to fix is the reason a brand would buy CartIQ in the first place — the shopper-side findings "
        f"in Tabs 2-4 are effectively the sales pitch underlying the brand-side findings in Tabs 5-6.")

st.divider()
st.subheader("3. Prescriptive actions")
st.markdown(
    "- **For the shopper-facing product**: target discount/intervention logic at high-cart-value, "
    "low-engagement sessions specifically — not blanket discounting (Tab 3's discount-impact finding: "
    "modest conversion lift, negligible AOV difference from discounting everyone).\n"
    "- **For CartIQ's own sales motion**: prioritize outbound to brands with high reported cart abandonment "
    f"and mid-to-large GMV tiers ({adopters} of 220 simulated respondents, {100*adopters/len(brand):.0f}%, "
    "already cross the 'likely adopter' threshold) — abandonment rate is a stronger, more actionable adoption "
    "signal than firmographic size alone.\n"
    "- **For pricing**: average predicted willingness-to-pay "
    f"(\\${avg_wtp:.0f}/month) sits above CartIQ's Growth-tier price point (\\$499/month, per the product's own "
    "pricing) — current entry pricing has headroom, and Tab 5's VIF-driven segmentation by GMV tier supports "
    "a distinct higher tier for the largest brands rather than one flat price."
)

st.divider()
st.subheader("4. Sustainability verdict")
st.success(
    f"**Yes — conditionally.** The signal is real (abandonment rate predicts adoption interest, "
    f"r={corr_adopt_abandon:.2f}), pricing has room (avg predicted WTP \\${avg_wtp:.0f}/mo vs. \\$499/mo entry "
    "price), and the underlying mechanism — score, intervene, measure lift via a randomized holdout — runs "
    "end-to-end on live infrastructure, not just on paper. The condition is Limitation #1 below: this verdict "
    "rests on a simulated survey, not a completed one."
)

st.divider()
st.subheader("5. Honest limitations")
diagnostic(
    "Simulated survey data",
    "The entire Business Intelligence side (Tabs 5, 6, and half of 2, 3, 7, 8) runs on a 220-row *simulated* "
    "B2B survey — no real brand has actually been surveyed yet. Every adoption/WTP finding above is a "
    "hypothesis to validate with a real survey, not a conclusion already reached.",
)
diagnostic(
    "Correlation, not causation",
    "Both datasets are observational. 'Cart abandonment rate predicts adoption interest' describes an "
    "association found by the models in Tabs 4-6 — it does not establish that reducing abandonment would "
    "cause a brand to adopt CartIQ, only that the two move together in this data.",
)
diagnostic(
    "Model performance gap on shopper conversion",
    "KNN's cross-validated recall on the shopper conversion task collapses to near-zero (Tab 4) — a reminder "
    "that even within one business question, algorithm choice materially changes what the model can detect, "
    "and the 'best' algorithm differs by dataset (Random Forest/XGBoost win on the imbalanced shopper data; "
    "all four models perform comparably well on the cleaner, more balanced brand-adoption data).",
)
