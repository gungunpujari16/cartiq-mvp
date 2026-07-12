"""
CartIQ's OWN go-to-market intelligence: will another e-commerce brand buy
CartIQ, and how much would they pay? This is a different question, a
different dataset, and a different audience than the Shopper Insights page
-- that page answers "will this shopper buy" for one brand's own customers;
this page answers "which brands should CartIQ's sales team target" using a
simulated B2B survey (BRD Section 7), per the BRD's own algorithm table
(Section 4: Random Forest / Ridge / K-Means / Apriori / Decision Tree).

No backend involved -- this page loads a bundled synthetic CSV and trains
each model in-page. See generate_brand_survey.py for how the data was made.
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

import bi_models as bi
from style import PALETTE, inject_css, insight, methodology, style_layout

st.set_page_config(page_title="CartIQ Business Intelligence", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")
inject_css()

st.sidebar.title("🧠 CartIQ")
st.sidebar.caption("Business Intelligence")
st.sidebar.markdown(
    '<div style="font-size:0.8rem; line-height:1.5;">This page analyzes CartIQ\'s own prospective '
    'customers (other brands), not shoppers. See the Shopper Insights page for the product '
    'dashboard a brand actually uses.</div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="dash-title">CartIQ Business Intelligence</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="dash-sub">Which prospective brands are likely to buy CartIQ, how much would they pay, and '
    'how should the sales team prioritize outreach -- based on a B2B brand survey (BRD Section 7).</div>',
    unsafe_allow_html=True,
)
st.markdown('<span class="sim-badge">⚠️ Simulated survey data (N=220) -- no real B2B survey has been run yet</span>', unsafe_allow_html=True)
st.write("")

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "brand_survey.csv"
if not DATA_PATH.exists():
    st.error("data/brand_survey.csv not found -- run `python generate_brand_survey.py` in `dashboard/` first.")
    st.stop()

df = pd.read_csv(DATA_PATH)
st.caption(f"{len(df)} simulated brand survey responses loaded.")

sections = st.tabs(["Adoption Prediction", "Willingness to Pay", "Brand Segments", "Association Rules", "Company Size"])

# ── 1. Random Forest -- adoption likelihood ─────────────────────────────
with sections[0]:
    st.subheader("Will a brand subscribe to CartIQ?")
    rf = bi.run_random_forest(df)
    c1, c2, c3 = st.columns(3)
    c1.metric("Model accuracy (held-out test)", f"{rf['accuracy']:.1%}")
    c2.metric("AUC-ROC", f"{rf['auc']:.3f}")
    c3.metric("Predicted likely adopters", f"{rf['predicted_likely_count']} / {rf['total']}")

    fig = px.bar(rf["importances"], x="importance", y="feature", orientation="h", color_discrete_sequence=[PALETTE[0]])
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(style_layout(fig, "Top Features Driving Adoption Prediction"), use_container_width=True)

    top_feat = rf["importances"].iloc[0]
    insight(f"<b>{top_feat['feature'].replace('_', ' ').title()}</b> is the strongest predictor of adoption "
            f"intent -- {rf['predicted_likely_count']} of {rf['total']} surveyed brands are predicted likely "
            f"to adopt CartIQ ({100*rf['predicted_likely_count']/rf['total']:.0f}%), giving the sales team a "
            f"ranked outreach list instead of contacting brands at random.")

    methodology(
        "Random Forest Classification",
        why="Handles mixed categorical and numerical brand features (GMV tier, platform, team size) "
            "natively, gives feature importance for sales targeting, and is robust to class imbalance.",
        why_not="Logistic Regression assumes linear separability between classes; KNN degrades badly on "
                "high-dimensional mixed categorical/numerical data like a brand survey.",
        calculates="P(a brand adopts CartIQ) as a binary classification -- target is adoption_likelihood "
                   "&ge; 7 on the survey's 1-10 NPS-style question (WTP-10).",
    )

# ── 2. Ridge Regression -- willingness to pay ────────────────────────────
with sections[1]:
    st.subheader("How much would they pay?")
    ridge = bi.run_ridge(df)
    c1, c2, c3 = st.columns(3)
    c1.metric("R² (test set)", f"{ridge['r2']:.3f}")
    c2.metric("RMSE", f"${ridge['rmse_usd']:.0f}/mo")
    c3.metric("Avg predicted WTP", f"${ridge['avg_predicted_wtp']:.0f}/mo")

    fig = px.histogram(x=ridge["predicted_wtp"], nbins=30, color_discrete_sequence=[PALETTE[1]])
    st.plotly_chart(style_layout(fig, "Predicted Willingness-to-Pay Distribution ($/mo)"), use_container_width=True)

    fig2 = px.bar(ridge["coefficients"], x="coefficient", y="feature", orientation="h", color_discrete_sequence=[PALETTE[2]])
    fig2.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(style_layout(fig2, "Ridge Coefficients (log-WTP scale)"), use_container_width=True)

    insight(f"Predicted WTP averages ${ridge['avg_predicted_wtp']:.0f}/month -- comparable to the BRD's "
            f"Growth tier ($499/mo), with the model explaining {ridge['r2']:.0%} of the variance in what "
            f"brands say they'd pay. This maps predicted WTP directly onto which pricing tier "
            f"(Starter/Growth/Scale/Enterprise) to lead with in a sales conversation.")

    methodology(
        "Ridge Regression (L2, alpha=1.0)",
        why="Survey features are highly correlated -- GMV, order volume, and team size all co-vary "
            "strongly. Ridge penalizes large coefficients and stabilizes the model without zeroing out "
            "any feature.",
        why_not="Lasso (L1) would zero out one of GMV/orders/team-size even though all three carry real "
                "signal; plain OLS overfits badly with this many one-hot-encoded features on ~220 rows.",
        calculates="log(1 + monthly WTP in USD), back-transformed to dollars for reporting -- log-transformed "
                   "to reduce right-skew from enterprise-tier outliers, per the TRD's own model spec.",
    )

# ── 3. K-Means -- brand segments ─────────────────────────────────────────
with sections[2]:
    st.subheader("Which brands to target first?")
    ridge_for_km = bi.run_ridge(df)
    km = bi.run_kmeans_brands(df, ridge_for_km["predicted_wtp"])
    st.dataframe(
        km["profiles"].rename(columns={"segment": "Segment", "brands": "Brands", "avg_predicted_wtp": "Avg Predicted WTP ($)"}),
        use_container_width=True, hide_index=True,
    )
    fig = px.scatter(km["points"], x="pca_x", y="pca_y", color="segment", color_discrete_sequence=PALETTE,
                      hover_data=["brand_id"])
    st.plotly_chart(style_layout(fig, "Brand Segments -- PCA 2D Projection"), use_container_width=True)

    premium = km["profiles"].iloc[-1]
    insight(f"The <b>{premium['segment']}</b> segment ({int(premium['brands'])} brands) has the highest "
            f"average predicted WTP at ${premium['avg_predicted_wtp']:.0f}/mo -- this is the segment sales "
            f"should prioritize for direct outreach, while lower segments are better served by self-serve "
            f"onboarding.")

    methodology(
        "K-Means Clustering (k=3)",
        why="Predicted WTP and GMV are numerical -- K-Means is purpose-built for numerical segmentation; "
            "the elbow + silhouette method validates k=3 as the natural split.",
        why_not="Latent Class Analysis (LCA) is for purely categorical data; K-Means is faster and more "
                "interpretable for these continuous WTP/GMV features.",
        calculates="Assigns each surveyed brand to Premium / Growth / Budget based on predicted WTP, GMV "
                   "tier, team size, current tool sophistication, and abandonment rate (all standardized).",
    )

# ── 4. Apriori -- association rules ──────────────────────────────────────
with sections[3]:
    st.subheader("What drives high willingness-to-pay?")
    ap = bi.run_apriori(df)
    if not ap["rules"].empty:
        st.dataframe(
            ap["rules"].rename(columns={"antecedents": "If a brand has...", "consequents": "...then",
                                          "support": "Support", "confidence": "Confidence", "lift": "Lift"}),
            use_container_width=True, hide_index=True,
        )
        top = ap["rules"].iloc[0]
        insight(f"Brands with <b>{top['antecedents']}</b> are {top['lift']:.1f}x more likely to fall in the "
                f"High WTP group (confidence {top['confidence']:.0%}) -- a concrete profile for the sales "
                f"team's ideal-customer-profile (ICP) definition, not just a single feature.")
    else:
        st.info("No rules cleared the lift threshold on this data sample.")

    methodology(
        "Apriori Association Rules",
        why="Discovers non-linear feature combinations (e.g. {Shopify + High GMV + Checkout Pain} => High "
            "WTP) that regression misses entirely, since it only estimates one-variable-at-a-time linear "
            "effects.",
        why_not="Regression shows linear effects only; ARM reveals interaction patterns regression cannot "
                "detect, which is exactly the kind of qualitative combination a sales rep can act on.",
        calculates="Support (how often a feature combination appears), Confidence (P(High WTP | combination)), "
                   "and Lift (how much more likely High WTP is given the combination vs the base rate) -- "
                   "filtered to lift > 1.3, close to the BRD's own sustainability threshold of lift > 1.5.",
    )

# ── 5. Decision Tree -- company size ─────────────────────────────────────
with sections[4]:
    st.subheader("Predict company size (when it wasn't asked directly)")
    dt = bi.run_decision_tree(df)
    st.metric("Accuracy (3-class: Small / Medium / Large)", f"{dt['accuracy']:.1%}")
    st.code(dt["tree_text"], language="text")

    insight(f"The tree classifies company size with {dt['accuracy']:.0%} accuracy using only GMV, order "
            f"volume, and industry -- useful when a lead's team size isn't captured, but their GMV tier is "
            f"(e.g. inferred from a public storefront), letting sales still route the lead appropriately.")

    methodology(
        "Decision Tree Classification (max depth 4)",
        why="Produces interpretable rules that are easy to explain to non-technical sales stakeholders, "
            "and performs reasonably on a small survey sample (n~220) where more complex models would "
            "overfit.",
        why_not="Neural networks are overkill for n~220 survey data; Random Forest is less interpretable "
                "for this specific use case, where the point is a readable rule set, not raw accuracy.",
        calculates="Predicted company-size bucket (Small/Medium/Large) from GMV tier, order volume, "
                   "abandonment rate, and industry -- deliberately excludes team_size itself as a feature, "
                   "since company_size is derived from it in the survey and using it would be circular.",
    )
