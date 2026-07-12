"""CartIQ Analytics — landing page. Content lives in pages/."""
import streamlit as st

from style import inject_css

st.set_page_config(page_title="CartIQ Analytics", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
inject_css()

st.sidebar.title("📊 CartIQ Analytics")
st.sidebar.caption("Pick a page above ⬆")

st.markdown('<div class="dash-title">CartIQ Analytics</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="dash-sub">A rigorous, self-contained analysis of two distinct CartIQ business questions '
    '— fully offline, no live backend required. Every technique is applied to a deliberately chosen subset '
    'of variables relevant to the question being asked, justified against at least one alternative, with '
    'results explained honestly (including where a model underperforms).</div>',
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)
with col1:
    st.subheader("🛍️ Business Question 1: Shopper Conversion")
    st.markdown(
        "**What drives a shopper to complete a purchase** on a brand's own storefront?\n\n"
        "**Dataset:** `ecommerce_cleaned.csv` — 1,400 real (synthetic Phase-0) shopper sessions.\n\n"
        "**Covered in:** Tabs 2-4, 7-8 (shopper side)."
    )
with col2:
    st.subheader("🧠 Business Question 2: Brand Adoption")
    st.markdown(
        "**What drives another brand to adopt CartIQ itself, and how much would they pay?**\n\n"
        "**Dataset:** `brand_survey.csv` — 220 simulated B2B survey responses "
        "(⚠️ no real survey has been run yet).\n\n"
        "**Covered in:** Tabs 2-3, 5-8 (brand side)."
    )

st.divider()
st.markdown("### Pages")
st.markdown(
    "1. 🧹 **Data Cleaning** — what was fixed, what was structurally missing (not random), what leakage was found\n"
    "2. 📊 **Descriptive Analytics** — KPIs, cross-tabs, correlations, for both questions\n"
    "3. 🔍 **Diagnostic Analytics** — why the patterns exist, and where the two business questions overlap\n"
    "4. 🎯 **Classification: Conversion** — KNN / Decision Tree / Random Forest / Gradient Boosting, 5-fold CV\n"
    "5. 💰 **Regression: Willingness to Pay** — VIF-justified Ridge / Lasso / Elastic Net, interactive shrinkage\n"
    "6. 🧠 **Classification: Adoption** — same 4-model rigor, different question and features\n"
    "7. 🧩 **Clustering** — K-Means validated against hierarchical clustering, for both shopper and brand segments\n"
    "8. 🔗 **Association Rules** — interactive Apriori, for both shopper behavior and brand traits\n"
    "9. 📋 **Findings & Recommendations** — synthesis, prescriptive actions, and honest limitations"
)
