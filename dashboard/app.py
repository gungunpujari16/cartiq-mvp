"""
CartIQ Dashboard -- landing page. The actual content lives in pages/, which
Streamlit auto-populates into the sidebar nav from this file's directory.
"""
import streamlit as st

from style import inject_css

st.set_page_config(page_title="CartIQ", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")
inject_css()

st.sidebar.title("🧠 CartIQ")
st.sidebar.caption("Pick a page above ⬆")

st.markdown('<div class="dash-title">CartIQ</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="dash-sub">A conversion-intelligence plugin for e-commerce brands, plus the sales '
    'intelligence CartIQ uses on itself. Two dashboards, two different questions:</div>',
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)
with col1:
    st.subheader("🛍️ Shopper Insights")
    st.markdown(
        "**Question:** will *this shopper*, on *one brand's own storefront*, buy right now?\n\n"
        "**Who uses it:** the brand itself, to understand and act on its own customers.\n\n"
        "**What's in it:** live purchase-intent scoring (XGBoost), behavioral customer segments "
        "(K-Means), checkout funnel diagnostics, channel performance, discount A/B testing, and an "
        "interpretable Decision Tree + Association Rules comparison for each.\n\n"
        "**Data:** real session events captured by the CartIQ JS snippet."
    )
with col2:
    st.subheader("🧠 Business Intelligence")
    st.markdown(
        "**Question:** will *this other brand* buy CartIQ itself, and how much would they pay?\n\n"
        "**Who uses it:** CartIQ's own sales/go-to-market team.\n\n"
        "**What's in it:** adoption-likelihood prediction (Random Forest), willingness-to-pay "
        "regression (Ridge), brand segmentation (K-Means), association rules on what drives high "
        "WTP (Apriori), and company-size inference (Decision Tree).\n\n"
        "**Data:** ⚠️ simulated B2B survey responses (N=220) -- no real survey has been run yet."
    )

st.divider()
st.caption(
    "Every chart on both pages is followed by a blue insight box (what the data shows) and, where a "
    "model produced the chart, an amber methodology box (which algorithm, why it was chosen over "
    "alternatives, and what it calculates) -- phrased to match the CartIQ BRD's own algorithm "
    "rationale (Section 4)."
)
