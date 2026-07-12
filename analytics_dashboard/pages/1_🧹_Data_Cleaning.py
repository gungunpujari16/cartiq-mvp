import pandas as pd
import streamlit as st

from data_utils import clean_ecommerce, load_brand_survey, load_ecommerce_raw
from style import diagnostic, inject_css

st.set_page_config(page_title="Data Cleaning", page_icon="🧹", layout="wide")
inject_css()

st.markdown('<div class="dash-title">🧹 Data Cleaning</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="dash-sub">Every fix below is shown with a before/after count and a stated reason — '
    'including the fixes that were deliberately *not* made (leakage fields left alone on purpose).</div>',
    unsafe_allow_html=True,
)

st.subheader("Dataset 1 — Shopper Sessions (`ecommerce_cleaned.csv`)")

raw = load_ecommerce_raw()
clean, log = clean_ecommerce()

c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{len(raw):,}", delta="0 dropped")
c2.metric("Columns (raw → after derived features)", f"{raw.shape[1]} → {clean.shape[1]}")
c3.metric("Cells requiring a decision", f"{int(raw.isna().sum().sum()):,} NaN cells found")

st.markdown("#### Cleaning log")
for entry in log:
    with st.expander(entry["step"], expanded=("leakage" in entry["step"].lower() or "structural" in entry["step"].lower())):
        st.write(entry["detail"])

diagnostic(
    "Leakage risk",
    "Order_Value is non-null <i>only</i> for converted sessions (1,150 NaN rows = exactly the 1,150 "
    "'Converted == No' rows). It is never imputed and is excluded from every feature set used to predict "
    "Converted (Tab 4) — including it would let a model 'predict' the outcome from a field that only "
    "exists because the outcome already happened.",
)
diagnostic(
    "Structural missingness (not random)",
    "Items_in_Cart is NaN exactly when Added_to_Cart == 'No' (663 of 663 — an exact match, not a "
    "coincidence). These shoppers never added anything, so 0 is the correct value, not a measurement "
    "gap — median-imputing this field would have invented cart contents that never existed.",
)

st.divider()
st.subheader("Dataset 2 — B2B Brand Survey (`brand_survey.csv`, simulated)")
brand = load_brand_survey()
c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{len(brand):,}")
c2.metric("Columns", brand.shape[1])
c3.metric("Missing cells", int(brand.isna().sum().sum()))

st.markdown('<span class="clean-badge">✓ No missing values, no casing/whitespace issues</span>', unsafe_allow_html=True)
st.caption(
    "This dataset is synthetically generated (see `generate_brand_survey.py`) to mirror the survey design "
    "in the CartIQ BRD Section 7 — it's clean by construction, since no real B2B survey has been run yet. "
    "A real survey export would need the same casing/whitespace/missingness checks applied to Dataset 1 "
    "above, which is why that checking logic exists and is documented rather than assumed unnecessary."
)

diagnostic(
    "Leakage risk (brand side)",
    "company_size is derived directly from team_size in the data generator. Every model that predicts "
    "company_size (Tab 8's context, referenced from the classification methodology) excludes team_size "
    "as a feature — including it would produce a trivial, meaningless 100% accuracy.",
)

with st.expander("Raw sample rows (shopper sessions)"):
    st.dataframe(raw.head(10), use_container_width=True)
with st.expander("Raw sample rows (brand survey)"):
    st.dataframe(brand.head(10), use_container_width=True)
