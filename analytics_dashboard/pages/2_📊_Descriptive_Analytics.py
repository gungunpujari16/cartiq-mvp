import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from data_utils import clean_ecommerce, load_brand_survey
from style import PALETTE, diagnostic, inject_css, insight, style_layout

st.set_page_config(page_title="Descriptive Analytics", page_icon="📊", layout="wide")
inject_css()

st.markdown('<div class="dash-title">📊 Descriptive Analytics</div>', unsafe_allow_html=True)

df, _ = clean_ecommerce()
brand = load_brand_survey()

tab1, tab2 = st.tabs(["🛍️ Shopper Conversion", "🧠 Brand Adoption"])

# ── Shopper side ─────────────────────────────────────────────────────────
with tab1:
    st.sidebar.header("Shopper filters")
    traffic_f = st.sidebar.multiselect("Traffic Source", sorted(df["Traffic_Source"].unique()))
    device_f = st.sidebar.multiselect("Device Type", sorted(df["Device_Type"].unique()))
    category_f = st.sidebar.multiselect("Product Category", sorted(df["Product_Category"].unique()))

    dff = df.copy()
    if traffic_f:
        dff = dff[dff["Traffic_Source"].isin(traffic_f)]
    if device_f:
        dff = dff[dff["Device_Type"].isin(device_f)]
    if category_f:
        dff = dff[dff["Product_Category"].isin(category_f)]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sessions", f"{len(dff):,}")
    c2.metric("Conversion Rate", f"{100*dff['Converted_bin'].mean():.1f}%")
    c3.metric("Cart Add Rate", f"{100*dff['Cart_bin'].mean():.1f}%")
    c4.metric("Avg Order Value", f"${dff.loc[dff['Converted_bin']==1,'Order_Value'].mean():.0f}")

    st.markdown("#### Conversion cross-tabs")
    cross_dim = st.selectbox(
        "Cross-tabulate Converted against:",
        ["Traffic_Source", "Device_Type", "Product_Category", "Time_of_Day", "Payment_Method", "Return_Customer"],
    )
    ct = dff.groupby(cross_dim)["Converted_bin"].agg(["mean", "count"]).reset_index()
    ct.columns = [cross_dim, "conversion_rate", "sessions"]
    ct["conversion_rate"] = (ct["conversion_rate"] * 100).round(1)
    fig = px.bar(ct.sort_values("conversion_rate", ascending=False), x=cross_dim, y="conversion_rate",
                 color="conversion_rate", color_continuous_scale="Blues", text="conversion_rate")
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(style_layout(fig, f"Conversion Rate by {cross_dim}"), use_container_width=True)

    st.markdown("#### Correlation heatmap (numeric features)")
    numeric_cols = ["Age", "Time_Spent_on_Site", "Pages_Viewed", "Session_Count", "Items_in_Cart", "Engagement_Score"]
    corr = dff[numeric_cols + ["Converted_bin"]].corr()
    fig2 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    st.plotly_chart(style_layout(fig2, "Correlation Matrix", height=450), use_container_width=True)

    strongest = corr["Converted_bin"].drop("Converted_bin").abs().idxmax()
    insight(f"<b>{strongest}</b> has the strongest correlation with conversion among numeric features "
            f"(r={corr.loc[strongest, 'Converted_bin']:.2f}) — but note this is still a modest individual "
            f"correlation, consistent with why the production classifier needs 11+ combined features rather "
            f"than any single strong predictor.")

    diagnostic(
        "Leakage reminder",
        "Order_Value isn't in the correlation matrix above on purpose — including it would show a "
        "near-perfect (and meaningless) correlation with Converted, since it's only recorded for "
        "converted sessions in the first place.",
    )

# ── Brand side ───────────────────────────────────────────────────────────
with tab2:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Surveyed Brands", len(brand))
    c2.metric("Avg WTP", f"${brand['willingness_to_pay_usd'].mean():.0f}/mo")
    c3.metric("Likely Adopters (≥7/10)", f"{(brand['adoption_likelihood']>=7).sum()} ({100*(brand['adoption_likelihood']>=7).mean():.0f}%)")
    c4.metric("Avg Abandonment Rate", f"{brand['cart_abandonment_rate'].mean():.1f}%")

    st.markdown("#### WTP / Adoption cross-tabs")
    cross_dim_b = st.selectbox("Cross-tabulate WTP against:", ["industry", "platform", "gmv_tier", "checkout_pain_point", "pricing_preference"])
    ctb = brand.groupby(cross_dim_b)["willingness_to_pay_usd"].agg(["mean", "count"]).reset_index()
    ctb.columns = [cross_dim_b, "avg_wtp", "brands"]
    fig3 = px.bar(ctb.sort_values("avg_wtp", ascending=False), x=cross_dim_b, y="avg_wtp",
                  color="brands", color_continuous_scale="Purples", text=ctb["avg_wtp"].round(0))
    st.plotly_chart(style_layout(fig3, f"Avg Willingness-to-Pay by {cross_dim_b}"), use_container_width=True)

    st.markdown("#### Correlation heatmap (numeric features)")
    num_b = ["cart_abandonment_rate", "current_tool_sophistication", "adoption_likelihood", "willingness_to_pay_usd"]
    corr_b = brand[num_b].corr()
    fig4 = px.imshow(corr_b, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    st.plotly_chart(style_layout(fig4, "Correlation Matrix", height=380), use_container_width=True)

    insight(f"Cart abandonment rate correlates with adoption likelihood at r={corr_b.loc['cart_abandonment_rate','adoption_likelihood']:.2f} "
            f"— brands already in pain are the ones most interested in a fix, a genuine (if simulated) demand signal.")
