import pandas as pd
import plotly.express as px
import streamlit as st

from data_utils import clean_ecommerce, load_brand_survey
from style import inject_css, insight, style_layout

st.set_page_config(page_title="Diagnostic Analytics", page_icon="🔍", layout="wide")
inject_css()

st.markdown('<div class="dash-title">🔍 Diagnostic Analytics</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="dash-sub">Why the patterns in the descriptive tab exist — and where the two business '
    'questions overlap.</div>',
    unsafe_allow_html=True,
)

df, _ = clean_ecommerce()
brand = load_brand_survey()

st.subheader("🛍️ Shopper side: where and why conversion breaks down")

col1, col2 = st.columns(2)
with col1:
    heat = df.pivot_table(index="Traffic_Source", columns="Device_Type", values="Converted_bin", aggfunc="mean") * 100
    fig = px.imshow(heat.round(1), text_auto=True, color_continuous_scale="Blues", aspect="auto")
    st.plotly_chart(style_layout(fig, "Conversion Rate (%): Traffic × Device", height=380), use_container_width=True)
with col2:
    heat2 = df.pivot_table(index="Product_Category", columns="Traffic_Source", values="Converted_bin", aggfunc="mean") * 100
    fig2 = px.imshow(heat2.round(1), text_auto=True, color_continuous_scale="Blues", aspect="auto")
    st.plotly_chart(style_layout(fig2, "Conversion Rate (%): Category × Traffic", height=380), use_container_width=True)

best_combo = heat.stack().idxmax()
insight(f"<b>{best_combo[0]} + {best_combo[1]}</b> is the best-converting traffic×device combination "
        f"({heat.stack().max():.1f}%) — a concrete channel/device pairing to prioritize in ad spend, "
        f"not just 'the best channel' in isolation.")

st.markdown("#### Discount impact")
disc_conv = df.groupby("Discount_Code_Used")["Converted_bin"].mean() * 100
disc_aov = df[df["Converted_bin"] == 1].groupby("Discount_Code_Used")["Order_Value"].mean()
c1, c2 = st.columns(2)
c1.metric("Conversion lift from discount", f"{disc_conv.get('Yes', 0) - disc_conv.get('No', 0):+.1f} pp")
c2.metric("AOV difference (discounted vs not)", f"${disc_aov.get('Yes', 0) - disc_aov.get('No', 0):+.0f}")
insight("A modest conversion lift with little-to-no AOV difference is the same finding CartIQ's own "
        "product is built around: blanket discounting is a margin cost with limited upside — which is "
        "exactly why the live product only discounts low-score, high-cart-value sessions with a holdout "
        "control, rather than discounting everyone.")

st.divider()
st.subheader("🧠 Brand side: does mentor/tool sophistication moderate the outcome?")

col3, col4 = st.columns(2)
with col3:
    fig3 = px.box(brand, x="checkout_pain_point", y="willingness_to_pay_usd", color="checkout_pain_point",
                  color_discrete_sequence=px.colors.qualitative.Set2)
    fig3.update_layout(showlegend=False)
    st.plotly_chart(style_layout(fig3, "WTP by Checkout Pain Point", height=400), use_container_width=True)
with col4:
    fig4 = px.scatter(brand, x="current_tool_sophistication", y="willingness_to_pay_usd", color="adoption_likelihood",
                       color_continuous_scale="Viridis", trendline="ols")
    st.plotly_chart(style_layout(fig4, "Current Tool Sophistication vs. WTP", height=400), use_container_width=True)

insight("Brands with lower current-tool sophistication (i.e. using no analytics/recovery tool at all) show "
        "no clear WTP penalty — they're not paying less because they're unsophisticated, they're paying "
        "based on pain (abandonment rate) and scale (GMV) instead. This matters for sales: an unsophisticated "
        "brand is not automatically a low-value lead.")

st.divider()
st.subheader("🔗 Where the two business questions overlap")
insight("<b>Cart abandonment / friction is the thread connecting both questions.</b> On the shopper side "
        "(Tab 4), engagement and cart signals drive conversion. On the brand side (Tab 6), cart abandonment "
        "rate is the strongest predictor of adoption interest. The same underlying problem — checkout "
        "friction — is what CartIQ's shopper-scoring product fixes for a brand's own customers, and what "
        "makes a brand want to buy CartIQ in the first place.")
