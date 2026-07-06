"""
CartIQ Analytics Dashboard (PRD Feature 4). Talks only to the CartIQ REST
API -- see api_client.py -- never to the database directly, mirroring the
TRD's Presentation-layer boundary. Visual style intentionally matches the
existing Phase 0 dashboard (IPBL/app.py / utils.py) for a consistent look
across both deliverables.
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from api_client import CartIQClient

st.set_page_config(page_title="CartIQ Dashboard", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

NAVY, BLUE, RED, GREEN, AMBER = "#1E3A5F", "#2563EB", "#E74C3C", "#16A34A", "#F59E0B"
PALETTE = [BLUE, RED, GREEN, AMBER, "#8B5CF6", "#EC4899", "#06B6D4", "#84CC16"]
SEGMENT_COLORS = {"High Intent": GREEN, "Medium": BLUE, "Low": AMBER, "Bounce Risk": RED}

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { background-color: #1E3A5F; }
    [data-testid="stSidebar"] * { color: #E2E8F0 !important; }
    [data-testid="stMetric"] { background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 14px 18px; }
    [data-testid="stMetricValue"] { font-size: 1.55rem; font-weight: 700; color: #1E3A5F; }
    .dash-title { font-size: 1.5rem; font-weight: 700; color: #1E3A5F; border-left: 5px solid #2563EB; padding-left: 12px; }
    .dash-sub { font-size: 0.86rem; color: #64748B; margin-bottom: 16px; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


def style_layout(fig, title: str = "", height: int = 380):
    fig.update_layout(
        font_family="Inter, sans-serif",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=10),
        title=dict(text=title, font=dict(size=14, color=NAVY), x=0.01),
        height=height,
        colorway=PALETTE,
    )
    return fig


# ── Sidebar: connection settings ──────────────────────────────────────────
st.sidebar.title("🧠 CartIQ")
st.sidebar.caption("Connect to your CartIQ backend")

default_key = ""
key_file = Path(__file__).resolve().parent.parent / "backend" / "demo_api_key.txt"
if key_file.exists():
    default_key = key_file.read_text().strip()

api_base = st.sidebar.text_input("API base URL", value=st.session_state.get("api_base", "http://127.0.0.1:8000"))
api_key = st.sidebar.text_input("API key", value=st.session_state.get("api_key", default_key), type="password")
st.session_state["api_base"], st.session_state["api_key"] = api_base, api_key

if st.sidebar.button("🔄 Refresh data"):
    st.cache_data.clear()

if not api_key:
    st.info("Enter your brand's CartIQ API key in the sidebar to load the dashboard. "
            "For the local demo, run `python ml/seed_demo_data.py` in `backend/` and paste the printed key.")
    st.stop()

client = CartIQClient(api_base, api_key)

try:
    me = client.whoami()
except Exception as e:
    st.error(f"Could not reach the CartIQ API at {api_base} -- is the backend running? ({e})")
    st.stop()

brand_id = me["brand_id"]
st.sidebar.success(f"Connected as **{me['name']}**")
st.sidebar.caption(f"brand_id: `{brand_id}`")


@st.cache_data(ttl=15)
def load_all(_client: CartIQClient, brand_id: str):
    return {
        "overview": _client.overview(brand_id),
        "funnel": _client.funnel(brand_id),
        "channels": _client.channels(brand_id),
        "revenue": _client.revenue(brand_id),
        "discounts": _client.discounts_summary(brand_id),
        "segments": _client.segments(brand_id),
        "sessions": _client.sessions(brand_id, page_size=100),
    }


data = load_all(client, brand_id)

st.markdown('<div class="dash-title">CartIQ Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="dash-sub">Live purchase-intent scoring, dynamic discounts, and conversion analytics for your storefront.</div>', unsafe_allow_html=True)

tabs = st.tabs(["Overview", "Funnel", "Channels", "Predictions", "Segments", "Revenue", "Discounts", "Settings"])

# ── Overview ───────────────────────────────────────────────────────────────
with tabs[0]:
    ov = data["overview"]
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Sessions", f"{ov['sessions']:,}")
    c2.metric("Cart Add Rate", f"{ov['cart_add_rate']}%")
    c3.metric("Conversion Rate", f"{ov['conversion_rate']}%")
    c4.metric("Avg Order Value", f"${ov['avg_order_value']:.0f}")
    c5.metric("Return Customer Rate", f"{ov['return_customer_rate']}%")
    c6.metric("Discount Usage", f"{ov['discount_usage_rate']}%")

# ── Funnel ─────────────────────────────────────────────────────────────────
with tabs[1]:
    funnel = pd.DataFrame(data["funnel"])
    if not funnel.empty:
        fig = go.Figure(go.Funnel(y=funnel["stage"], x=funnel["count"], marker={"color": PALETTE}))
        st.plotly_chart(style_layout(fig, "Checkout Funnel"), use_container_width=True)
        funnel_display = funnel.rename(columns={"stage": "Stage", "count": "Count", "drop_off_pct": "Drop-off vs prior stage (%)"})
        st.dataframe(funnel_display, use_container_width=True, hide_index=True)
    else:
        st.info("No sessions yet.")

# ── Channels ───────────────────────────────────────────────────────────────
with tabs[2]:
    channels = pd.DataFrame(data["channels"])
    if not channels.empty:
        fig = px.bar(
            channels.sort_values("conversion_rate", ascending=False),
            x="traffic_source", y="conversion_rate", color="conversion_rate",
            color_continuous_scale="Blues", text=channels["conversion_rate"].apply(lambda v: f"{v}%"),
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_layout(fig, "Conversion Rate by Traffic Source"), use_container_width=True)
        st.dataframe(
            channels.rename(columns={"traffic_source": "Traffic Source", "sessions": "Sessions", "conversions": "Conversions", "conversion_rate": "Conversion Rate (%)"}),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("No sessions yet.")

# ── Predictions (live scoring feed) ─────────────────────────────────────────
with tabs[3]:
    sessions = pd.DataFrame(data["sessions"]["sessions"])
    if not sessions.empty:
        live = sessions[~sessions["is_seed"]]
        st.caption(f"{len(live)} live (non-seed) sessions tracked via the JS snippet, {len(sessions) - len(live)} seeded from historical data.")
        scored = sessions.dropna(subset=["score"]).sort_values("last_seen", ascending=False)
        seg_counts = scored["segment"].value_counts().reindex(list(SEGMENT_COLORS.keys())).fillna(0)
        cols = st.columns(4)
        for col, seg in zip(cols, SEGMENT_COLORS.keys()):
            col.metric(seg, int(seg_counts.get(seg, 0)))

        display_cols = ["id", "device_type", "traffic_source", "cart_value", "score", "segment", "converted", "is_seed", "last_seen"]
        st.dataframe(
            scored[display_cols].head(50).rename(columns={
                "id": "Session", "device_type": "Device", "traffic_source": "Traffic",
                "cart_value": "Cart Value", "score": "Score", "segment": "Segment",
                "converted": "Converted", "is_seed": "Seeded", "last_seen": "Last Seen",
            }),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("No sessions yet -- click through the demo store or run seed_demo_data.py.")

# ── Segments ────────────────────────────────────────────────────────────────
with tabs[4]:
    seg = data["segments"]
    profiles = pd.DataFrame(seg.get("profiles", []))
    points = pd.DataFrame(seg.get("points", []))
    if not profiles.empty:
        st.dataframe(
            profiles.rename(columns={
                "label": "Segment", "size": "Sessions", "avg_engagement": "Avg Engagement",
                "avg_cart_value": "Avg Cart Value ($)", "avg_conversion_rate": "Conversion Rate (%)",
            }),
            use_container_width=True, hide_index=True,
        )
        fig = px.scatter(points, x="pca_x", y="pca_y", color="cluster", color_discrete_sequence=PALETTE)
        st.plotly_chart(style_layout(fig, "Customer Segments -- PCA 2D Projection"), use_container_width=True)
    else:
        st.info("Not enough sessions yet to compute segments (need at least 3).")

# ── Revenue ─────────────────────────────────────────────────────────────────
with tabs[5]:
    rev = data["revenue"]
    by_cat = pd.DataFrame(rev.get("by_category", []))
    order_values = rev.get("order_values", [])
    col1, col2 = st.columns(2)
    with col1:
        if not by_cat.empty:
            fig = px.bar(by_cat, x="product_category", y="avg_order_value", color="orders", color_continuous_scale="Blues")
            st.plotly_chart(style_layout(fig, "Avg Order Value by Category"), use_container_width=True)
        else:
            st.info("No conversions yet.")
    with col2:
        if order_values:
            fig = px.histogram(x=order_values, nbins=30, color_discrete_sequence=[BLUE])
            st.plotly_chart(style_layout(fig, "Order Value Distribution"), use_container_width=True)
        else:
            st.info("No conversions yet.")

# ── Discounts ───────────────────────────────────────────────────────────────
with tabs[6]:
    disc = data["discounts"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Discounts Issued", disc["issued"])
    c2.metric("Holdout (Control) Sessions", disc["holdout"])
    c3.metric("Treated Conversion Rate", f"{disc['treated_conversion_rate']}%")
    c4.metric("Holdout Conversion Rate", f"{disc['holdout_conversion_rate']}%")
    lift = round(disc["treated_conversion_rate"] - disc["holdout_conversion_rate"], 1)
    if disc["issued"] or disc["holdout"]:
        st.markdown(
            f'<div style="background:#EFF6FF;border-left:4px solid #2563EB;border-radius:6px;padding:10px 14px;">'
            f'Incremental lift from discounts vs the 10% holdout control: <strong>{lift:+.1f} pp</strong></div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("No discounts issued yet.")

# ── Settings ────────────────────────────────────────────────────────────────
with tabs[7]:
    st.subheader("Installation snippet")
    st.code(
        f'''<script>
  window.CARTIQ_CONFIG = {{ apiBaseUrl: "{api_base}", apiKey: "{api_key}" }};
</script>
<script src="https://cdn.cartiq.io/cartiq.js" async></script>''',
        language="html",
    )
    st.subheader("Discount engine settings")
    st.json({
        "discount_score_threshold": me["discount_score_threshold"],
        "discount_min_cart_value": me["discount_min_cart_value"],
        "discount_max_pct": me["discount_max_pct"],
    })
    st.caption("Brand-level threshold tuning is stored server-side (Brand table) -- exposed here read-only for the MVP; a real Settings tab would let the brand edit and PATCH these.")
