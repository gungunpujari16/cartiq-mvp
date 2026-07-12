"""
CartIQ Analytics Dashboard (PRD Feature 4). Talks only to the CartIQ REST
API -- see api_client.py -- never to the database directly, mirroring the
TRD's Presentation-layer boundary.

Answers: "will THIS shopper (of one brand's own storefront) buy right now,
and what should we do about it." Every insight below is about one brand's
own customers -- contrast with the Business Intelligence page, which
answers a different question about OTHER brands buying CartIQ itself.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.tree import DecisionTreeClassifier, export_text

from api_client import CartIQClient
from style import PALETTE, SEGMENT_COLORS, inject_css, insight, methodology, style_layout

st.set_page_config(page_title="CartIQ Dashboard", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")
inject_css()

# ── Sidebar: connection settings ──────────────────────────────────────────
st.sidebar.title("🧠 CartIQ")
st.sidebar.caption("Connect to your CartIQ backend")

default_key = ""
key_file = Path(__file__).resolve().parent.parent.parent / "backend" / "demo_api_key.txt"
if key_file.exists():
    default_key = key_file.read_text().strip()

try:
    default_api_base = st.secrets.get("API_BASE_URL", "http://127.0.0.1:8000")
except Exception:
    default_api_base = "http://127.0.0.1:8000"

api_base = st.sidebar.text_input("API base URL", value=st.session_state.get("api_base", default_api_base))
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

st.markdown('<div class="dash-title">CartIQ Analytics -- Shopper Insights</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="dash-sub">What your own shoppers are doing right now, why, and what to do about it. '
    'Every chart below is followed by a plain-English insight and, where a model produced it, the '
    'methodology behind that model.</div>',
    unsafe_allow_html=True,
)

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

    if ov["sessions"]:
        gap = round(ov["cart_add_rate"] - ov["conversion_rate"], 1)
        insight(
            f"{ov['cart_add_rate']}% of sessions add something to cart, but only {ov['conversion_rate']}% "
            f"convert -- a {gap} point gap between intent and completion. That gap is where checkout "
            f"friction, not demand, is costing revenue. See the Funnel tab for exactly where it breaks."
        )
        st.caption("Descriptive analytics -- these are direct aggregations of your session data, not model output.")

# ── Funnel ─────────────────────────────────────────────────────────────────
with tabs[1]:
    funnel = pd.DataFrame(data["funnel"])
    if not funnel.empty:
        fig = go.Figure(go.Funnel(y=funnel["stage"], x=funnel["count"], marker={"color": PALETTE}))
        st.plotly_chart(style_layout(fig, "Checkout Funnel"), use_container_width=True)
        funnel_display = funnel.rename(columns={"stage": "Stage", "count": "Count", "drop_off_pct": "Drop-off vs prior stage (%)"})
        st.dataframe(funnel_display, use_container_width=True, hide_index=True)

        worst = funnel.iloc[1:].loc[funnel.iloc[1:]["drop_off_pct"].idxmax()]
        insight(f"The biggest drop-off is at <b>{worst['stage']}</b>, losing {worst['drop_off_pct']}% of shoppers "
                f"who reached that point. Fixing this single stage has more leverage than any top-of-funnel "
                f"traffic campaign.")
        st.caption("Descriptive analytics -- stage-wise aggregation of session counts, no model involved.")
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
        best = channels.loc[channels["conversion_rate"].idxmax()]
        worst = channels.loc[channels["conversion_rate"].idxmin()]
        insight(f"<b>{best['traffic_source']}</b> converts at {best['conversion_rate']}%, your best channel -- "
                f"<b>{worst['traffic_source']}</b> converts at only {worst['conversion_rate']}%. Shifting spend "
                f"toward {best['traffic_source']} and away from {worst['traffic_source']} is a budget "
                f"reallocation, not new spend.")

        # ── Association Rules (Apriori) on shopper behavior ──────────────
        sessions_df = pd.DataFrame(data["sessions"]["sessions"])
        with st.expander("📎 What behavior combinations associate with conversion? (Association Rules)"):
            if len(sessions_df) >= 20 and sessions_df["converted"].nunique() > 1:
                basket = pd.concat([
                    pd.get_dummies(sessions_df["device_type"], prefix="Device"),
                    pd.get_dummies(sessions_df["traffic_source"], prefix="Traffic"),
                    pd.get_dummies(sessions_df["product_category"].fillna("Unknown"), prefix="Category"),
                    sessions_df["converted"].rename("Converted"),
                ], axis=1).astype(bool)
                frequent = apriori(basket, min_support=0.04, use_colnames=True)
                if not frequent.empty:
                    rules = association_rules(frequent, metric="lift", min_threshold=1.2)
                    rules = rules[rules["consequents"].apply(lambda s: "Converted" in s)]
                    rules = rules.sort_values("lift", ascending=False).head(8).copy()
                    if not rules.empty:
                        rules["antecedents"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
                        rules["consequents"] = rules["consequents"].apply(lambda s: ", ".join(sorted(s)))
                        st.dataframe(
                            rules[["antecedents", "consequents", "support", "confidence", "lift"]].round(3)
                            .rename(columns={"antecedents": "If a session has...", "consequents": "...then",
                                              "support": "Support", "confidence": "Confidence", "lift": "Lift"}),
                            use_container_width=True, hide_index=True,
                        )
                        top = rules.iloc[0]
                        insight(f"Sessions with <b>{top['antecedents']}</b> are {top['lift']:.1f}x more likely to "
                                f"convert than average (confidence {top['confidence']:.0%}) -- a concrete targeting "
                                f"signal for ad spend or on-site personalization.")
                    else:
                        st.info("No rules met the lift threshold yet -- needs more session volume.")
                else:
                    st.info("No frequent itemsets found yet -- needs more session volume.")
            else:
                st.info("Need at least 20 sessions with a mix of converted/non-converted to mine association rules.")

            methodology(
                "Apriori Association Rules",
                why="Discovers non-linear behavior combinations (e.g. {Mobile + Social Media} -> Converted) that "
                    "a bar chart or regression would miss -- it finds co-occurrence patterns directly.",
                why_not="Regression only shows linear, one-variable-at-a-time effects and can't surface "
                        "interaction patterns between categorical combinations the way ARM does.",
                calculates="Support (how often the combination appears), Confidence (P(conversion | combination)), "
                           "and Lift (how much more likely conversion is given the combination, vs the base rate).",
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
        bounce_pct = round(100 * seg_counts.get("Bounce Risk", 0) / max(len(scored), 1), 1)
        insight(f"{bounce_pct}% of currently scored sessions are flagged <b>Bounce Risk</b> -- these are the "
                f"sessions the discount engine will target if their cart value clears your minimum threshold. "
                f"See the Discounts tab for how effective that intervention has been.")

        methodology(
            "XGBoost Gradient Boosting Classifier",
            why="Highest accuracy on tabular event data of this kind; native handling of missing features "
                "common in partial sessions; fast (&lt;5ms) inference needed for a real-time score.",
            why_not="KNN is too slow at production scale; a single Decision Tree underfits (see comparison "
                    "below); neural nets need far more data than a per-brand session volume typically provides.",
            calculates="Purchase probability (0-100) from 11 session-level features (time on site, pages "
                       "viewed, cart value, device, traffic source, etc.) -- see the model card in "
                       "backend/README for the full feature list and measured AUC (0.77 on held-out test data).",
        )

        with st.expander("🌳 Compare: Decision Tree (interpretable alternative)"):
            feat_df = sessions.dropna(subset=["converted"]).copy()
            if len(feat_df) >= 20 and feat_df["converted"].nunique() > 1:
                X = pd.DataFrame({
                    "time_on_site": feat_df["time_on_site"].fillna(0),
                    "pages_viewed": feat_df["pages_viewed"].fillna(0),
                    "cart_value": feat_df["cart_value"].fillna(0),
                    "items_in_cart": feat_df["items_in_cart"].fillna(0),
                    "return_customer": feat_df["return_customer"].astype(int),
                    "discount_used": feat_df["discount_used"].astype(int),
                })
                X = pd.concat([X, pd.get_dummies(feat_df["device_type"], prefix="device")], axis=1)
                y = feat_df["converted"].astype(int)

                tree = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42)
                tree.fit(X, y)
                tree_acc = round((tree.predict(X) == y).mean(), 3)

                c1, c2 = st.columns(2)
                c1.metric("Decision Tree accuracy (this brand's data)", f"{tree_acc:.1%}")
                c2.metric("XGBoost AUC (production model, held-out test)", "0.77")
                st.code(export_text(tree, feature_names=list(X.columns), max_depth=3), language="text")

                methodology(
                    "Decision Tree Classifier (comparison only, not used in production)",
                    why="Every split is a human-readable rule -- useful for explaining to a non-technical "
                        "stakeholder exactly why a session scored the way it did.",
                    why_not="Trees overfit easily and their accuracy trails ensemble methods like XGBoost on "
                            "the same data; that's the trade being shown here: interpretability vs accuracy.",
                    calculates="Same purchase-probability question as the production model, but via a single "
                               "shallow tree (max depth 3) instead of 300 boosted trees.",
                )
            else:
                st.info("Need at least 20 sessions with both converted and non-converted examples to fit a comparison tree.")
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

        top_seg = profiles.loc[profiles["avg_conversion_rate"].idxmax()]
        insight(f"<b>{top_seg['label']}</b> converts at {top_seg['avg_conversion_rate']}%, your highest-value "
                f"segment ({top_seg['size']} sessions) -- worth a dedicated retention or upsell flow rather "
                f"than treating all shoppers identically.")

        methodology(
            "K-Means Clustering (k=3)",
            why="Segmentation features here (engagement, cart value) are continuous numerical variables -- "
                "K-Means is purpose-built for that, computationally efficient, and produces clean "
                "PCA-visualizable clusters.",
            why_not="Latent Class Analysis (LCA) is the right tool when segmentation variables are "
                    "categorical; ours are continuous, making K-Means the statistically correct choice here.",
            calculates="Assigns each session to 1 of 3 behavioral clusters (labeled by engagement level: "
                       "At-Risk / Engaged / Champions) based on time on site, pages viewed, and cart value, "
                       "each standardized before clustering.",
        )
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
            fig = px.histogram(x=order_values, nbins=30, color_discrete_sequence=["#2563EB"])
            st.plotly_chart(style_layout(fig, "Order Value Distribution"), use_container_width=True)
        else:
            st.info("No conversions yet.")

    if not by_cat.empty:
        top_cat = by_cat.loc[by_cat["avg_order_value"].idxmax()]
        insight(f"<b>{top_cat['product_category']}</b> has the highest average order value "
                f"(${top_cat['avg_order_value']:.0f}) across {int(top_cat['orders'])} orders -- your highest "
                f"revenue-per-conversion category, worth prioritizing in merchandising and ad creative.")
        st.caption("Descriptive analytics -- direct aggregation of converted order values, no model involved.")

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
        insight(f"Discounted sessions convert {lift:+.1f} percentage points differently than the untreated "
                f"10% holdout control -- that delta, not the raw discount-usage rate, is the true incremental "
                f"revenue impact of the discount engine.")
        methodology(
            "Randomized Holdout (A/B Controlled Experiment)",
            why="The only way to measure true incremental lift, not correlation -- 10% of otherwise-eligible "
                "sessions are randomly withheld from receiving a discount so their conversion rate serves as "
                "the counterfactual baseline.",
            calculates="Treated conversion rate minus holdout conversion rate = incremental lift attributable "
                       "to the discount itself, isolated from the fact that low-score sessions convert "
                       "differently anyway.",
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
