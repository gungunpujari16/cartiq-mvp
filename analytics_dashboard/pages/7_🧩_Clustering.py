import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram

from data_utils import clean_ecommerce, encode_brand_clustering_features, encode_shopper_clustering_features, load_brand_survey
from models_clustering import elbow_and_silhouette, run_clustering
from style import NAVY, PALETTE, diagnostic, inject_css, insight, methodology, style_layout

st.set_page_config(page_title="Clustering", page_icon="🧩", layout="wide")
inject_css()

st.markdown('<div class="dash-title">🧩 Clustering</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="dash-sub">K-Means requires scaled, Euclidean-meaningful continuous features — both segment '
    'views below deliberately exclude demographics/categoricals (mixing in one-hot dummies would distort '
    'distances); a separate demographic segmentation would need a different method (e.g. k-modes or Gower '
    'distance), out of scope here but worth naming as a limitation.</div>',
    unsafe_allow_html=True,
)

view = st.radio("Segment which population?", ["🛍️ Shopper Sessions", "🧠 Brand Survey Respondents"], horizontal=True)

if view.startswith("🛍️"):
    df, _ = clean_ecommerce()
    X = encode_shopper_clustering_features(df)
    id_col = None
    feature_desc = "Time on site, pages viewed, items in cart, session count"
else:
    df = load_brand_survey()
    X = encode_brand_clustering_features(df)
    id_col = df["brand_id"] if "brand_id" in df.columns else None
    feature_desc = "GMV tier, team size, cart abandonment rate, current tool sophistication, willingness-to-pay"
    st.markdown('<span class="sim-badge">⚠️ Simulated survey data (N=220)</span>', unsafe_allow_html=True)

st.caption(f"Clustering features: {feature_desc}")

with st.spinner("Computing elbow + silhouette across k=2..15..."):
    cache_key = f"elbow_{view}"
    es = st.session_state.get(cache_key)
    if es is None:
        es = elbow_and_silhouette(X, 2, 15)
        st.session_state[cache_key] = es

col1, col2 = st.columns(2)
with col1:
    fig = px.line(es, x="k", y="inertia", markers=True)
    st.plotly_chart(style_layout(fig, "Elbow Method (inertia)", height=320), use_container_width=True)
with col2:
    fig2 = px.line(es, x="k", y="silhouette", markers=True, color_discrete_sequence=[PALETTE[2]])
    st.plotly_chart(style_layout(fig2, "Silhouette Score", height=320), use_container_width=True)

best_k = int(es.loc[es["silhouette"].idxmax(), "k"])
insight(f"Silhouette peaks at <b>k={best_k}</b> ({es['silhouette'].max():.3f}). The elbow chart's visual "
        f"'kink' is more ambiguous on its own — this is exactly why both methods are shown together rather "
        f"than picking k from the elbow chart alone.")

k = st.slider("Choose k to visualize", 2, 8, value=min(best_k, 8))
if k != best_k:
    st.caption(f"Note: k={k} selected, but silhouette analysis suggests k={best_k} is the statistically cleanest "
               f"split. k=3 is shown here for comparison against the 3-segment framing (e.g. Budget/Growth/"
               f"Premium) used in the live CartIQ product dashboard, which was chosen for business "
               f"interpretability rather than derived from silhouette analysis.")

with st.spinner("Running K-Means + PCA + hierarchical validation..."):
    result = run_clustering(X, k)

col3, col4 = st.columns(2)
with col3:
    plot_df = X.copy()
    plot_df["pca_x"] = result["pca_coords"][:, 0]
    plot_df["pca_y"] = result["pca_coords"][:, 1]
    plot_df["cluster"] = [f"Cluster {c}" for c in result["kmeans_labels"]]
    fig3 = px.scatter(plot_df, x="pca_x", y="pca_y", color="cluster", color_discrete_sequence=PALETTE)
    ev = result["explained_variance"]
    st.plotly_chart(style_layout(fig3, f"K-Means Clusters — PCA 2D ({100*(ev[0]+ev[1]):.0f}% variance explained)", height=400), use_container_width=True)
    st.caption(f"PC1 explains {100*ev[0]:.1f}% of variance, PC2 explains {100*ev[1]:.1f}% — the 2D view is a "
               f"lossy projection of {X.shape[1]}-dimensional data, shown transparently rather than as pure decoration.")
with col4:
    st.metric("Silhouette at k=" + str(k), f"{result['silhouette']:.3f}")
    st.metric("K-Means vs. Hierarchical agreement (Adjusted Rand Index)", f"{result['agreement_ari']:.3f}")
    if result["agreement_ari"] > 0.5:
        st.success("Strong agreement — two independent clustering methods found broadly the same structure, "
                   "evidence the segments are real rather than an artifact of K-Means' spherical-cluster assumption.")
    else:
        st.warning("Weak agreement between the two methods — treat these segments as exploratory, not conclusive.")

st.markdown("#### Hierarchical clustering (dendrogram) — second-method validation")
st.caption("Truncated to the last 15 merges for legibility (scipy's `truncate_mode='lastp'`); the Adjusted "
           "Rand Index above is computed on the full, untruncated tree over all rows, not this truncated view.")
dendro = scipy_dendrogram(result["linkage_matrix"], no_plot=True, truncate_mode="lastp", p=15)
fig_dendro = go.Figure()
for xs, ys in zip(dendro["icoord"], dendro["dcoord"]):
    fig_dendro.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=NAVY, width=1.5),
                                     hoverinfo="skip", showlegend=False))
fig_dendro.update_xaxes(showticklabels=False, title="Cluster merges (leaf order)")
fig_dendro.update_yaxes(title="Ward linkage distance")
st.plotly_chart(style_layout(fig_dendro, "Hierarchical Clustering Dendrogram (Ward linkage)", height=380),
                 use_container_width=True)
st.caption(f"Cutting this tree at k={k} clusters is the same operation `fcluster` performed to produce the "
           f"hierarchical labels compared against K-Means above (ARI={result['agreement_ari']:.3f}).")

methodology(
    "K-Means (validated with hierarchical clustering)",
    why="Segmentation features here are continuous and numerical (behavioral/firmographic signals) — "
        "K-Means is purpose-built for that; the elbow + silhouette pair validates k rather than guessing it.",
    why_not="A pure categorical/demographic segmentation would need k-modes or Gower distance instead — "
            "named as a limitation, not attempted here, since it's a different method for a different question.",
    calculates=f"Assigns each row to one of {k} clusters by minimizing within-cluster variance in standardized "
               f"feature space; validated against Ward-linkage agglomerative clustering on the same features.",
)
