import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from data_utils import encode_wtp_features, load_brand_survey
from models_regression import compute_vif, fit_regression_suite
from style import PALETTE, diagnostic, inject_css, insight, methodology, style_layout

st.set_page_config(page_title="Regression: Willingness to Pay", page_icon="💰", layout="wide")
inject_css()

st.markdown('<div class="dash-title">💰 Regression: Willingness to Pay</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="dash-sub">Target: <code>log(1 + willingness_to_pay_usd)</code>. VIF is computed <b>first</b> '
    '— it\'s the actual justification for reaching for regularization at all.</div>',
    unsafe_allow_html=True,
)
st.markdown('<span class="sim-badge">⚠️ Simulated survey data (N=220)</span>', unsafe_allow_html=True)

df = load_brand_survey()
X, y = encode_wtp_features(df)

st.subheader("Step 1 — Variance Inflation Factor")
vif = compute_vif(X)
col1, col2 = st.columns([1, 1])
with col1:
    st.dataframe(vif, use_container_width=True, hide_index=True)
with col2:
    fig = px.bar(vif, x="VIF", y="feature", orientation="h", color="VIF", color_continuous_scale="Oranges")
    fig.add_vline(x=5, line_dash="dash", line_color="#E74C3C", annotation_text="VIF=5 (watch threshold)")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
    st.plotly_chart(style_layout(fig, "VIF by Feature", height=320), use_container_width=True)

top_vif = vif.iloc[0]
diagnostic(
    "Multicollinearity finding",
    f"<b>{top_vif['feature']}</b> has VIF={top_vif['VIF']:.2f} — above the usual VIF>5 watch threshold. "
    f"GMV tier, order volume, and team size are correlated by construction (a bigger brand tends to have "
    f"more orders and a bigger team) — this is the actual reason Ridge/Lasso/Elastic Net are used below "
    f"instead of plain OLS, not a checklist formality.",
)

st.subheader("Step 2 — OLS vs. Regularized Models")
c1, c2 = st.columns(2)
alpha = c1.select_slider(
    "Regularization strength (α, log scale)",
    options=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
    value=0.03,
)
l1_ratio = c2.slider("Elastic Net L1 ratio (0=Ridge-like, 1=Lasso-like)", 0.0, 1.0, 0.5, 0.05)

results = fit_regression_suite(X, y, alpha=alpha, l1_ratio=l1_ratio)

metrics_df = pd.DataFrame({
    name: {"R²": r["r2"], "RMSE ($/mo)": r["rmse_usd"], "Zero coefficients": r["n_zero_coefs"], "‖coef‖₂": r["coef_l2_norm"]}
    for name, r in results.items()
}).T.round(3)
st.dataframe(metrics_df, use_container_width=True)

coef_cols = st.columns(4)
for col, (name, r) in zip(coef_cols, results.items()):
    with col:
        top_coefs = r["coefficients"].head(6).reset_index()
        top_coefs.columns = ["feature", "coefficient"]
        fig = px.bar(top_coefs, x="coefficient", y="feature", orientation="h",
                     color=top_coefs["coefficient"] > 0, color_discrete_map={True: PALETTE[2], False: PALETTE[1]})
        fig.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(style_layout(fig, name, height=280), use_container_width=True)

lasso_zeros = results["Lasso"]["n_zero_coefs"]
ridge_norm = results["Ridge"]["coef_l2_norm"]
ols_norm = results["OLS (no regularization)"]["coef_l2_norm"]
if results["Lasso"]["r2"] < 0:
    insight(f"At α={alpha}, Lasso has collapsed to R²={results['Lasso']['r2']:.2f} — worse than predicting "
            f"the mean — with {lasso_zeros} of {len(X.columns)} coefficients zeroed. This is the concrete "
            f"version of the BRD's own warning: Lasso can arbitrarily zero out correlated features that are "
            f"jointly important, destroying signal rather than cleanly selecting it. Try lowering α with the "
            f"slider to see Lasso recover.")
else:
    insight(f"Ridge's coefficient norm shrank from {ols_norm:.2f} (OLS) to {ridge_norm:.2f} at α={alpha}, while "
            f"keeping all {len(X.columns)} features — Lasso instead zeroed {lasso_zeros} of them. This is the "
            f"actual difference in how the two regularizers handle the correlated GMV/orders/team-size cluster "
            f"flagged in Step 1.")

methodology(
    "Ridge Regression (L2)",
    why="Survey features are highly correlated (Step 1's VIF finding) — Ridge penalizes large coefficients "
        "and stabilizes the model without zeroing out any feature, keeping the joint signal from correlated features.",
    why_not="Lasso (L1) zeros out coefficients entirely — with GMV/orders/team-size correlated, it would "
            "arbitrarily drop some of them even though all three carry real signal (see the Lasso collapse above).",
    calculates="log(1 + monthly WTP in USD), back-transformed to dollars for the RMSE metric above.",
)

st.subheader("Coefficient shrinkage across α")
alphas_sweep = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
sweep_rows = []
for a in alphas_sweep:
    rr = fit_regression_suite(X, y, alpha=a, l1_ratio=l1_ratio)
    sweep_rows.append({"alpha": a, "Ridge ‖coef‖₂": rr["Ridge"]["coef_l2_norm"], "Lasso zero count": rr["Lasso"]["n_zero_coefs"], "Ridge R²": rr["Ridge"]["r2"], "Lasso R²": rr["Lasso"]["r2"]})
sweep = pd.DataFrame(sweep_rows)
fig_sweep = px.line(sweep, x="alpha", y=["Ridge ‖coef‖₂", "Lasso zero count"], log_x=True, markers=True)
st.plotly_chart(style_layout(fig_sweep, "Shrinkage as α increases (log scale)", height=350), use_container_width=True)
st.caption("As α increases: Ridge's coefficient norm shrinks smoothly toward zero (all features kept, just smaller); "
           "Lasso's zero-count climbs in steps (features dropped entirely, one at a time).")
