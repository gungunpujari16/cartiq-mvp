import pandas as pd
import plotly.express as px
import streamlit as st

from data_utils import encode_adoption_features, load_brand_survey
from models_classification import MODEL_METHODOLOGY, naive_baseline_accuracy, run_classification_suite
from style import PALETTE, inject_css, insight, methodology, style_layout

st.set_page_config(page_title="Classification: Adoption", page_icon="🧠", layout="wide")
inject_css()

st.markdown('<div class="dash-title">🧠 Classification: Brand Adoption</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="dash-sub">Target: <code>adoption_likelihood ≥ 7</code> — a different business question and '
    'a different feature set from Tab 4 (this model is never reused across the two questions).</div>',
    unsafe_allow_html=True,
)
st.markdown('<span class="sim-badge">⚠️ Simulated survey data (N=220)</span>', unsafe_allow_html=True)

df = load_brand_survey()
X, y = encode_adoption_features(df)
baseline = naive_baseline_accuracy(y)

st.info(
    f"Class balance: **{int(y.sum())} likely adopters / {int((1-y).sum())} unlikely** "
    f"({100*y.mean():.1f}% positive). Naive 'always predict majority class' floor: **{100*baseline:.1f}%**."
)

with st.spinner("Training and cross-validating 4 models..."):
    results = st.session_state.get("clf_adopt_results")
    if results is None:
        results = run_classification_suite(X, y)
        st.session_state["clf_adopt_results"] = results

comparison = pd.DataFrame({
    name: {"Accuracy": r["cv_accuracy"], "Precision": r["cv_precision"], "Recall": r["cv_recall"],
           "F1": r["cv_f1"], "ROC-AUC": r["cv_roc_auc"]}
    for name, r in results.items()
}).T.round(3)
st.markdown("#### Model comparison (mean across 5 folds)")
st.dataframe(comparison.style.highlight_max(axis=0, color="#DCFCE7"), use_container_width=True)

best_auc_model = comparison["ROC-AUC"].idxmax()
insight(f"<b>{best_auc_model}</b> leads on ROC-AUC ({comparison.loc[best_auc_model, 'ROC-AUC']:.3f}). Unlike "
        f"Tab 4's shopper model, all four algorithms clear the {100*baseline:.1f}% naive floor comfortably here "
        f"— adoption intent has a cleaner signal (dominated by cart_abandonment_rate) than shopper-level "
        f"conversion does.")

st.markdown("#### Inspect one model")
selected = st.selectbox("Model", list(results.keys()), index=list(results.keys()).index(best_auc_model))
r = results[selected]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Accuracy", f"{r['cv_accuracy']:.3f}")
c2.metric("Precision", f"{r['cv_precision']:.3f}")
c3.metric("Recall", f"{r['cv_recall']:.3f}")
c4.metric("F1", f"{r['cv_f1']:.3f}")
c5.metric("ROC-AUC", f"{r['cv_roc_auc']:.3f}")

col1, col2 = st.columns(2)
with col1:
    cm = r["confusion_matrix"]
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Purples",
                     labels=dict(x="Predicted", y="Actual"), x=["Unlikely", "Likely"], y=["Unlikely", "Likely"])
    st.plotly_chart(style_layout(fig, f"{selected} — Confusion Matrix (out-of-fold, all 220 rows)", height=380), use_container_width=True)
with col2:
    if r["importances"] is not None:
        imp = r["importances"].reset_index()
        imp.columns = ["feature", "importance"]
        fig2 = px.bar(imp, x="importance", y="feature", orientation="h", color_discrete_sequence=[PALETTE[4]])
        fig2.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(style_layout(fig2, f"{selected} — Top 10 Feature Importances", height=380), use_container_width=True)
    else:
        st.info(f"{selected} doesn't expose feature importances directly.")

meth = MODEL_METHODOLOGY[selected]
methodology(selected, why=meth["why"], why_not=meth["why_not"], calculates=meth["calculates"])

st.divider()
insight("<b>Cross-question finding:</b> if cart_abandonment_rate tops the feature-importance chart above "
        "(check the Random Forest / XGBoost view), that's the same signal Tab 4 identifies as a top driver "
        "of shopper conversion — the strongest thread connecting both business questions (see Tab 3 and Tab 9).")
