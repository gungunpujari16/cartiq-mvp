import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_utils import clean_ecommerce, encode_shopper_features, encode_shopper_features_with_leakage
from models_classification import MODEL_METHODOLOGY, naive_baseline_accuracy, run_classification_suite
from style import PALETTE, diagnostic, inject_css, insight, methodology, style_layout

st.set_page_config(page_title="Classification: Conversion", page_icon="🎯", layout="wide")
inject_css()

st.markdown('<div class="dash-title">🎯 Classification: Shopper Conversion</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="dash-sub">Target: <code>Converted</code> (Yes/No). Compares four algorithms rather than '
    'picking one, each evaluated the same way — 5-fold stratified cross-validation, not a single train/test '
    'split.</div>',
    unsafe_allow_html=True,
)

df, _ = clean_ecommerce()
X, y = encode_shopper_features(df)
baseline = naive_baseline_accuracy(y)

st.info(
    f"Class balance: **{int(y.sum())} converted / {int((1-y).sum())} not converted** "
    f"({100*y.mean():.1f}% positive). A model that predicts 'not converted' for everyone would score "
    f"**{100*baseline:.1f}% accuracy trivially** — every result below needs to be read against that "
    f"floor, not against 0%."
)

with st.spinner("Training and cross-validating 4 models..."):
    results = st.session_state.get("clf_conv_results")
    if results is None:
        results = run_classification_suite(X, y)
        st.session_state["clf_conv_results"] = results

comparison = pd.DataFrame({
    name: {"Accuracy": r["cv_accuracy"], "Precision": r["cv_precision"], "Recall": r["cv_recall"],
           "F1": r["cv_f1"], "ROC-AUC": r["cv_roc_auc"]}
    for name, r in results.items()
}).T.round(3)
st.markdown("#### Model comparison (mean across 5 folds)")
st.dataframe(comparison.style.highlight_max(axis=0, color="#DCFCE7"), use_container_width=True)

best_auc_model = comparison["ROC-AUC"].idxmax()
insight(f"<b>{best_auc_model}</b> has the highest ROC-AUC ({comparison.loc[best_auc_model, 'ROC-AUC']:.3f}). "
        f"Note KNN's accuracy ({comparison.loc['KNN', 'Accuracy']:.3f}) barely beats — or even trails — the "
        f"{100*baseline:.1f}% naive floor while its recall collapses to {comparison.loc['KNN', 'Recall']:.3f}, "
        f"a textbook case of accuracy being a misleading metric on imbalanced data.")

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
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                     labels=dict(x="Predicted", y="Actual"), x=["Not Converted", "Converted"], y=["Not Converted", "Converted"])
    st.plotly_chart(style_layout(fig, f"{selected} — Confusion Matrix (out-of-fold, all 1,400 rows)", height=380), use_container_width=True)
with col2:
    if r["importances"] is not None:
        imp = r["importances"].reset_index()
        imp.columns = ["feature", "importance"]
        fig2 = px.bar(imp, x="importance", y="feature", orientation="h", color_discrete_sequence=[PALETTE[0]])
        fig2.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(style_layout(fig2, f"{selected} — Top 10 Feature Importances", height=380), use_container_width=True)
    else:
        st.info(f"{selected} doesn't expose feature importances directly (it's a distance-based method, not a tree).")

meth = MODEL_METHODOLOGY[selected]
methodology(selected, why=meth["why"], why_not=meth["why_not"], calculates=meth["calculates"])

st.divider()
st.subheader("Concrete leakage demonstration")
st.write("What happens if `Order_Value` (excluded above) is left in the feature set?")
if st.button("Run the leaky version"):
    with st.spinner("Training with the leaky feature included..."):
        X_leaky, y_leaky = encode_shopper_features_with_leakage(df)
        leaky_results = run_classification_suite(X_leaky, y_leaky)
    leaky_gb = leaky_results["Gradient Boosting (XGBoost)"]
    clean_gb = results["Gradient Boosting (XGBoost)"]
    c1, c2 = st.columns(2)
    c1.metric("XGBoost accuracy — clean features", f"{clean_gb['cv_accuracy']:.3f}")
    c2.metric("XGBoost accuracy — WITH Order_Value", f"{leaky_gb['cv_accuracy']:.3f}", delta=f"{leaky_gb['cv_accuracy']-clean_gb['cv_accuracy']:+.3f}")
    diagnostic(
        "This is the leakage finding, made concrete",
        f"Accuracy jumps to {leaky_gb['cv_accuracy']:.1%} the moment Order_Value is included — not because "
        f"the model got smarter, but because Order_Value is only non-null for rows that already converted. "
        f"The model is reading the answer off a field that's a symptom of the outcome, not a cause of it. "
        f"This is why it's excluded everywhere above.",
    )
