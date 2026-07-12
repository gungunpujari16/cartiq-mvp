"""
Shared look-and-feel for both dashboard pages: colors, Plotly styling, and
two callout box types used throughout to make the "what/why" explicit for
grading -- insight boxes (blue, data-driven takeaway) and methodology boxes
(amber, which algorithm + why it was chosen + what it calculates). The
insight-box pattern is reused verbatim from the Phase 0 dashboard
(IPBL/app.py) for visual consistency across both deliverables.
"""
import streamlit as st

NAVY, BLUE, RED, GREEN, AMBER = "#1E3A5F", "#2563EB", "#E74C3C", "#16A34A", "#F59E0B"
PURPLE = "#8B5CF6"
PALETTE = [BLUE, RED, GREEN, AMBER, PURPLE, "#EC4899", "#06B6D4", "#84CC16"]
SEGMENT_COLORS = {"High Intent": GREEN, "Medium": BLUE, "Low": AMBER, "Bounce Risk": RED}


def inject_css():
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

        .insight-box {
            background: #EFF6FF; border-left: 4px solid #2563EB; border-radius: 6px;
            padding: 10px 14px; font-size: 0.84rem; color: #1E3A5F; margin: 8px 0;
        }
        .methodology-box {
            background: #FFFBEB; border-left: 4px solid #F59E0B; border-radius: 6px;
            padding: 10px 14px; font-size: 0.82rem; color: #78350F; margin: 8px 0;
        }
        .methodology-box b { color: #92400E; }
        .sim-badge {
            display: inline-block; background: #FEE2E2; color: #991B1B;
            border-radius: 999px; padding: 3px 12px; font-size: 0.78rem; font-weight: 600;
        }
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


def insight(text: str):
    """Blue callout: a data-driven takeaway computed from what's currently on screen."""
    st.markdown(f'<div class="insight-box">💡 <b>Insight:</b> {text}</div>', unsafe_allow_html=True)


def methodology(algorithm: str, why: str, why_not: str = "", calculates: str = ""):
    """Amber callout: which algorithm, why it was chosen, what it calculates -- phrased to
    mirror the BRD's own 'Why This Algorithm / Why Not Others' table (Section 4) so it's
    directly traceable back to the requirements document."""
    lines = [f"<b>Algorithm:</b> {algorithm}", f"<b>Why this algorithm:</b> {why}"]
    if why_not:
        lines.append(f"<b>Why not others:</b> {why_not}")
    if calculates:
        lines.append(f"<b>What it calculates:</b> {calculates}")
    body = "<br>".join(lines)
    st.markdown(f'<div class="methodology-box">📐 {body}</div>', unsafe_allow_html=True)
