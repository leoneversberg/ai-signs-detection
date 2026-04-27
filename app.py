import plotly.graph_objects as go
import streamlit as st

from ai_analyzer import analyze_text


# -----------------------------
# Gauge
# -----------------------------
def render_gauge(probability: int):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability,
            number={"suffix": "%", "font": {"size": 26}},
            title={"text": "AI Probability", "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "black"},
                # Colored ranges
                "steps": [
                    {"range": [0, 25], "color": "#22c55e"},  # green
                    {"range": [25, 75], "color": "#eab308"},  # yellow
                    {"range": [75, 100], "color": "#ef4444"},  # red
                ],
                # Optional threshold line
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.75,
                    "value": probability,
                },
            },
        )
    )

    fig.update_layout(height=240, margin=dict(t=40, b=20, l=10, r=10))
    return fig


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="AI Text Analyzer", layout="wide")

st.markdown("### AI Text Detection")
st.caption("Analyze stylistic patterns associated with AI-generated writing,")

# Layout
col_main, col_side = st.columns([2.2, 1], border=True)

# -----------------------------
# LEFT: Main workflow
# -----------------------------
with col_main:
    text_input = st.text_area("Text", height=450, placeholder="Paste your text here...")

    analyze_clicked = st.button("Analyze")

    result_container = st.container()

# -----------------------------
# RIGHT: Status panel
# -----------------------------
with col_side:
    st.subheader("Result")

    gauge_placeholder = st.empty()
    scale_placeholder = st.empty()

# -----------------------------
# Logic
# -----------------------------
if analyze_clicked:
    if not text_input.strip():
        st.warning("Please enter some text.")
        st.stop()

    with st.spinner("Analyzing..."):
        try:
            result = analyze_text(text_input)
        except Exception as e:
            result = {}
            st.error(f"Unexpected error: {e}")

    verdict = result.get("verdict", {})
    probability = verdict.get("ai_probability", 0)
    reasoning = verdict.get("reasoning", "No reasoning provided.")
    rubrics = result.get("rubrics", {}).get("results", {})

    # -----------------------------
    # RIGHT: Status panel
    # -----------------------------
    gauge_placeholder.plotly_chart(render_gauge(probability), width="stretch")

    with scale_placeholder:
        with st.expander("How to interpret this score", expanded=True):
            st.markdown("""
            **AI Probability Scale**

            - **0-20**: Little to no evidence of AI patterns  
            - **21-40**: Weak or sparse signals  
            - **41-60**: Mixed or moderate evidence  
            - **61-80**: Strong evidence across multiple categories  
            - **81-100**: Overwhelming and consistent AI-like patterns  
            """)

    # -----------------------------
    # LEFT: Detailed output
    # -----------------------------
    with result_container:
        st.divider()

        st.subheader("Reasoning")
        st.write(reasoning)

        if rubrics:
            with st.expander("Detailed breakdown", expanded=True):
                for rubric, data in rubrics.items():
                    st.markdown(f"**{rubric.replace('_', ' ').title()}**")
                    st.caption(f"Score: {data['score']}")
                    st.write(data["reasoning"])
                    st.divider()

else:
    gauge_placeholder.info("Run analysis to see results")
