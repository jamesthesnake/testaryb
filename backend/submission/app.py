from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from backend.submission.macro_specialist import FRED_SERIES, MacroSpecialist

st.set_page_config(page_title="Macro Specialist", layout="wide")

# ---------------------------------------------------------------------------
# Shared specialist (cached so FRED data is re-used across reruns)
# ---------------------------------------------------------------------------

@st.cache_resource
def get_specialist() -> MacroSpecialist:
    return MacroSpecialist()


specialist = get_specialist()

# ---------------------------------------------------------------------------
# Sidebar — economic indicator charts
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Economic Indicators")
    chart_series = st.selectbox(
        "Series",
        list(FRED_SERIES.keys()),
        format_func=lambda k: FRED_SERIES[k][1],
    )
    n_periods = st.slider("Observations", min_value=12, max_value=120, value=40, step=4)

    if st.button("Load chart"):
        with st.spinner("Fetching FRED data…"):
            try:
                df = specialist.get_series_df(chart_series, limit=n_periods)
                series_id, description = FRED_SERIES[chart_series]
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.plot(df["date"], df["value"], linewidth=1.5)
                ax.set_title(description, fontsize=9)
                ax.set_xlabel("Date", fontsize=8)
                tick_step = max(1, len(df) // 6)
                ax.set_xticks(df["date"][::tick_step])
                ax.tick_params(axis="x", labelrotation=45, labelsize=7)
                ax.tick_params(axis="y", labelsize=7)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.caption(
                    f"Source: [FRED — {series_id}](https://fred.stlouisfed.org/series/{series_id})"
                )
            except Exception as e:
                st.error(f"Failed to load data: {e}")

# ---------------------------------------------------------------------------
# Main — chat interface
# ---------------------------------------------------------------------------

st.title("Macro Specialist")
st.caption("Ask macroeconomic questions. Data is retrieved live from FRED and analysed by GPT-4o.")

if "messages" not in st.session_state:
    st.session_state.messages: list[dict] = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander("Sources"):
                for c in msg["citations"]:
                    st.markdown(f"- **{c['description']}** — [{c['source']}]({c['url']})")

query = st.chat_input("e.g. What is the current US unemployment rate?")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Analysing…"):
            try:
                result = specialist.ask(query)
                st.markdown(result.answer)
                if result.citations:
                    with st.expander("Sources"):
                        for c in result.citations:
                            st.markdown(f"- **{c.description}** — [{c.source}]({c.url})")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result.answer,
                    "citations": [c.model_dump() for c in result.citations],
                })
            except Exception as e:
                err = f"Error: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
