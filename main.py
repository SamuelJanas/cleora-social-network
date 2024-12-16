import streamlit as st
from streamlit_shortcuts import add_keyboard_shortcuts

from dashboards.raw_data import main as raw_data
from dashboards.statistics import main as statistics
from dashboards.network import main as network
from dashboards.raw_experiment import main as raw_experiment
from dashboards.cleora import main as cleora

st.set_page_config(
    page_title="Cleora knowledge extraction",
    layout="wide",
)

def main():
    # Initialize session state
    if "slide" not in st.session_state:
        st.session_state.slide = 0

    dashboards = [
        raw_data,
        statistics,
        network,
        raw_experiment,
        cleora,
    ]

    # Navigation buttons
    col1, _, col3 = st.columns([1, 10, 1])
    add_keyboard_shortcuts(
        {
            "ArrowRight": "->",
            "ArrowLeft": "<-",
        }
    )
    if col1.button("<-"):
        if st.session_state.slide > 0:
            st.session_state.slide -= 1
    if col3.button("->"):
        if st.session_state.slide < len(dashboards) - 1:
            st.session_state.slide += 1


    dashboard = dashboards[st.session_state.slide]
    dashboard()


if __name__ == "__main__":
    main()
