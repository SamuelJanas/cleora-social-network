import streamlit as st
import pandas as pd

def main():
    _, cen, _ = st.columns([1, 3, 1])
    with cen: # Center the content
        with st.echo():
            raw_data = pd.read_csv('data/witcher.csv', index_col=0)

            st.markdown("## Raw dataset")
            st.write(raw_data)
