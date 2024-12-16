import streamlit as st
import pandas as pd

def main():
    with st.echo():
        raw_data = pd.read_csv('data/witcher.csv', index_col=0)

        st.markdown("## Raw dataset")
        st.write(raw_data)
