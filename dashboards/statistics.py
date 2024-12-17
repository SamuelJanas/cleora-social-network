import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def main():
    raw_data = pd.read_csv('data/witcher.csv', index_col=0)

    with st.echo():
        raw_data.drop('Type', axis=1, inplace=True)
        leSource = LabelEncoder()
        leSource.fit(raw_data['Source'])
        raw_data['Source'] = leSource.transform(raw_data['Source'])

        leTarget = LabelEncoder()
        leTarget.fit(raw_data['Target'])
        raw_data['Target'] = leTarget.transform(raw_data['Target'])

    if "data" not in st.session_state:
        st.session_state.data= raw_data
    with st.expander("Entire DataFrame"):
        st.write(raw_data)
    st.markdown("## Dataset statistics")
    st.write(raw_data.describe())

    fig = plt.figure(figsize=(10, 5))
    sns.histplot(raw_data['Weight'], bins=30, kde=False)
    plt.title(f'Number of interactions between characters (log scale)')
    plt.yscale('log')
    plt.ylim(0.1, 1000)
    st.pyplot(fig)

    dist_fig = plt.figure(figsize=(10, 5))
    sns.countplot(y="book", data=raw_data)
    plt.title("Target (book) distribution")
    st.pyplot(dist_fig)

    # calculate the probability of guessing the correct book
    book_counts = raw_data['book'].value_counts()
    guess_proba = book_counts / book_counts.sum()

    st.markdown("""
                ### Conclusion
                - Most characters have a low number of interactions, with a few exceptions.
                - Guessing the correct book randomly would result in the following accuracies:
                """)
    st.write(guess_proba)
