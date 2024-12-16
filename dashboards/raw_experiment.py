import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report

def main():
    with st.echo():
        y = st.session_state.data['book']
        X = st.session_state.data.drop('book', axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=23)

        clf = RandomForestClassifier(random_state=23)
        clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    st.write("# Results")
    f1 = f1_score(y_test, y_pred, average="weighted")
    accuracy = accuracy_score(y_test, y_pred)

    st.write(f"F1 score: {f1}, Accuracy: {accuracy}")
    st.write(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)))
