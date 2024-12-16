import streamlit as st
from sklearn.metrics import classification_report, f1_score, accuracy_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def main():
    # Replace with the actual file paths if they are in a specific folder.
    source_target_embeddings = pd.read_csv("output/emb__Source__Target.out", delim_whitespace=True, header=None, skiprows=1)
    source_weight_embeddings = pd.read_csv("output/emb__Source__Weight.out", delim_whitespace=True, header=None, skiprows=1)
    weight_target_embeddings = pd.read_csv("output/emb__Target__Weight.out", delim_whitespace=True, header=None, skiprows=1)
    interaction_data = pd.read_csv("data/witcher_processed.tsv", sep="\t")
    book_labels = pd.read_csv("data/witcher_target.tsv", sep="\t")

    # Step 1: Merge Data
    # Extract column names for embeddings (assuming embeddings start from column 2)
    source_target_embeddings.columns = ["Node"] + [f"Embed_{i}" for i in range(1, source_target_embeddings.shape[1])]
    source_weight_embeddings.columns = ["Node"] + [f"Embed_{i}" for i in range(1, source_weight_embeddings.shape[1])]
    weight_target_embeddings.columns = ["Node"] + [f"Embed_{i}" for i in range(1, weight_target_embeddings.shape[1])]

    # Merge embeddings with interaction data
    interaction_data = interaction_data.merge(source_target_embeddings, left_on="Source", right_on="Node", suffixes=("", "_Source"))
    interaction_data = interaction_data.merge(source_target_embeddings, left_on="Target", right_on="Node", suffixes=("", "_Target"))
    interaction_data = interaction_data.merge(book_labels, left_index=True, right_index=True)

    interaction_data.drop(columns=["Node", "Embed_1"])

    # Combine source and target embeddings (e.g., concatenation)
    source_emb_cols = [col for col in interaction_data.columns if "Embed_" in col and "_Source" in col]
    target_emb_cols = [col for col in interaction_data.columns if "Embed_" in col and "_Target" in col]

    interaction_data["Weight_Normalized"] = interaction_data["Weight"] / interaction_data["Weight"].max()

    # Example: Concatenate source and target embeddings
    interaction_data["Features"] = interaction_data[source_emb_cols + target_emb_cols].values.tolist()

    # Step 3: Train/Test Split
    X = np.stack(interaction_data["Features"])
    y = interaction_data["book"]  # Target label
    st.write("### Final dataset:")
    st.write(X)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=23)
    clf = RandomForestClassifier(random_state=23)
    clf.fit(X_train, y_train)

    # Step 6: Evaluate
    y_pred = clf.predict(X_test)
    y_pred = clf.predict(X_test)

    st.write("# Results")
    f1 = f1_score(y_test, y_pred, average="weighted")
    accuracy = accuracy_score(y_test, y_pred)

    st.write(f"F1 score: {f1}, Accuracy: {accuracy}")
    st.write(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)))
