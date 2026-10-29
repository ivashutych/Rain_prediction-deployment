import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Rain Prediction")

uploaded_model = st.file_uploader(
    "Upload your Random Forest model",
    type=["joblib"]
)

if uploaded_model is None:
    st.info("Please upload a model to continue.")
    st.stop()

model = joblib.load(uploaded_model)

imputer = joblib.load("artifacts/imputer.joblib")
imputer_cat = joblib.load("artifacts/imputer_cat.joblib")
scaler = joblib.load("artifacts/scaler.joblib")
encoder = joblib.load("artifacts/encoder.joblib")
feature_cols = joblib.load("artifacts/feature_cols.joblib")

numeric_cols = list(imputer.feature_names_in_)
cat_cols = list(imputer_cat.feature_names_in_)

st.subheader("Enter weater data")

input_data = {}

for col in numeric_cols: 
    input_data[col] = st.number_input(col, value = 0.0)
    
for col, categories in zip(cat_cols, encoder.categories_):
    input_data[col] = st.selectbox(
        col,
        options = list(categories)
    )

input_df = pd.DataFrame([input_data])

st.write(input_df)

processed_df = input_df.copy()

processed_df[numeric_cols] = imputer.transform(processed_df[numeric_cols])
processed_df[cat_cols] = imputer_cat.transform(processed_df[cat_cols])

processed_df[numeric_cols] = scaler.transform(processed_df[numeric_cols])

encoded_cols = encoder.get_feature_names_out(cat_cols)

encoded_df = pd.DataFrame(
    encoder.transform(processed_df[cat_cols]),
    columns = encoded_cols,
    index = processed_df.index)

X = pd.concat([processed_df[numeric_cols], encoded_df], axis = 1)

X = X[feature_cols]

st.write("Processed features:", X.shape)

if st.button("Predict Rain"):
    positive_class_index = list(model.classes_).index("Yes")    
    probability = model.predict_proba(X)[0, positive_class_index]
    st.metric("Probability rain tomorrow", f"{probability:.0%}")

    if probability>= 0.5:
        st.warning("Yes")
    else:
        st.success("No")