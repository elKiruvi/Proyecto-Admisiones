"""Streamlit form for the fitted admissions model."""

from __future__ import annotations

import streamlit as st

from src.pipelines.inference_pipeline.inference import load_model, predict_admission


@st.cache_resource
def get_model():
    """Load the immutable fitted Pipeline once per Streamlit process."""
    return load_model()


st.set_page_config(page_title="Admissions Prediction Demo")
st.title("Admissions Prediction Demo")
st.write(
    "Enter an applicant profile to obtain a raw regression estimate from the fitted admissions model."
)

with st.form("admissions_prediction_form"):
    gre_score = st.number_input("GRE Score", value=316, step=1, format="%d")
    toefl_score = st.number_input("TOEFL Score", value=107, step=1, format="%d")
    university_rating = st.selectbox("University Rating", options=[1, 2, 3, 4, 5], index=2)
    sop = st.selectbox(
        "SOP",
        options=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
        index=5,
    )
    lor = st.selectbox(
        "LOR",
        options=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
        index=4,
    )
    cgpa = st.number_input("CGPA", value=8.62, step=0.01, format="%.2f")
    research_label = st.selectbox("Research", options=["No", "Yes"])
    submitted = st.form_submit_button("Predict")

if submitted:
    features = {
        "GRE Score": gre_score,
        "TOEFL Score": toefl_score,
        "University Rating": university_rating,
        "SOP": sop,
        "LOR": lor,
        "CGPA": cgpa,
        "Research": int(research_label == "Yes"),
    }
    try:
        prediction = predict_admission(get_model(), features)
    except (FileNotFoundError, TypeError, ValueError) as error:
        st.error(str(error))
    else:
        st.metric("Raw LinearRegression estimate", f"{prediction:.3f}")
        st.info(
            "This is a raw regression estimate, not a calibrated probability or percentage. "
            "Values outside the training range may be extrapolations."
        )
