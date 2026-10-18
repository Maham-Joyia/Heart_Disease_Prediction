import pickle

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------
st.set_page_config(
    page_title="HeartScope | Cardiac Risk Screener",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------
# Look & feel (teal / mint clinical theme)
# ---------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
    .block-container { padding-top: 1.5rem; max-width: 820px; }

    .top-strip {
        background: #0f766e; color: #fff; border-radius: 14px;
        padding: 1.6rem 1.8rem; margin-bottom: 1.2rem;
        border-bottom: 6px solid #5eead4;
    }
    .top-strip h1 { margin: 0; font-size: 2rem; font-weight: 700; color: #fff; }
    .top-strip p  { margin: .4rem 0 0; font-size: .95rem; color: #ccfbf1; }

    .panel-title {
        font-weight: 600; font-size: 1rem; color: #0f766e;
        text-transform: uppercase; letter-spacing: 1px; margin: .6rem 0 .4rem;
    }
    .chip-row { display: flex; gap: .6rem; flex-wrap: wrap; margin: .3rem 0 1rem; }
    .chip {
        background: #ccfbf1; color: #115e59; border-radius: 999px;
        padding: .25rem .8rem; font-size: .8rem; font-weight: 500;
    }

    .outcome-box {
        border-radius: 14px; padding: 1.3rem 1.5rem; margin-top: 1.2rem;
        display: flex; align-items: center; gap: 1rem;
    }
    .outcome-box .icon { font-size: 2.4rem; }
    .outcome-box .headline { font-size: 1.3rem; font-weight: 700; margin: 0; }
    .outcome-box .detail { font-size: .9rem; margin: .2rem 0 0; }
    .outcome-high { background: #fff4e5; border-left: 8px solid #ea580c; color: #7c2d12; }
    .outcome-low  { background: #e8fbf3; border-left: 8px solid #059669; color: #064e3b; }

    div.stButton > button {
        background: #0f766e; color: #fff; border: 0; border-radius: 999px;
        padding: .65rem 1.6rem; font-weight: 600; width: 100%;
    }
    div.stButton > button:hover { background: #115e59; color: #fff; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# Trained model
# ---------------------------------------------------------------
@st.cache_resource
def load_heart_model():
    with open("heart_svc_trained_model.pkl", "rb") as fh:
        return pickle.load(fh)


heart_model = load_heart_model()

# ---------------------------------------------------------------
# Allowed values + label encoders (same classes as in the notebook)
# ---------------------------------------------------------------
PAIN_TYPES = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
YES_NO = ["Yes", "No"]
THAL_RESULTS = ["Normal", "Fixed Defect", "Reversible Defect"]
VESSEL_COUNTS = ["0 Vessels", "1 Vessel", "2 Vessels", "3 Vessels"]

enc_pain = LabelEncoder().fit(np.ravel(PAIN_TYPES))
enc_angina = LabelEncoder().fit(np.ravel(YES_NO))
enc_thal = LabelEncoder().fit(np.ravel(THAL_RESULTS))
enc_vessels = LabelEncoder().fit(np.ravel(VESSEL_COUNTS))

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.markdown(
    """
    <div class="top-strip">
        <h1>🩺 HeartScope</h1>
        <p>A quick cardiac risk screener built on four clinical indicators
        from the UCI Heart Disease (Cleveland) data.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_check, tab_about = st.tabs(["🔎 Screening", "ℹ️ About the project"])

# ---------------------------------------------------------------
# Tab 1: screening form + result
# ---------------------------------------------------------------
with tab_check:
    st.markdown('<div class="panel-title">Patient indicators</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        in_pain = st.selectbox(
            "Chest pain type",
            PAIN_TYPES,
            help="Kind of chest pain the patient reports.",
        )
        in_angina = st.radio(
            "Angina triggered by exercise?",
            YES_NO,
            index=1,
            horizontal=True,
            help="Does physical effort bring on chest pain?",
        )
    with c2:
        in_thal = st.radio(
            "Thallium stress-test result",
            THAL_RESULTS,
            help="Outcome of the thallium stress test.",
        )
        in_vessels = st.select_slider(
            "Major vessels highlighted by fluoroscopy",
            options=VESSEL_COUNTS,
            help="Number of major vessels showing blockage.",
        )

    st.write("")
    go = st.button("Run screening")

    if go:
        raw_input = pd.DataFrame({
            "ChestPain": [in_pain],
            "ExerciseAngina": [in_angina],
            "Thal": [in_thal],
            "Vessels": [in_vessels],
        })

        coded_input = raw_input.copy()
        coded_input["ChestPain"] = enc_pain.transform(raw_input["ChestPain"])
        coded_input["ExerciseAngina"] = enc_angina.transform(raw_input["ExerciseAngina"])
        coded_input["Thal"] = enc_thal.transform(raw_input["Thal"])
        coded_input["Vessels"] = enc_vessels.transform(raw_input["Vessels"])

        verdict = heart_model.predict(coded_input)[0]

        if verdict == 1:
            st.markdown(
                """
                <div class="outcome-box outcome-high">
                    <div class="icon">🚨</div>
                    <div>
                        <p class="headline">Heart disease likely</p>
                        <p class="detail">This profile matches the high-risk pattern learned by the
                        model. A cardiologist visit is strongly recommended.</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="outcome-box outcome-low">
                    <div class="icon">💚</div>
                    <div>
                        <p class="headline">No heart disease indicated</p>
                        <p class="detail">The model does not see a high-risk pattern in this
                        profile. Regular check-ups are still a good idea.</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("Show what was sent to the model"):
            st.write("Entered values")
            st.dataframe(raw_input, hide_index=True)
            st.write("Label-encoded values")
            st.dataframe(coded_input, hide_index=True)

# ---------------------------------------------------------------
# Tab 2: project details
# ---------------------------------------------------------------
with tab_about:
    st.markdown('<div class="panel-title">What is this?</div>', unsafe_allow_html=True)
    st.write(
        "HeartScope is a small end-to-end machine learning project. It reads four "
        "categorical clinical indicators and predicts whether heart disease is present."
    )
    st.markdown(
        """
        <div class="chip-row">
            <span class="chip">Model: Support Vector Classifier</span>
            <span class="chip">Encoding: LabelEncoder</span>
            <span class="chip">Inputs: 4 attributes</span>
            <span class="chip">Data: UCI Cleveland</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="panel-title">Attributes used</div>', unsafe_allow_html=True)
    st.table(pd.DataFrame({
        "Attribute": ["Chest pain type", "Exercise angina", "Thallium test", "Major vessels"],
        "Possible values": [
            ", ".join(PAIN_TYPES),
            ", ".join(YES_NO),
            ", ".join(THAL_RESULTS),
            ", ".join(VESSEL_COUNTS),
        ],
    }))
    st.caption("Course AIC354 - Machine Learning Fundamentals, COMSATS University Islamabad, Lahore Campus.")

st.caption("Educational project only. This is not a medical diagnosis tool.")
