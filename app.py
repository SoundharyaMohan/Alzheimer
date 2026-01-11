# import streamlit as st
# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from catboost import CatBoostClassifier, Pool
# from PIL import Image

# # --------------------------------------------------
# # PAGE CONFIG
# # --------------------------------------------------
# st.set_page_config(
#     page_title="🧠 Alzheimer’s Multi-Stage Diagnosis",
#     layout="wide"
# )

# st.title("🧠 Alzheimer’s Multi-Stage Diagnosis System")
# st.caption("Clinical decision support • Stage-1 Risk → Stage-2 MRI")

# # --------------------------------------------------
# # LOAD MODELS
# # --------------------------------------------------
# @st.cache_resource
# def load_stage1_model():
#     model = CatBoostClassifier()
#     model.load_model("alzheimers_catboost_model.cbm")
#     return model

# @st.cache_resource
# def load_stage2_model():
#     return tf.keras.models.load_model("alzheimers_mri_stage2_model.keras")

# stage1_model = load_stage1_model()
# stage2_model = load_stage2_model()

# # --------------------------------------------------
# # LOAD FEATURE ORDER
# # --------------------------------------------------
# @st.cache_resource
# def load_schema():
#     df = pd.read_csv("alzheimers_stage1_cleaned_dataset.csv")
#     X = df.drop(columns=["Alzheimer’s Diagnosis"])
#     cat_cols = X.select_dtypes(include=["object","category"]).columns.tolist()
#     cat_idx = [X.columns.get_loc(c) for c in cat_cols]
#     return X.columns.tolist(), cat_idx

# FEATURE_ORDER, CAT_FEATURE_INDICES = load_schema()

# # --------------------------------------------------
# # SESSION STATE
# # --------------------------------------------------
# if "demo" not in st.session_state:
#     st.session_state.demo = False

# # --------------------------------------------------
# # TOP ACTIONS
# # --------------------------------------------------
# col1, col2 = st.columns([3,1])
# with col2:
#     if st.button("⚡ Auto-Fill Demo Patient"):
#         st.session_state.demo = True

# # --------------------------------------------------
# # PROGRESS BAR
# # --------------------------------------------------
# progress = 0.35 if not st.session_state.demo else 0.65
# st.progress(progress)

# st.divider()

# # --------------------------------------------------
# # INPUT SECTIONS (MAIN PAGE)
# # --------------------------------------------------
# with st.expander("👤 Demographics", expanded=True):
#     c1, c2, c3 = st.columns(3)
#     age = c1.slider("Age", 40, 95, 72 if st.session_state.demo else 55)
#     gender = c2.radio("Gender", ["Male","Female"])
#     education = c3.selectbox(
#         "Education",
#         ["No Formal","Primary","Secondary","Graduate","Postgraduate"]
#     )

# with st.expander("🏃 Lifestyle & Mental Health"):
#     c1, c2, c3 = st.columns(3)
#     activity = c1.selectbox("Physical Activity", ["Sedentary","Light","Moderate","Active"])
#     sleep = c2.selectbox("Sleep Quality", ["Very Poor","Poor","Good","Excellent"])
#     depression = c3.selectbox("Depression Level", ["None","Mild","Moderate","Severe"])

# with st.expander("🩺 Medical History"):
#     c1, c2, c3 = st.columns(3)
#     diabetes = c1.checkbox("Diabetes", value=st.session_state.demo)
#     hypertension = c2.checkbox("Hypertension", value=st.session_state.demo)
#     family = c3.checkbox("Family History of Alzheimer’s", value=st.session_state.demo)

# with st.expander("🌍 Environment & Social"):
#     c1, c2, c3 = st.columns(3)
#     pollution = c1.selectbox("Air Pollution Exposure", ["Low","Moderate","High"])
#     living = c2.selectbox("Living Area", ["Rural","Suburban","Urban"])
#     stress = c3.selectbox("Stress Level", ["Low","Moderate","High","Very High"])

# # --------------------------------------------------
# # MAP INPUTS
# # --------------------------------------------------
# edu_map = {"No Formal":0,"Primary":5,"Secondary":10,"Graduate":15,"Postgraduate":20}
# activity_map = {"Sedentary":0,"Light":1,"Moderate":2,"Active":3}
# sleep_map = {"Very Poor":0,"Poor":1,"Good":2,"Excellent":3}
# depression_map = {"None":0,"Mild":1,"Moderate":2,"Severe":3}
# pollution_map = {"Low":0,"Moderate":1,"High":2}
# living_map = {"Rural":0,"Suburban":1,"Urban":2}
# stress_map = {"Low":0,"Moderate":1,"High":2,"Very High":3}

# input_data = {
#     "Country": 0,
#     "Age": age,
#     "Gender": 1 if gender=="Male" else 0,
#     "Education Level": edu_map[education],
#     "BMI": 27.5,
#     "Physical Activity Level": activity_map[activity],
#     "Smoking Status": 1,
#     "Alcohol Consumption": 1,
#     "Diabetes": int(diabetes),
#     "Hypertension": int(hypertension),
#     "Cholesterol Level": 2,
#     "Family History of Alzheimer’s": int(family),
#     "Depression Level": depression_map[depression],
#     "Sleep Quality": sleep_map[sleep],
#     "Dietary Habits": 1,
#     "Air Pollution Exposure": pollution_map[pollution],
#     "Employment Status": 2,
#     "Marital Status": 1,
#     "Genetic Risk Factor (APOE-ε4 allele)": 1,
#     "Social Engagement Level": 1,
#     "Income Level": 1,
#     "Stress Levels": stress_map[stress],
#     "Urban vs Rural Living": living_map[living],
# }

# st.divider()

# # --------------------------------------------------
# # PREDICTION
# # --------------------------------------------------
# if st.button("🔍 Detect Alzheimer’s Risk", type="primary"):
#     df = pd.DataFrame([input_data])[FEATURE_ORDER]
#     pool = Pool(df, cat_features=CAT_FEATURE_INDICES)
#     prob = stage1_model.predict_proba(pool)[0][1]
#     st.session_state.stage1_prob = prob

# # --------------------------------------------------
# # RESULTS
# # --------------------------------------------------
# if "stage1_prob" in st.session_state:
#     prob = st.session_state.stage1_prob

#     st.subheader("🧪 Stage-1 Risk Assessment")
#     st.metric("Alzheimer’s Risk Probability", f"{prob:.2%}")

#     if prob >= 0.6:
#         st.warning("⚠️ High risk detected — MRI scan recommended")

#         uploaded = st.file_uploader("🧠 Upload Brain MRI", type=["jpg","jpeg","png"])
#         if uploaded:
#             img = Image.open(uploaded).convert("RGB").resize((224,224))
#             arr = np.expand_dims(np.array(img)/255.0, axis=0)
#             preds = stage2_model.predict(arr)[0]

#             labels = ["Mild Dementia","Moderate Dementia","No Dementia","Very Mild Dementia"]
#             idx = np.argmax(preds)

#             st.success(f"🧠 MRI Diagnosis: **{labels[idx]}**")
#             st.metric("Confidence", f"{preds[idx]*100:.2f}%")
#     else:
#         st.success("✅ Low risk — MRI not required")

# st.caption("⚠️ For research & educational use only.")


import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from catboost import CatBoostClassifier, Pool
from PIL import Image

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="🧠 Alzheimer’s Multi-Stage Diagnosis",
    layout="wide"
)

st.title("🧠 Alzheimer’s Multi-Stage Diagnosis System")
st.caption("Clinical decision support • Stage-1 Risk → Stage-2 MRI")

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------
@st.cache_resource
def load_stage1_model():
    model = CatBoostClassifier()
    model.load_model("alzheimers_catboost_model.cbm")
    return model

@st.cache_resource
def load_stage2_model():
    return tf.keras.models.load_model("alzheimers_mri_stage2_model.keras")

stage1_model = load_stage1_model()
stage2_model = load_stage2_model()

# --------------------------------------------------
# LOAD FEATURE ORDER (DO NOT TOUCH)
# --------------------------------------------------
@st.cache_resource
def load_schema():
    df = pd.read_csv("alzheimers_stage1_cleaned_dataset.csv")
    X = df.drop(columns=["Alzheimer’s Diagnosis"])
    cat_cols = X.select_dtypes(include=["object","category"]).columns.tolist()
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]
    return X.columns.tolist(), cat_idx

FEATURE_ORDER, CAT_FEATURE_INDICES = load_schema()

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "stage1_prob" not in st.session_state:
    st.session_state.stage1_prob = None

if "demo" not in st.session_state:
    st.session_state.demo = False

# --------------------------------------------------
# STEP CARDS (TOP UI)
# --------------------------------------------------
def step_card(title, subtitle, color, icon, active):
    border = f"3px solid {color}" if active else "1px solid #ddd"
    bg = f"{color}20" if active else "#fafafa"
    st.markdown(
        f"""
        <div style="
            border:{border};
            border-radius:14px;
            padding:18px;
            height:130px;
            text-align:center;
            background:{bg};
        ">
            <div style="font-size:36px">{icon}</div>
            <div style="font-weight:700;font-size:17px">{title}</div>
            <div style="font-size:13px;color:#555">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

current_step = 1
if st.session_state.stage1_prob is not None:
    current_step = 2
    if st.session_state.stage1_prob >= 0.6:
        current_step = 3

c1, c2, c3 = st.columns(3)
with c1:
    step_card("STEP 1", "Patient Data Input", "#4A90E2", "👤", current_step == 1)
with c2:
    step_card("STEP 2", "Risk Prediction", "#F5A623", "📊", current_step == 2)
with c3:
    step_card("STEP 3", "MRI Diagnosis", "#2ECC71", "🧠", current_step == 3)

st.divider()

# --------------------------------------------------
# TOP ACTIONS
# --------------------------------------------------
col1, col2 = st.columns([4,1])
with col2:
    if st.button("⚡ Auto-Fill Demo Patient"):
        st.session_state.demo = True

st.progress(0.3 if st.session_state.stage1_prob is None else 0.6)

# --------------------------------------------------
# INPUT SECTIONS
# --------------------------------------------------
with st.expander("👤 Demographics", expanded=True):
    c1, c2, c3 = st.columns(3)
    age = c1.slider("Age", 40, 95, 72 if st.session_state.demo else 55)
    gender = c2.radio("Gender", ["Male","Female"])
    education = c3.selectbox(
        "Education",
        ["No Formal","Primary","Secondary","Graduate","Postgraduate"]
    )

with st.expander("🏃 Lifestyle & Mental Health"):
    c1, c2, c3 = st.columns(3)
    activity = c1.selectbox("Physical Activity", ["Sedentary","Light","Moderate","Active"])
    sleep = c2.selectbox("Sleep Quality", ["Very Poor","Poor","Good","Excellent"])
    depression = c3.selectbox("Depression Level", ["None","Mild","Moderate","Severe"])

with st.expander("🩺 Medical History"):
    c1, c2, c3 = st.columns(3)
    diabetes = c1.checkbox("Diabetes", value=st.session_state.demo)
    hypertension = c2.checkbox("Hypertension", value=st.session_state.demo)
    family = c3.checkbox("Family History of Alzheimer’s", value=st.session_state.demo)

with st.expander("🌍 Environment & Social"):
    c1, c2, c3 = st.columns(3)
    pollution = c1.selectbox("Air Pollution Exposure", ["Low","Moderate","High"])
    living = c2.selectbox("Living Area", ["Rural","Suburban","Urban"])
    stress = c3.selectbox("Stress Level", ["Low","Moderate","High","Very High"])

# --------------------------------------------------
# MAP INPUTS (MATCH TRAINING)
# --------------------------------------------------
maps = {
    "edu": {"No Formal":0,"Primary":5,"Secondary":10,"Graduate":15,"Postgraduate":20},
    "activity": {"Sedentary":0,"Light":1,"Moderate":2,"Active":3},
    "sleep": {"Very Poor":0,"Poor":1,"Good":2,"Excellent":3},
    "depression": {"None":0,"Mild":1,"Moderate":2,"Severe":3},
    "pollution": {"Low":0,"Moderate":1,"High":2},
    "living": {"Rural":0,"Suburban":1,"Urban":2},
    "stress": {"Low":0,"Moderate":1,"High":2,"Very High":3},
}

input_data = {
    "Country": 0,
    "Age": age,
    "Gender": 1 if gender=="Male" else 0,
    "Education Level": maps["edu"][education],
    "BMI": 27.5,
    "Physical Activity Level": maps["activity"][activity],
    "Smoking Status": 1,
    "Alcohol Consumption": 1,
    "Diabetes": int(diabetes),
    "Hypertension": int(hypertension),
    "Cholesterol Level": 2,
    "Family History of Alzheimer’s": int(family),
    "Depression Level": maps["depression"][depression],
    "Sleep Quality": maps["sleep"][sleep],
    "Dietary Habits": 1,
    "Air Pollution Exposure": maps["pollution"][pollution],
    "Employment Status": 2,
    "Marital Status": 1,
    "Genetic Risk Factor (APOE-ε4 allele)": 1,
    "Social Engagement Level": 1,
    "Income Level": 1,
    "Stress Levels": maps["stress"][stress],
    "Urban vs Rural Living": maps["living"][living],
}

st.divider()

# --------------------------------------------------
# PREDICT BUTTON
# --------------------------------------------------
if st.button("🔍 Detect Alzheimer’s Risk", type="primary"):
    df = pd.DataFrame([input_data])[FEATURE_ORDER]
    pool = Pool(df, cat_features=CAT_FEATURE_INDICES)
    st.session_state.stage1_prob = stage1_model.predict_proba(pool)[0][1]

# --------------------------------------------------
# RESULTS
# --------------------------------------------------
if st.session_state.stage1_prob is not None:
    prob = st.session_state.stage1_prob
    st.subheader("🧪 Stage-1 Risk Assessment")
    st.metric("Alzheimer’s Risk Probability", f"{prob:.2%}")

    if prob >= 0.6:
        st.warning("⚠️ High risk detected — MRI scan recommended")

        uploaded = st.file_uploader("🧠 Upload Brain MRI", type=["jpg","jpeg","png"])
        if uploaded:
            img = Image.open(uploaded).convert("RGB").resize((224,224))
            arr = np.expand_dims(np.array(img)/255.0, axis=0)
            preds = stage2_model.predict(arr)[0]
            labels = ["Mild Dementia","Moderate Dementia","No Dementia","Very Mild Dementia"]
            idx = np.argmax(preds)

            st.success(f"🧠 MRI Diagnosis: **{labels[idx]}**")
            st.metric("Confidence", f"{preds[idx]*100:.2f}%")
    else:
        st.success("✅ Low risk — MRI not required")

st.caption("⚠️ For research & educational use only.")
