import streamlit as st
import pandas as pd
import xgboost as xgb

# =======================
# LOAD MODEL XGBOOST JSON
# =======================
@st.cache_resource
def load_model():
    model_ai = xgb.XGBClassifier()
    model_ai.load_model("test_model/survey_xgb_cloud.json")
    return model_ai

model_ai = load_model()

# =======================
# PAGE CONFIG
# =======================
st.set_page_config(
    page_title="AIRA - Lung Cancer Prediction",
    layout="wide"
)

# =======================
# SIDEBAR INPUT PASIEN
# =======================
st.sidebar.title("📝 Input Pasien")

gender = st.sidebar.selectbox("Gender", ["M", "F"])
age = st.sidebar.number_input("Age", min_value=1, max_value=100, value=50)
smoking = st.sidebar.radio("Smoking", ["YES", "NO"], horizontal=True)
alcohol = st.sidebar.radio("Alcohol Consumption", ["YES", "NO"], horizontal=True)
anxiety = st.sidebar.selectbox("😰 Anxiety", ["NO", "YES"])
peer = st.sidebar.selectbox("👥 Peer Pressure", ["NO", "YES"])
chronic = st.sidebar.checkbox("🧬 Chronic Disease")
allergy = st.sidebar.checkbox("🤧 Allergy")
yellow = st.sidebar.checkbox("🟡 Yellow Fingers")
fatigue = st.sidebar.checkbox("😴 Fatigue")
wheezing = st.sidebar.checkbox("🌬️ Wheezing")
coughing = st.sidebar.checkbox("🤒 Coughing")
shortness = st.sidebar.checkbox("😮‍💨 Shortness of Breath")
swallowing = st.sidebar.checkbox("🥴 Swallowing Difficulty")
chest_pain = st.sidebar.checkbox("💔 Chest Pain")

# =======================
# TABEL INPUT USER
# =======================
st.subheader("📊 Ringkasan Input Pasien")
input_data_summary = {
    "Gender": gender,
    "Age": age,
    "Smoking": smoking,
    "Alcohol": alcohol,
    "Yellow Fingers": "YES" if yellow else "NO",
    "Anxiety": anxiety,
    "Peer Pressure": peer,
    "Chronic Disease": "YES" if chronic else "NO",
    "Fatigue": "YES" if fatigue else "NO",
    "Allergy": "YES" if allergy else "NO",
    "Wheezing": "YES" if wheezing else "NO",
    "Coughing": "YES" if coughing else "NO",
    "Shortness of Breath": "YES" if shortness else "NO",
    "Swallowing Difficulty": "YES" if swallowing else "NO",
    "Chest Pain": "YES" if chest_pain else "NO"
}
st.dataframe(pd.DataFrame([input_data_summary]), use_container_width=True)

# =======================
# ENCODING INPUT UNTUK MODEL
# =======================
def encode_gender(x): return 1 if x == "M" else 2
def encode_yesno(x): return 1 if x == "YES" else 0

input_for_model = pd.DataFrame([{
    "gender": encode_gender(gender),
    "age": age,
    "smoking": encode_yesno(smoking),
    "yellow_fingers": encode_yesno("YES" if yellow else "NO"),
    "anxiety": encode_yesno(anxiety),
    "peer_pressure": encode_yesno(peer),
    "chronic disease": encode_yesno("YES" if chronic else "NO"),
    "fatigue ": encode_yesno("YES" if fatigue else "NO"),
    "allergy ": encode_yesno("YES" if allergy else "NO"),
    "wheezing": encode_yesno("YES" if wheezing else "NO"),
    "alcohol consuming": encode_yesno(alcohol),
    "coughing": encode_yesno("YES" if coughing else "NO"),
    "shortness of breath": encode_yesno("YES" if shortness else "NO"),
    "swallowing difficulty": encode_yesno("YES" if swallowing else "NO"),
    "chest pain": encode_yesno("YES" if chest_pain else "NO")
}])

# =======================
# BUTTON PREDIKSI DENGAN PROBABILITAS
# =======================
st.divider()
if st.button("🔍 Prediksi Risiko"):
    try:
        pred_class = model_ai.predict(input_for_model)[0]          # 0 = tidak kanker, 1 = kanker
        proba = model_ai.predict_proba(input_for_model)[0]         # [P(0), P(1)]
        risk_proba = proba[pred_class] * 100

        if pred_class == 1:
            st.error(f"🔴 Risiko tinggi terkena kanker paru-paru ({risk_proba:.2f}%). Segera konsultasikan ke dokter!")
        else:
            st.success(f"🟢 Risiko rendah terkena kanker paru-paru ({risk_proba:.2f}%). Tetap jaga gaya hidup sehat!")
    except Exception as e:
        st.error(f"Terjadi kesalahan teknis saat prediksi: {e}")