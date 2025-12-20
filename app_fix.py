import streamlit as st
import pandas as pd
import xgboost as xgb

# =======================
# PAGE CONFIG
# =======================
st.set_page_config(
    page_title="AIRA - Lung Cancer Prediction",
    layout="wide"
)

# =======================
# LOAD MODEL XGBOOST JSON
# =======================
@st.cache_resource
def load_model():
    model_ai = xgb.XGBClassifier()
    model_ai.load_model("test_model/survey_xgb_cloud.json")  # pastikan path sesuai
    return model_ai

model_ai = load_model()

# =======================
# SIDEBAR INPUT PASIEN
# =======================
st.sidebar.title("📝 Input Pasien")

# ---- Profil Pasien ----
st.sidebar.markdown("### 👤 Profil Pasien")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.number_input("Age", min_value=1, max_value=100, value=50)

# ---- Kebiasaan ----
st.sidebar.markdown("### 🚬 Kebiasaan")
smoking = st.sidebar.radio("Smoking", ["YES", "NO"], horizontal=True)
alcohol = st.sidebar.radio("Alcohol Consumption", ["YES", "NO"], horizontal=True)

# ---- Faktor Psikososial ----
st.sidebar.markdown("### 🧠 Faktor Psikososial")
anxiety = st.sidebar.selectbox("😰 Anxiety", ["NO", "YES"])
peer = st.sidebar.selectbox("👥 Peer Pressure", ["NO", "YES"])

# ---- Riwayat Medis ----
st.sidebar.markdown("### 🩺 Riwayat Medis")
chronic = st.sidebar.checkbox("🧬 Chronic Disease")
allergy = st.sidebar.checkbox("🤧 Allergy")

# ---- Gejala Klinis ----
st.sidebar.markdown("### ⚠️ Gejala Klinis")
yellow = st.sidebar.checkbox("🟡 Yellow Fingers")
fatigue = st.sidebar.checkbox("😴 Fatigue")
wheezing = st.sidebar.checkbox("🌬️ Wheezing")
coughing = st.sidebar.checkbox("🤒 Coughing")
shortness = st.sidebar.checkbox("😮‍💨 Shortness of Breath")
swallowing = st.sidebar.checkbox("🥴 Swallowing Difficulty")
chest_pain = st.sidebar.checkbox("💔 Chest Pain")

# =======================
# MAIN PAGE
# =======================
st.title("AIRA 🫁")
st.header("Artificial Intelligence for Respiratory Analysis")

st.markdown("""
<div style="max-width: 900px; text-align: justify;">
<strong>AIRA</strong> adalah solusi berbasis <em>Artificial Intelligence</em> yang dirancang untuk  
<strong>mendukung deteksi dini risiko kanker paru-paru</strong> melalui analisis pola kebiasaan  
dan indikator kesehatan individu.<br><br>

Dengan pendekatan prediktif, <strong>AIRA membantu meningkatkan kesadaran sejak dini</strong>  
sehingga langkah pencegahan dan pemeriksaan lanjutan dapat dilakukan lebih cepat.<br><br>

⚠️ <em>AIRA bukan alat diagnosis medis, melainkan sistem pendukung deteksi awal.</em>
</div>
""", unsafe_allow_html=True)

st.image("assets/gmbr paru.jpg", width=350)
st.caption("Sumber: freepik")

st.divider()

# =======================
# INFO FITUR
# =======================
with st.expander("📌 Deskripsi Faktor & Gejala"):
    st.markdown("""
    - 🚬 *Smoking*: Kebiasaan merokok yang meningkatkan risiko gangguan dan kanker paru.  
    - 🟡 *Yellow Fingers*: Perubahan warna jari akibat paparan nikotin dalam jangka panjang.  
    - 😰 *Anxiety*: Kondisi kecemasan yang dapat memengaruhi pola pernapasan.  
    - 👥 *Peer Pressure*: Tekanan sosial yang mendorong kebiasaan berisiko.  
    - 🩺 *Chronic Disease*: Riwayat penyakit kronis yang dapat memengaruhi sistem pernapasan.  
    - 😴 *Fatigue*: Kondisi mudah lelah atau penurunan energi secara terus-menerus.  
    - 🤧 *Allergy*: Reaksi alergi yang dapat mengganggu fungsi pernapasan.  
    - 🌬️ *Wheezing*: Bunyi napas seperti siulan akibat penyempitan saluran napas.  
    - 🍺 *Alcohol Consumption*: Konsumsi alkohol yang dapat menurunkan daya tahan tubuh.  
    - 🤒 *Coughing*: Batuk yang terjadi secara terus-menerus atau berkepanjangan.  
    - 😮‍💨 *Shortness of Breath*: Kesulitan atau sesak saat bernapas.  
    - 😖 *Swallowing Difficulty*: Kesulitan menelan yang berkaitan dengan gangguan saluran napas.  
    - 💔 *Chest Pain*: Nyeri pada dada, terutama saat bernapas atau batuk.  
    """)

# =======================
# ENCODING SESUAI DATASET
# =======================
def encode_yesno_dataset(x):
    return 2 if x=="YES" else 1

input_data = {
    "gender": 1 if gender=="Male" else 2,
    "age": age,
    "smoking": encode_yesno_dataset(smoking),
    "yellow_fingers": encode_yesno_dataset("YES" if yellow else "NO"),
    "anxiety": encode_yesno_dataset(anxiety),
    "peer_pressure": encode_yesno_dataset(peer),
    "chronic disease": encode_yesno_dataset("YES" if chronic else "NO"),
    "fatigue ": encode_yesno_dataset("YES" if fatigue else "NO"),
    "allergy ": encode_yesno_dataset("YES" if allergy else "NO"),
    "wheezing": encode_yesno_dataset("YES" if wheezing else "NO"),
    "alcohol consuming": encode_yesno_dataset(alcohol),
    "coughing": encode_yesno_dataset("YES" if coughing else "NO"),
    "shortness of breath": encode_yesno_dataset("YES" if shortness else "NO"),
    "swallowing difficulty": encode_yesno_dataset("YES" if swallowing else "NO"),
    "chest pain": encode_yesno_dataset("YES" if chest_pain else "NO")
}

df_input = pd.DataFrame([input_data])

# =======================
# TABEL INPUT USER
# =======================
st.subheader("📊 Ringkasan Input Pasien")
st.dataframe(df_input, use_container_width=True)

# =======================
# BUTTON PREDIKSI
# =======================
st.divider()
if st.button("🔍 Prediksi Risiko"):
    try:
        pred_class = model_ai.predict(df_input)[0]
        proba = model_ai.predict_proba(df_input)[0]
        risk_proba = proba[pred_class]*100

        if pred_class==2:  # kalau model pakai 2 = kanker
            st.error(f"🔴 Risiko tinggi terkena kanker paru-paru ({risk_proba:.2f}%). Segera konsultasikan ke dokter!")
        else:
            st.success(f"🟢 Risiko rendah terkena kanker paru-paru ({risk_proba:.2f}%). Tetap jaga gaya hidup sehat!")
    except Exception as e:
        st.error(f"Terjadi kesalahan teknis saat prediksi: {e}")