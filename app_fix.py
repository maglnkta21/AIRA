import streamlit as st
import pandas as pd
import joblib  # untuk load model

# =======================
# LOAD MODEL
# =======================
# Path relatif dari app_fix.py ke folder test_model
model = joblib.load("test_model/survey_xgb_1.pkl")

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

# Profil Pasien
st.sidebar.markdown("### 👤 Profil Pasien")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.number_input("Age", min_value=1, max_value=100, value=50)

# Kebiasaan
st.sidebar.markdown("### 🚬 Kebiasaan")
smoking = st.sidebar.radio("Smoking", ["YES", "NO"], horizontal=True)
alcohol = st.sidebar.radio("Alcohol Consumption", ["YES", "NO"], horizontal=True)

# Faktor Psikososial
st.sidebar.markdown("### 🧠 Faktor Psikososial")
anxiety = st.sidebar.selectbox("😰 Anxiety", ["NO", "YES"])
peer = st.sidebar.selectbox("👥 Peer Pressure", ["NO", "YES"])

# Riwayat Medis
st.sidebar.markdown("### 🩺 Riwayat Medis")
chronic = st.sidebar.checkbox("🧬 Chronic Disease")
allergy = st.sidebar.checkbox("🤧 Allergy")

# Gejala Klinis
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

st.markdown(
    """
    <div style="max-width: 900px; text-align: justify;">
    <strong>AIRA</strong> adalah solusi berbasis <em>Artificial Intelligence</em> yang dirancang untuk  
    <strong>mendukung deteksi dini risiko kanker paru-paru</strong> melalui analisis pola kebiasaan  
    dan indikator kesehatan individu.<br><br>

    Dengan pendekatan prediktif, <strong>AIRA membantu meningkatkan kesadaran sejak dini</strong>  
    sehingga langkah pencegahan dan pemeriksaan lanjutan dapat dilakukan lebih cepat.<br><br>

    ⚠️ <em>AIRA bukan alat diagnosis medis, melainkan sistem pendukung deteksi awal.</em>
    </div>
    """,
    unsafe_allow_html=True
)

st.image("assets/gmbr paru.jpg", width=350)
st.caption("Sumber: freepik")

st.divider()

# INFO FITUR
with st.expander("📌 Deskripsi Faktor & Gejala"):
    st.markdown("""
    - 🚬 **Smoking**: Kebiasaan merokok yang meningkatkan risiko gangguan dan kanker paru.  
    - 🟡 **Yellow Fingers**: Perubahan warna jari akibat paparan nikotin dalam jangka panjang.  
    - 😰 **Anxiety**: Kondisi kecemasan yang dapat memengaruhi pola pernapasan.  
    - 👥 **Peer Pressure**: Tekanan sosial yang mendorong kebiasaan berisiko.  
    - 🩺 **Chronic Disease**: Riwayat penyakit kronis yang dapat memengaruhi sistem pernapasan.  
    - 😴 **Fatigue**: Kondisi mudah lelah atau penurunan energi secara terus-menerus.  
    - 🤧 **Allergy**: Reaksi alergi yang dapat mengganggu fungsi pernapasan.  
    - 🌬️ **Wheezing**: Bunyi napas seperti siulan akibat penyempitan saluran napas.  
    - 🍺 **Alcohol Consumption**: Konsumsi alkohol yang dapat menurunkan daya tahan tubuh.  
    - 🤒 **Coughing**: Batuk yang terjadi secara terus-menerus atau berkepanjangan.  
    - 😮‍💨 **Shortness of Breath**: Kesulitan atau sesak saat bernapas.  
    - 😖 **Swallowing Difficulty**: Kesulitan menelan yang berkaitan dengan gangguan saluran napas.  
    - 💔 **Chest Pain**: Nyeri pada dada, terutama saat bernapas atau batuk.  
    """)

# =======================
# TABEL INPUT USER
# =======================
st.subheader("📊 Ringkasan Input Pasien")

input_data = {
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

df = pd.DataFrame([input_data])
st.dataframe(df, use_container_width=True)

# =======================
# ENCODING INPUT UNTUK MODEL
# =======================
def encode_yesno(value):
    return 2 if value == "YES" else 1

def encode_gender(value):
    return 1 if value == "Male" else 2

input_for_model = pd.DataFrame([{
    "gender": encode_gender(gender),
    "age": age,
    "smoking": encode_yesno(smoking),
    "yellow_fingers": encode_yesno("YES" if yellow else "NO"),
    "anxiety": encode_yesno(anxiety),
    "peer_pressure": encode_yesno(peer),
    "chronic disease": encode_yesno("YES" if chronic else "NO"),  # ada spasi
    "fatigue ": encode_yesno("YES" if fatigue else "NO"),          # spasi di akhir
    "allergy ": encode_yesno("YES" if allergy else "NO"),          # spasi di akhir
    "wheezing": encode_yesno("YES" if wheezing else "NO"),
    "alcohol consuming": encode_yesno(alcohol),                    # spasi di tengah
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
    pred_class = model.predict(input_for_model)[0]        # 0 = tidak kanker, 1 = kanker
    proba = model.predict_proba(input_for_model)[0]       # [P(0), P(1)]
    risk_proba = proba[pred_class] * 100                  # konversi ke persen

    if pred_class == 1:
        st.error(f"🔴 Perhatian! Risiko terkena kanker paru-paru tergolong tinggi, dengan tingkat keyakinan sekitar {risk_proba:.2f}%. Segera konsultasikan ke dokter!")
    else:
        st.success(f"🟢 Risiko terkena kanker paru-paru tergolong rendah, dengan tingkat keyakinan sekitar {risk_proba:.2f}%. Tetap jaga gaya hidup sehat!")
