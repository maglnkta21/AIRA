import joblib
import xgboost as xgb
import os

def cek_komponen():
    scaler_path = "test_model/scaler.pkl"
    model_path = "test_model/survey_xgb_cloud.json"
    
    print("--- Memulai Pengecekan Load Model ---")
    
    # 1. Cek keberadaan file
    if not os.path.exists(scaler_path):
        print(f"❌ File tidak ditemukan: {scaler_path}")
        return
    if not os.path.exists(model_path):
        print(f"❌ File tidak ditemukan: {model_path}")
        return

    try:
        # 2. Test Load Scaler
        scaler = joblib.load(scaler_path)
        print("✅ Scaler (PKL) berhasil di-load!")
        
        # 3. Test Load Model XGBoost
        model_ai = xgb.XGBClassifier()
        model_ai.load_model(model_path)
        print("✅ Model XGBoost (JSON) berhasil di-load!")
        
        print("\nKesimpulan: SEMUA BERHASIL. File siap di-upload ke GitHub.")
        
    except Exception as e:
        print(f"❌ Terjadi kesalahan saat loading: {e}")

if __name__ == "__main__":
    cek_komponen()