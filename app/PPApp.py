import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# --- 1. Modell betöltése (Útvonal korrekcióval) ---
@st.cache_resource
def load_models():
    # Mivel az app/ mappában vagy, a modellek pedig a models/ mappában a projekt gyökerében
    # Megpróbáljuk mindkét irányból (projekt gyökér vs app mappa)
    base_path = "models/" if os.path.exists("models/") else "../models/"
    
    model = joblib.load(f'{base_path}powerPerf_rf_model.pkl')
    columns = joblib.load(f'{base_path}powerPerf_model_columns.pkl')
    return model, columns

try:
    pp_model, model_columns = load_models()
except Exception as e:
    st.error(f"Hiba a modellek betöltésekor: {e}")
    st.stop()

# --- 2. UI Beállítása ---
st.set_page_config(page_title="CPU Predictor", layout="wide")
st.title("🚀 CPU Power Performance Becslés")

st.sidebar.header("CPU Specifikációk megadása")

# --- 3. Felhasználói bemenetek ---
def user_input_features():
    # Numerikus értékek
    cpuMark = st.sidebar.number_input("CPU Mark (Összteljesítmény)", min_value=0, value=15000)
    threadMark = st.sidebar.number_input("Thread Mark (Egyszálas teljesítmény)", min_value=0, value=2500)
    cores = st.sidebar.slider("Magok száma", 1, 128, 8)
    price = st.sidebar.number_input("Ár (USD)", min_value=0.0, value=300.0)
    testDate = st.sidebar.number_input("Megjelenés éve", min_value=2000, max_value=2026, value=2023)
    
    # Kategóriák (Ezeket a tanítási adathalmazod alapján állítottam be)
    category = st.sidebar.selectbox("Kategória", ['Desktop', 'Laptop', 'Server', 'Mobile', 'Embedded'])
    
    # Socket - Itt a top 20-at használtad, a leggyakoribbakat sorolom fel példaként
    socket_list = ['LGA1700', 'AM4', 'AM5', 'LGA1200', 'FCBGA1449', 'LGA1151', 'Other']
    socket = st.sidebar.selectbox("Foglalat (Socket)", socket_list)
    
    data = {
        'cpuMark': cpuMark,
        'threadMark': threadMark,
        'cores': cores,
        'price': price,
        'testDate': testDate,
        'category': category,
        'socket_grouped': socket
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Adatok megjelenítése a főképernyőn
st.subheader("Megadott adatok")
st.write(input_df)

# --- 4. Predikció logika ---
if st.button('Becslés indítása'):
    with st.spinner('Számítás folyamatban...'):
        # 1. One-Hot Encoding elvégzése a bemeneten
        input_encoded = pd.get_dummies(input_df, columns=['category', 'socket_grouped'])
        
        # 2. Oszlopok összehangolása a modell elvárásaival (Reindex)
        # Ez hozzáadja a hiányzó 0-ás oszlopokat és sorba rendezi őket
        input_final = input_encoded.reindex(columns=model_columns, fill_value=0)
        
        # 3. Predikció (Log skálán)
        prediction_log = pp_model.predict(input_final)
        
        # 4. Visszaalakítás valós értékre (np.expm1) és kerekítés
        prediction_actual = np.expm1(prediction_log)[0]
        prediction_final = round(prediction_actual, 2)
        
        # --- 5. Eredmény megjelenítése ---
        st.success(f"### A becsült Power Performance érték: **{prediction_final}**")
        
        # Egy kis vizuális segítség: hol helyezkedik el ez?
        st.info("A Power Performance a teljesítmény és a fogyasztás arányát jelzi (magasabb = hatékonyabb).")
        
        # Metrika kártya
        col1, col2 = st.columns(2)
        col1.metric("Becsült Hatékonyság", f"{prediction_final}")
        col2.metric("Bemeneti CPU Mark", f"{input_df['cpuMark'][0]}")