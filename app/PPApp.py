import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# --- 1. Adatok és Modellek betöltése ---
@st.cache_data
def load_data():
    # A legvégső fájlt töltsd be, amiben már minden (TDP, Price ÉS PowerPerf) pótolva van!
    data_path = "data/processed/CPU_benchmark_final_imputed.csv" 
    if not os.path.exists(data_path):
        data_path = "../" + data_path
    
    df = pd.read_csv(data_path)
    
    # Itt most már mind a három pótlási jelzőt nézzük
    imputed_mask = (df['TDP_is_imputed'] == 1) | \
                   (df['price_is_imputed'] == 1) | \
                   (df['powerPerf_is_imputed'] == 1)
                   
    return df[imputed_mask].copy()

@st.cache_resource
def load_models():
    base_path = "models/" if os.path.exists("models/") else "../models/"
    model = joblib.load(f'{base_path}powerPerf_rf_model.pkl')
    columns = joblib.load(f'{base_path}powerPerf_model_columns.pkl')
    return model, columns

# Adatok betöltése
try:
    imputed_cpus = load_data()
    pp_model, model_columns = load_models()
except Exception as e:
    st.error(f"Hiba az adatok/modellek betöltésekor: {e}")
    st.stop()

# --- 2. UI Beállítása ---
st.set_page_config(page_title="CPU Predictor", layout="wide")
st.title("🚀 CPU Power Performance Becslés")

# --- 3. Legördülő menü a pótolt CPU-khoz ---
st.sidebar.header("Pótolt adatok tesztelése")
selected_cpu_name = st.sidebar.selectbox(
    "Válassz egy korábban hiányos adatú processzort:",
    ["--- Manuális bevitel ---"] + list(imputed_cpus['cpuName'].unique())
)

# Alapadatok kinyerése a választott CPU-hoz
default_values = None
if selected_cpu_name != "--- Manuális bevitel ---":
    default_values = imputed_cpus[imputed_cpus['cpuName'] == selected_cpu_name].iloc[0]
    st.sidebar.info(f"Kiválasztva: {selected_cpu_name}")

st.sidebar.divider()
st.sidebar.header("Specifikációk finomhangolása")

# --- 4. Felhasználói bemenetek (Dinamikus alapértékekkel) ---
def user_input_features(defaults):
    # Segédfüggvény az alapérték meghatározásához
    def get_val(key, default):
        return defaults[key] if defaults is not None else default

    cpuMark = st.sidebar.number_input("CPU Mark", min_value=0, 
                                     value=int(get_val('cpuMark', 15000)))
    threadMark = st.sidebar.number_input("Thread Mark", min_value=0, 
                                         value=int(get_val('threadMark', 2500)))
    cores = st.sidebar.slider("Magok száma", 1, 128, 
                               value=int(get_val('cores', 8)))
    price = st.sidebar.number_input("Ár (USD)", min_value=0.0, 
                                     value=float(get_val('price', 300.0)))
    testDate = st.sidebar.number_input("Megjelenés éve", min_value=2000, max_value=2026, 
                                        value=int(get_val('testDate', 2023)))
    
    # Kategória és Socket listák
    cat_list = ['Desktop', 'Laptop', 'Server', 'Mobile', 'Embedded']
    sock_list = ['LGA1700', 'AM4', 'AM5', 'LGA1200', 'FCBGA1449', 'LGA1151', 'Other']
    
    # Alapértelmezett index keresése a listában
    def get_idx(lst, val):
        return lst.index(val) if val in lst else 0

    category = st.sidebar.selectbox("Kategória", cat_list, 
                                    index=get_idx(cat_list, get_val('category', 'Desktop')))
    socket = st.sidebar.selectbox("Foglalat (Socket)", sock_list, 
                                   index=get_idx(sock_list, get_val('socket_grouped', 'Other')))
    
    data = {
        'cpuMark': cpuMark, 'threadMark': threadMark, 'cores': cores,
        'price': price, 'testDate': testDate, 'category': category,
        'socket_grouped': socket
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features(default_values)

# --- 5. Megjelenítés és Predikció ---
col_info, col_pred = st.columns([1, 1])

with col_info:
    st.subheader("Bemeneti paraméterek")
    st.write(input_df)
    if default_values is not None:
        st.write("📌 *Ez egy eredetileg hiányos adatú processzor, amit regresszióval pótoltál.*")

with col_pred:
    if st.button('Becslés indítása', use_container_width=True):
        input_encoded = pd.get_dummies(input_df, columns=['category', 'socket_grouped'])
        input_final = input_encoded.reindex(columns=model_columns, fill_value=0)
        
        prediction_log = pp_model.predict(input_final)
        prediction_actual = np.expm1(prediction_log)[0]
        
        st.success(f"### Becsült Power Performance:\n# {round(prediction_actual, 2)}")
        
        if default_values is not None:
            actual_pp = round(default_values['powerPerf'], 2)
            st.metric("Eredeti (pótolt) érték a CSV-ben", actual_pp)
            diff = round(prediction_actual - actual_pp, 2)
            st.write(f"Különbség: {diff}")