import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# --- 1. Adatok és Modellek betöltése ---
@st.cache_data
def load_data():
    # Útvonal keresése
    data_path = "data/processed/CPU_benchmark_final_imputed.csv" 
    if not os.path.exists(data_path):
        data_path = "../" + data_path
    
    df = pd.read_csv(data_path)
    
    # Csak azok a sorok, ahol a powerPerf érték pótolt (imputed) volt
    only_pp_imputed_df = df[df['powerPerf_is_imputed'] == 1].copy()
    
    return only_pp_imputed_df


@st.cache_resource
def load_assets():
    base_path = "models/" if os.path.exists("models/") else "../models/"
    model = joblib.load(f'{base_path}powerPerf_rf_model.pkl')
    columns = joblib.load(f'{base_path}powerPerf_model_columns.pkl')
    # Dinamikus listák betöltése a notebookból
    categories = joblib.load(f'{base_path}categories_list.pkl')
    sockets = joblib.load(f'{base_path}sockets_list.pkl')
    return model, columns, categories, sockets

# Adatok betöltése
try:
    df_final = load_data()
    imputed_cpus = df_final[df_final['powerPerf_is_imputed'] == 1].copy()
    pp_model, model_columns, cat_list, sock_list = load_assets()
except Exception as e:
    st.error(f"Hiba a betöltéskor: {e}")
    st.stop()

# --- 2. UI Beállítása ---
st.set_page_config(page_title="CPU Predictor", layout="wide")
st.title("🚀 CPU Power Performance Becslés")

# --- 3. Legördülő menü ---
st.sidebar.header("Pótolt adatok tesztelése")

# Kiírjuk, hány ilyen processzor van
st.sidebar.write(f"Talált pótolt processzorok: {len(imputed_cpus)} db")

selected_cpu_name = st.sidebar.selectbox(
    "Válassz egy pótolt PowerPerf értékű CPU-t:",
    ["--- Manuális bevitel ---"] + list(imputed_cpus['cpuName'].unique())
)

# Alapadatok kinyerése a választott CPU-hoz
default_values = None
if selected_cpu_name != "--- Manuális bevitel ---":
    default_values = imputed_cpus[imputed_cpus['cpuName'] == selected_cpu_name].iloc[0]
    st.sidebar.info(f"Kiválasztva: {selected_cpu_name}")

def get_idx(lst, val):
    # Ha a CPU socketje nincs a listában, automatikusan az 'Other' indexét adja vissza
    if val in lst:
        return lst.index(val)
    elif 'Other' in lst:
        return lst.index('Other')
    return 0

st.sidebar.divider()
st.sidebar.header("Specifikációk finomhangolása")

# --- 4. Felhasználói bemenetek (Dinamikus alapértékekkel) ---
def user_input_features(defaults):
    # Segédfüggvény az alapérték meghatározásához
    def get_val(key, default):
        return defaults[key] if defaults is not None else default

    cpuMark = st.sidebar.number_input("CPU Mark", min_value=0, value=int(get_val('cpuMark', 15000)))
    threadMark = st.sidebar.number_input("Thread Mark", min_value=0, value=int(get_val('threadMark', 2500)))
    cores = st.sidebar.slider("Magok száma", 1, 128, value=int(get_val('cores', 8)))
    price = st.sidebar.number_input("Ár (USD)", min_value=0.0, value=float(get_val('price', 300.0)))
    testDate = st.sidebar.number_input("Év", min_value=2000, max_value=2026, value=int(get_val('testDate', 2023)))
    
    # Kategória és Socket listák
    category = st.sidebar.selectbox("Kategória", cat_list, index=get_idx(cat_list, get_val('category', '')))
    socket = st.sidebar.selectbox("Socket", sock_list, index=get_idx(sock_list, get_val('socket_grouped', '')))
    
    # Alapértelmezett index keresése a listában

    return pd.DataFrame({
        'cpuMark': [cpuMark], 'threadMark': [threadMark], 'cores': [cores],
        'price': [price], 'testDate': [testDate], 'category': [category],
        'socket_grouped': [socket]
    })

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
            
            # Százalékos eltérés kiszámítása
            if actual_pp != 0:
                percent_diff = ((prediction_actual - actual_pp) / actual_pp) * 100
            else:
                percent_diff = 0
            
            # Megjelenítés metrika kártyaként
            # A delta paraméter automatikusan mutatja az irányt és a százalékot
            st.metric(
                label="Eredeti (pótolt) érték a CSV-ben", 
                value=actual_pp, 
                delta=f"{round(percent_diff, 2)}% eltérés",
                delta_color="inverse" if abs(percent_diff) > 1 else "normal" 
            )

            # Szöveges értékelés a pontosságról
            if abs(percent_diff) < 0.01:
                st.write("✨ **Tökéletes egyezés:** A webapp és a tanítási adatok azonos eredményt adnak.")
            elif abs(percent_diff) < 5:
                st.write("✅ **Magas pontosság:** Az eltérés minimális, valószínűleg kerekítési különbség.")
            else:
                st.write(f"⚠️ **Eltérés észlelhető:** A modell {round(abs(percent_diff), 2)}%-kal más értéket számolt, mint ami a CSV-ben szerepel.")

            