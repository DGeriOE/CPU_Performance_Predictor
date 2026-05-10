import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# --- 1. Adatok és Modellek betöltése ---
@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "..", "data", "processed", "CPU_benchmark_final_imputed.csv") 
    
    # Itt most a teljes adathalmazt betöltjük, nem szűrünk előre
    df = pd.read_csv(data_path)
    return df

@st.cache_resource
def load_assets():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(current_dir, "..", "models")
    
    model = joblib.load(os.path.join(base_path, 'powerPerf_rf_model.pkl'))
    columns = joblib.load(os.path.join(base_path, 'powerPerf_model_columns.pkl'))
    categories = joblib.load(os.path.join(base_path, 'categories_list.pkl'))
    sockets = joblib.load(os.path.join(base_path, 'sockets_list.pkl'))
    
    return model, columns, categories, sockets

# Adatok betöltése
try:
    df_full = load_data()
    pp_model, model_columns, cat_list, sock_list = load_assets()
except Exception as e:
    st.error(f"Hiba a betöltéskor: {e}")
    st.stop()

# --- 2. UI Beállítása ---
st.set_page_config(page_title="CPU Predictor", layout="wide")
st.title("🚀 CPU Power Performance Becslés és Validáció")

# --- 3. Szűrő és Legördülő menü a Side baron ---
st.sidebar.header("Adathalmaz választása")

# ÚJ: Választókapcsoló az adathalmaz típusához
data_mode = st.sidebar.radio(
    "Milyen adatok között böngésznél?",
    ["Pótolt adatok (Imputed)", "Eredeti adatok (Original)"],
    help="A 'Pótolt' a modell által generált értékeket mutatja, az 'Eredeti' a valós benchmark eredményeket."
)

# Adatok szűrése a választás alapján
if data_mode == "Pótolt adatok (Imputed)":
    current_display_df = df_full[df_full['powerPerf_is_imputed'] == 1].copy()
    status_msg = "📌 Ez egy eredetileg hiányos adatú processzor, amit regresszióval pótoltál."
else:
    current_display_df = df_full[df_full['powerPerf_is_imputed'] == 0].copy()
    status_msg = "🔍 Ez egy valós mérési adat. Teszteld, mennyire pontos a modell becslése!"

st.sidebar.write(f"Talált processzorok: {len(current_display_df)} db")

selected_cpu_name = st.sidebar.selectbox(
    "Válassz egy CPU-t a listából:",
    ["--- Manuális bevitel ---"] + list(current_display_df['cpuName'].unique())
)

# Alapadatok kinyerése
default_values = None
if selected_cpu_name != "--- Manuális bevitel ---":
    default_values = current_display_df[current_display_df['cpuName'] == selected_cpu_name].iloc[0]

def get_idx(lst, val):
    if val in lst:
        return lst.index(val)
    elif 'Other' in lst:
        return lst.index('Other')
    return 0

st.sidebar.divider()
st.sidebar.header("Specifikációk finomhangolása")

# --- 4. Felhasználói bemenetek ---
def user_input_features(defaults):
    def get_val(key, default):
        return defaults[key] if defaults is not None else default

    cpuMark = st.sidebar.number_input("CPU Mark", min_value=0, value=int(get_val('cpuMark', 15000)))
    threadMark = st.sidebar.number_input("Thread Mark", min_value=0, value=int(get_val('threadMark', 2500)))
    cores = st.sidebar.slider("Magok száma", 1, 128, value=int(get_val('cores', 8)))
    price = st.sidebar.number_input("Ár (USD)", min_value=0.0, value=float(get_val('price', 300.0)))
    testDate = st.sidebar.number_input("Év", min_value=2000, max_value=2026, value=int(get_val('testDate', 2023)))
    category = st.sidebar.selectbox("Kategória", cat_list, index=get_idx(cat_list, get_val('category', '')))
    socket = st.sidebar.selectbox("Socket", sock_list, index=get_idx(sock_list, get_val('socket_grouped', '')))

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
        st.info(status_msg)

with col_pred:
    if st.button('Becslés indítása', use_container_width=True):
        # Kódolás és predikció
        input_encoded = pd.get_dummies(input_df, columns=['category', 'socket_grouped'])
        input_final = input_encoded.reindex(columns=model_columns, fill_value=0)
        
        prediction_log = pp_model.predict(input_final)
        prediction_actual = np.expm1(prediction_log)[0]
        
        st.success(f"### Becsült Power Performance:\n# {round(prediction_actual, 2)}")
        
        if default_values is not None:
            actual_pp = round(default_values['powerPerf'], 2)
            percent_diff = ((prediction_actual - actual_pp) / actual_pp) * 100 if actual_pp != 0 else 0
            
            label_text = "Eredeti érték (CSV)" if data_mode.startswith("Eredeti") else "Pótolt érték (CSV)"
            
            st.metric(
                label=label_text, 
                value=actual_pp, 
                delta=f"{round(percent_diff, 2)}% eltérés",
                delta_color="inverse" if abs(percent_diff) > 5 else "normal" 
            )

            # Kiértékelés
            if abs(percent_diff) < 0.1:
                st.write("✨ **Tökéletes egyezés!**")
            elif abs(percent_diff) < 10:
                st.write("✅ **Megbízható becslés:** A modell jól követi a trendet.")
            else:
                st.write("⚠️ **Jelentős eltérés:** Ennél a típusnál a modell bizonytalanabb.")