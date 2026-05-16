import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# --- 1. Adatok és Modellek betöltése ---
@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "..", "data", "processed", "CPU_benchmark_v4_price_imputed.csv") 
    df = pd.read_csv(data_path)
    return df

@st.cache_resource
def load_assets():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(current_dir, "..", "models")
    
    model = joblib.load(os.path.join(base_path, 'powerPerf_model.pkl'))
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
st.set_page_config(page_title="CPU Predictor", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
    <style>
        /* Eltünteti a felesleges paddingot a fő konténer körül */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        /* Megakadályozza az iframe ugrálást */
        iframe {
            display: block;
        }
    </style>
""", unsafe_allow_html=True)
st.title("🚀 CPU Power Performance Becslés és Validáció")

# --- 3. Szűrő és Legördülő menü a Side baron ---
st.sidebar.header("CPU választása")

# ÚJ: Választókapcsoló az adathalmaz típusához
data_mode = st.sidebar.radio(
    "Milyen adatok között böngésznél?",
    ["Pótolt adatok (Imputed)", "Eredeti adatok (Original)"],
    help="A 'Pótolt' a modell által generált értékeket mutatja, az 'Eredeti' a valós benchmark eredményeket."
)

# Adatok szűrése a választás alapján
if data_mode == "Pótolt adatok (Imputed)":
    current_display_df = df_full[df_full['price_is_imputed'] == 1].copy()
    status_msg = "📌 Ez egy eredetileg hiányos adatú processzor, ahol azt regresszióval pótoltuk."
else:
    current_display_df = df_full[df_full['price_is_imputed'] == 0].copy()
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

    threadMark = st.sidebar.number_input("Thread Mark", min_value=0, value=int(get_val('threadMark', 2500)))
    cores = st.sidebar.slider("Magok száma", 1, 128, value=int(get_val('cores', 8)))
    price = st.sidebar.number_input("Ár (USD)", min_value=0.0, value=float(get_val('price', 300.0)))
    testDate = st.sidebar.number_input("Év", min_value=2000, max_value=2026, value=int(get_val('testDate', 2023)))
    category = st.sidebar.selectbox("Kategória", cat_list, index=get_idx(cat_list, get_val('category', '')))
    socket = st.sidebar.selectbox("Socket", sock_list, index=get_idx(sock_list, get_val('socket_grouped', '')))

    return pd.DataFrame({
        'threadMark': [threadMark], 'cores': [cores],
        'price': [price], 'testDate': [testDate], 'category': [category],
        'socket_grouped': [socket]
    })

input_df = user_input_features(default_values)
input_encoded = pd.get_dummies(input_df, columns=['category', 'socket_grouped'])
input_final = input_encoded.reindex(columns=model_columns, fill_value=0)

prediction = pp_model.predict(input_final)
prediction_actual = prediction[0]

# --- 5. Megjelenítés és Predikció ---
col_info, col_pred = st.columns([1, 1])

with col_info:
    st.subheader("Bemeneti paraméterek")
    st.write(input_df)
    if default_values is not None:
        st.info(status_msg)
    st.divider()
    
    st.subheader("📊 Piaci elhelyezkedés")
    
    import plotly.express as px
    
    # 1. Hisztogram generálása logaritmikus Y-tengellyel
    fig = px.histogram(
        df_full, 
        x='powerPerf', 
        nbins=100, # A felbontás sűrűsége
        color_discrete_sequence=['#b4b4b4'],
        opacity=0.6,
        log_y=True # Logaritmikus skála bekapcsolása!
    )

    fig.update_traces(
        marker_line_color='#111111', # Sötét körvonal színe
        marker_line_width=1.5,       # Körvonal vastagsága
        opacity=0.8                  # Pici átlátszóság
    )

    # 2. A becsült érték vonala (Piros szaggatott)
    fig.add_vline(
        x=prediction_actual, 
        line_dash="dash", 
        line_color="#ff4b4b", # Streamlit piros
        annotation_text=f"Becsült: {prediction_actual:.1f}",
        annotation_position="top right"
    )

    # 3. Ha van eredeti adat, a valós érték vonala (Zöld folytonos)
    if default_values is not None:
        actual_val = default_values['powerPerf']
        fig.add_vline(
            x=actual_val, 
            line_dash="solid", 
            line_color="#21c354", # Streamlit zöld
            annotation_text=f"Valós: {actual_val:.1f}",
            annotation_position="top left"
        )

    # 4. Megjelenés finomhangolása (margók, tengelynevek)
    fig.update_layout(
        title_text="Power Performance eloszlás",
        xaxis_title="Power Performance",
        yaxis_title="Processzorok száma (log skála)",
        margin=dict(l=20, r=20, t=40, b=20),
        height=400,
        showlegend=False,
        hovermode="x" # Az egér követése
    )
    
    fig.update_yaxes(exponentformat="power", showexponent="all")
    st.plotly_chart(fig, use_container_width=True)
    
    percentile = (df_full['powerPerf'] < prediction_actual).mean() * 100
    st.write(f"Ez a processzor hatékonyabb a piacon levő modellek **{percentile:.1f}%**-ánál.")

with col_pred:
    # Eredmény megjelenítése     
    sucess_text = "Becsült Power Performance:" if data_mode.startswith("Eredeti") else "Pótolt értékekkel becsült Power Performance:"
    st.success(f"### {sucess_text}\n# {prediction_actual:.2f}")
    
    # 1. Fix magasságú doboz létrehozása (a magasságot finomhangolhatod, 130-150 pixel általában elég)
    eval_container = st.container(height=140, border=False)
    
    with eval_container:
        if default_values is not None:
            actual_pp = round(default_values['powerPerf'], 2)
            percent_diff = ((prediction_actual - actual_pp) / actual_pp) * 100 if actual_pp != 0 else 0
            
            label_text = "Eredeti érték (CSV)"
            
            # A metrika megjelenítése
            st.metric(
                label=label_text, 
                value=actual_pp, 
                delta=f"{round(percent_diff, 2)}% eltérés",
                delta_color="normal" if (abs(percent_diff) <= 10) == (percent_diff >= 0) else "inverse"
            )

            # Kiértékelés
            if abs(percent_diff) < 0.1:
                st.write("✨ **Tökéletes egyezés!**")
            elif abs(percent_diff) < 10:
                st.write("✅ **Megbízható becslés:** A modell jól követi a trendet.")
            else:
                st.write("⚠️ **Jelentős eltérés:** Ennél a típusnál a modell bizonytalanabb.")
        else:
            # 2. Ha manuális bevitel van, egy láthatatlan vagy diszkrét szöveget teszünk be,
            #    hogy a doboz kitöltse a lefoglalt helyet.
            st.caption("ℹ️ Válassz egy konkrét processzort az összehasonlításhoz!")

    st.divider()

    st.subheader("🎯 Mi befolyásolta a döntést?")
    # Feature Importance kinyerése és rendezése
    importances = pd.Series(pp_model.feature_importances_, index=model_columns)
    top_10_features = importances.sort_values(ascending=False).head(10).round(3)
    st.bar_chart(top_10_features)
    st.caption("A 10 legfontosabb tényező, ami meghatározta ezt a becslést.")