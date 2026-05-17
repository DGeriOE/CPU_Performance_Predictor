import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import altair as alt

# --- 1. Adatok és Modellek betöltése ---
@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(current_dir, "..", "data", "processed")
    data_path = os.path.join(processed_dir, "CPU_benchmark_v4_price_imputed.csv")
    df = pd.read_csv(data_path)
    inference_path = os.path.join(processed_dir, "CPU_benchmark_v4_inference.csv")
    df_inference = pd.read_csv(inference_path)
    return df, df_inference

@st.cache_resource
def load_assets():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(current_dir, "..", "models")
    
    model = joblib.load(os.path.join(base_path, 'powerPerf_model.pkl'))
    columns = joblib.load(os.path.join(base_path, 'powerPerf_model_columns.pkl'))
    categories = joblib.load(os.path.join(base_path, 'categories_list.pkl'))
    sockets = joblib.load(os.path.join(base_path, 'sockets_list.pkl'))

    importance = joblib.load(os.path.join(base_path, 'powerPerf_importance.pkl'))
    
    return model, columns, categories, sockets, importance

# Adatok betöltése
try:
    df_full, df_inference = load_data()
    pp_model, model_columns, cat_list, sock_list, importance_data = load_assets()
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
    ["Látatlan adatok (Inference)", "Eredeti adatok (tanulóhalmaz)"],
    help="A 'Pótolt' a modell által generált értékeket mutatja, az 'Eredeti' a valós benchmark eredményeket."
)

# Adatok szűrése a választás alapján
if data_mode == "Látatlan adatok (Inference)":
    current_display_df = df_inference.copy()
    status_msg = "📌 Modell által eddig nem látott adatok (TDP hiányzik)"
else:
    current_display_df = df_full.copy()
    status_msg = "📌 Tanítóhalmaz elemei"


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

    threadMark = st.sidebar.number_input("Thread Mark", min_value=0, max_value=5000, value=int(get_val('threadMark', 2500)))
    cores = st.sidebar.slider("Magok száma", 1, 128, value=int(get_val('cores', 8)))
    price = st.sidebar.number_input("Ár (USD)", min_value=0.0, max_value=10000.0, value=float(get_val('price', 300.0)))
    testDate = st.sidebar.number_input("Év", min_value=2003, max_value=2026, value=int(get_val('testDate', 2023)))
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
input_final = input_final.astype(float)

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

    # 1. Alap hisztogram létrehozása (nbins=100 és log skála)
    base_hist = alt.Chart(df_full).encode(
        x=alt.X('powerPerf:Q', bin=alt.Bin(maxbins=100), title="Power Performance"),
        y=alt.Y('count():Q', scale=alt.Scale(type='log', domain=[0.5, 600]), title="Processzorok száma (log skála)")
    )

    # A hasábok testreszabása (szín, körvonal, átlátszóság)
    bars = base_hist.mark_bar(
        color='#b4b4b4',
        opacity=0.5,
        stroke='#111111',
        strokeWidth=1,
        binSpacing=1.5
    )

    # Rétegek listája (első elem a hisztogram)
    chart_layers = [bars]

    # 2. A becsült érték vonala és szövege (Piros szaggatott)
    df_pred = pd.DataFrame([{"val": prediction_actual, "szoveg": f"Becsült: {prediction_actual:.1f}"}])

    vonal_pred = alt.Chart(df_pred).mark_rule(
        color='#ff4b4b', 
        strokeDash=[5, 5], 
        strokeWidth=2
    ).encode(x='val:Q')

    szoveg_pred = alt.Chart(df_pred).mark_text(
        align='left', 
        dx=5, 
        dy=-160, # A grafikon tetejére pozícionálja a feliratot
        color='#ff4b4b', 
        fontWeight='bold'
    ).encode(x='val:Q', text='szoveg:N')

    chart_layers.extend([vonal_pred, szoveg_pred])

    # 3. Ha van eredeti adat, a valós érték vonala és szövege (Zöld folytonos)
    if default_values is not None:
        actual_val = default_values['powerPerf']
        df_act = pd.DataFrame([{"val": actual_val, "szoveg": f"Valós: {actual_val:.1f}"}])
        
        vonal_act = alt.Chart(df_act).mark_rule(
            color='#21c354', 
            strokeWidth=2
        ).encode(x='val:Q')
        
        szoveg_act = alt.Chart(df_act).mark_text(
            align='right', 
            dx=-5, 
            dy=-160, 
            color='#21c354', 
            fontWeight='bold'
        ).encode(x='val:Q', text='szoveg:N')
        
        chart_layers.extend([vonal_act, szoveg_act])

        # 4. Rétegek összefésülése és tulajdonságok beállítása
        final_chart = alt.layer(*chart_layers).properties(
            title="Power Performance eloszlás",
            height=400
        ).interactive() # Zoomolható, körbevezethető egérrel!

        # Megjelenítés az appban
        st.altair_chart(final_chart, use_container_width=True)

        # Százalékos számítás változatlan marad
        percentile = (df_full['powerPerf'] < prediction_actual).mean() * 100
        st.write(f"Ez a processzor hatékonyabb a piacon levő modellek **{percentile:.1f}%**-ánál.")

with col_pred:
    # Eredmény megjelenítése     
    sucess_text = "Becsült Power Performance:" # if data_mode.startswith("Eredeti") else "Becsült Power Performance:"
    st.success(f"### {sucess_text}\n# {prediction_actual:.2f}")
    
    # 1. Fix magasságú doboz létrehozása (a magasságot finomhangolhatod, 130-150 pixel általában elég)
    eval_container = st.container(height=140, border=False)
    
    with eval_container:
        if default_values is not None:
            actual_pp = round(default_values['powerPerf'], 2)
            percent_diff = ((prediction_actual - actual_pp) / actual_pp) * 100 if actual_pp != 0 else 0
            
            if data_mode.startswith("Eredeti"):
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
    importances = pd.Series(importance_data.values(), index=importance_data.keys())
    top_10_features = importances.sort_values(ascending=False).head(10).round(3)
    top_10_features.index = top_10_features.index.str.replace('category_', 'Kategória: ')
    top_10_features.index = top_10_features.index.str.replace('socket_grouped_', 'Foglalat: ')
    df_chart = top_10_features.reset_index()
    df_chart.columns = ['Változó', 'Fontosság']
    
    # Gyönyörű, vízszintes, interaktív Streamlit-Altair diagram építése
    chart = alt.Chart(df_chart).mark_bar().encode(
        x=alt.X('Fontosság:Q', title='Fontossági mutató (Permutation Importance)'),
        y=alt.Y('Változó:N', sort='-x', title=''), # Autómatikusan csökkenő sorrendbe rakja a neveket
        color=alt.Color('Fontosság:Q', scale=alt.Scale(scheme='viridis'), legend=None) # Szép viridis színátmenet
    ).properties(
        height=350
    ).interactive() # Körbe lehet húzni, bele lehet nagyítani!

    # Átadjuk a natív Streamlit komponensnek
    st.altair_chart(chart, use_container_width=True)
    st.caption("A top 10 legfontosabb tényező, ami meghatározza a Stacking modell döntési logikáját (Permutation Importance).")

    # importances = pd.Series(pp_model.feature_importances_, index=model_columns)
    # top_10_features = importances.sort_values(ascending=False).head(10).round(3)
    # st.bar_chart(top_10_features)
    # st.caption("A 10 legfontosabb tényező, ami meghatározta ezt a becslést.")