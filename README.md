---
title: CPU Performance Predictor
emoji: 🚀
colorFrom: blue
colorTo: red
sdk: streamlit
app_file: app/PPApp.py
pinned: false
---

# CPU Power Performance Predikció 🚀

Ez a repozitórium egy egyetemi gépi tanulás feladat megoldását tartalmazza. A projekt célja asztali és mobil processzorok (CPU) `powerPerf` (teljesítmény/fogyasztás) arányának megjóslása többváltozós **lineáris regresszió** segítségével.

## 📊 Az Adathalmaz
A feladathoz használt adatbázis a Kaggle platformról származik, amely a PassMark szoftver benchmark eredményeit tartalmazza:
[CPU Benchmarks Dataset (Kaggle)](https://www.kaggle.com/datasets/alanjo/cpu-benchmarks)

Statisztika és stratégia:
- 3825 line:
   - **1858**-nál hiányzik a **price** (ezzel együtt cpuValue és threadValue is) --> price pótlásaival **tanítóhalmaz**
   - **685**-nél hiányzik a **TDP** (ezzel együtt a powerPerf is) --> TDP pótlásaival végső **tesztelő halmaz**
 

*Megjegyzés: Az adathalmaz egy része logaritmikus transzformáción (log-transform) esik át a modell betanítása előtt a jobb eloszlás és a lineáris kapcsolatok erősítése érdekében.*

## 📁 Projekt Struktúra
- `data/` : Ide kerülnek a nyers (`raw`) és az előfeldolgozott (`processed`) .csv fájlok. (Ezek a `.gitignore` miatt nincsenek a repóban, a letöltött Kaggle adatbázist ide kell bemásolni).
- `notebooks/` : Jupyter notebookok az adatfeltáráshoz (EDA), vizualizációhoz és a modellezéshez.
- `models/` : A kimentett, betanított gépi tanulási modellek (pl. `.joblib` formátumban).
- `app.py` : A Streamlit webalapú alkalmazás forráskódja (a végső demóhoz).

## 🛠️ Telepítés és Futtatás (Conda)

A projekt egy dedikált Conda környezetet használ a függőségek kezelésére.

1. **Repozitórium klónozása:**
   ```bash
   git clone https://github.com/DGeriOE/CPU_Performance_Predictor.git
   cd CPU_Performance_Predictor
   ```

2. **Környezet létrehozása:**
   ```bash
   conda env create -f environment.yml
   ```

3. **Környezet aktiválása:**
   ```bash
   conda activate cpu-perf-pred
   ```

4. **Jupyter Notebook indítása (az 5-9. heti munkához):**
   ```bash
   jupyter notebook
   ```

5. **A Webalkalmazás indítása (később):**
   ```bash
   streamlit run app.py
   ```

## ✅ Ütemterv / Státusz

- [x] 1. fázis: Adatfeltárás (EDA), hiányzó értékek kezelése, adatszivárgás megszüntetése.
- [x] 2. fázis: Feature engineering (pl. dátumok átalakítása, kategóriák One-Hot kódolása).
- [x] 3. fázis: Lineáris regressziós modell építése és kiértékelése ($R^2$, MSE, MAE).
- [x] 4. fázis: Webalkalmazás (Streamlit) fejlesztése és a modell integrálása.
- [x] 5. fázis: Publikálás valós eszközön történő teszteléshez.
