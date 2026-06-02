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

Ez a repository egy egyetemi gépi tanulási feladat megoldását tartalmazza. A projekt célja asztali és mobil processzorok (CPU-k) `powerPerf` (teljesítmény/fogyasztás) arányának becslése többváltozós **lineáris regresszió** segítségével.

## 📊 Az adathalmaz
A feladathoz használt adathalmaz a Kaggle platformról származik, amely a PassMark szoftver benchmark-eredményeit tartalmazza:
[CPU Benchmarks Dataset (Kaggle)](https://www.kaggle.com/datasets/alanjo/cpu-benchmarks)

Statisztika és stratégia:
- 3825 sor:
   - **1858** sornál hiányzik a `price` (ezzel együtt a `cpuValue` és a `threadValue` is) --> a `price` pótlásával ez a rész alkotja a **tanítóhalmazt**
   - **685** sornál hiányzik a `TDP` (ezzel együtt a `powerPerf` is) --> a `TDP` pótlásával ez a rész alkotja a végső **tesztelőhalmazt**

*Megjegyzés: Az adathalmaz egy része logaritmikus transzformáción (log-transform) esik át a modell betanítása előtt a jobb eloszlás és a lineáris kapcsolatok erősítése érdekében.*

## 📁 Projektstruktúra
- `data/` : Ide kerülnek a nyers (`raw`) és az előfeldolgozott (`processed`) `.csv` fájlok. (Ezek a `.gitignore` fájl miatt nincsenek a repóban, a letöltött Kaggle adathalmazt ide kell bemásolni).
- `notebooks/` : Jupyter notebookok az adatfeltáráshoz (EDA), a vizualizációhoz és a modellezéshez.
- `models/` : Az elmentett, betanított gépi tanulási modellek (pl. `.joblib` formátumban).
- `app/` : A Streamlit webalapú alkalmazás mappája.
  - `PPApp.py` : A webalkalmazás fő forráskódja.

## 🛠️ Telepítés és futtatás (Conda)

A projekt egy dedikált Conda-környezetet használ a függőségek kezelésére.

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

4. **Jupyter notebook indítása (az 5-9. heti munkához):**
   ```bash
   jupyter notebook
   ```

5. **A webalkalmazás indítása:**
   ```bash
   streamlit run app/PPApp.py
   ```

## ✅ Ütemterv / Állapot

- [x] 1. fázis: Adatfeltárás (EDA), hiányzó értékek kezelése, adatszivárgás megszüntetése.
- [x] 2. fázis: Feature engineering (pl. dátumok átalakítása, kategóriák One-Hot kódolása).
- [x] 3. fázis: Lineáris regressziós modell építése és kiértékelése ($R^2$, MSE, MAE).
- [x] 4. fázis: Webalkalmazás (Streamlit) fejlesztése és a modell integrálása.
- [x] 5. fázis: Publikálás valós eszközön történő teszteléshez.
