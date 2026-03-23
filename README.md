# CPU Power Performance Predikció 🚀

Ez a repozitórium egy egyetemi gépi tanulás feladat megoldását tartalmazza. A projekt célja asztali és mobil processzorok (CPU) `powerPerf` (teljesítmény/fogyasztás) arányának megjóslása többváltozós **lineáris regresszió** segítségével.

## 📊 Az Adathalmaz
A feladathoz használt adatbázis a Kaggle platformról származik, amely a PassMark szoftver benchmark eredményeit tartalmazza:
[CPU Benchmarks Dataset (Kaggle)](https://www.kaggle.com/datasets/alanjo/cpu-benchmarks)

*Megjegyzés: A célváltozó (powerPerf) logaritmikus transzformáción (log-transform) esik át a modell betanítása előtt a jobb eloszlás és a lineáris kapcsolatok erősítése érdekében.*

## 📁 Projekt Struktúra
- `data/` : Ide kerülnek a nyers (`raw`) és az előfeldolgozott (`processed`) .csv fájlok. (Ezek a `.gitignore` miatt nincsenek a repóban, a letöltött Kaggle adatbázist ide kell bemásolni).
- `notebooks/` : Jupyter notebookok az adatfeltáráshoz (EDA), vizualizációhoz és a modellezéshez.
- `models/` : A kimentett, betanított gépi tanulási modellek (pl. `.joblib` formátumban).
- `app.py` : A Streamlit webalapú alkalmazás forráskódja (a végső demóhoz).

## 🛠️ Telepítés és Futtatás (Conda)

A projekt egy dedikált Conda környezetet használ a függőségek kezelésére.

1. **Repozitórium klónozása:**
   ```bash
   git clone <a-te-repo-linked>
   cd <a-repo-mappája>
Környezet létrehozása:Bashconda env create -f environment.yml
Környezet aktiválása:Bashconda activate cpu-perf-prediction
Jupyter Notebook indítása (az 5-9. heti munkához):Bashjupyter notebook
A Webalkalmazás indítása (később):Bashstreamlit run app.py
✅ Ütemterv / Státusz[ ] 1. fázis: Adatfeltárás (EDA), hiányzó értékek kezelése, adatszivárgás megszüntetése.[ ] 2. fázis: Feature engineering (pl. dátumok átalakítása, kategóriák One-Hot kódolása).[ ] 3. fázis: Lineáris regressziós modell építése és kiértékelése ($R^2$, MSE, MAE).[ ] 4. fázis: Webalkalmazás (Streamlit) fejlesztése és a modell integrálása.[ ] 5. fázis: Publikálás valós eszközön történő teszteléshez.
