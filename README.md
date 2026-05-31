# Asteroid Hazard Streamlit

Canli uygulama: https://asteroid-hazard.streamlit.app/

XGBoost modeli ile asteroidlerin yorunge ozelliklerinden tehlikeli sinifa ait olma olasiligini hesaplayan Streamlit analiz paneli.

## Ozellikler

- 20 yorunge ozelligi uzerinden tahmin yapar.
- Hazir tehlikeli ve tehlikesiz asteroid senaryolarini tek tikla yukler.
- Tahmin sonucunu olasilik yuzdesi ve risk seviyesi olarak gosterir.
- Model performans metriklerini, confusion matrix gorselini ve oznitelik onemi grafigini sunar.
- Sunum dostu tek sayfalik arayuz olarak tasarlanmistir.

## Model

- Algoritma: XGBoost
- Accuracy: 0.94
- Precision: 0.80
- Recall: 0.86
- F1-score: 0.83
- ROC-AUC: 0.98
- PR-AUC: 0.90

Egitim/test ayrimi 80/20 olarak yapildi. Egitim verisinde SMOTE ile sinif dengelemesi uygulandi.

## Veri Kaynagi

NASA CNEOS / JPL verileri: https://cneos.jpl.nasa.gov/

## Proje Yapisi

- `app.py`: Streamlit arayuzu ve tahmin akisi.
- `requirements.txt`: Gerekli Python paketleri.
- `xgboost_asteroit_modeli.pkl`: Egitilmis XGBoost modeli.
- `asteroit_scaler.pkl`: Egitimde kullanilan scaler.
- `oznitelik_isimleri.pkl`: Modelin bekledigi 20 ozellik sirasi.
- `asteroid_presets.json`: Sunum icin hazir test senaryolari.
- `assets/`: Arka plan, confusion matrix ve oznitelik onemi gorselleri.

## Lokal Calistirma

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
