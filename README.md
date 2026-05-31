# Asteroit Tehlike Analizi

Streamlit tabanli XGBoost analiz paneli. Uygulama, 20 yorunge ozelligini kullanarak bir asteroitin tehlikeli sinifa ait olma olasiligini tahmin eder.

## Proje Dosyalari

- `app.py`: Streamlit arayuzu ve tahmin akisi.
- `requirements.txt`: Deploy ve lokal kurulum icin Python paketleri.
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

Windows icin alternatif:

```powershell
.\run_app.bat
```

## Streamlit Community Cloud Deploy

1. Projeyi GitHub reposuna yukle.
2. [share.streamlit.io](https://share.streamlit.io/) adresine GitHub hesabinla gir.
3. `Create app` butonuna bas.
4. `Yup, I have an app` secenegini sec.
5. Repository olarak bu projeyi, branch olarak `main`, main file path olarak `app.py` sec.
6. `Advanced settings` icinden Python surumunu `3.12` sec.
7. Deploy et ve loglarda dependency kurulumunun tamamlanmasini bekle.

Streamlit Community Cloud, paketleri `requirements.txt` dosyasindan kurar. Bu projede ek secrets veya veritabani ayari gerekmez.
