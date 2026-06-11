import base64
import json
import random
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "xgboost_asteroid_model.pkl"
SCALER_PATH = BASE_DIR / "asteroid_scaler.pkl"
FEATURES_PATH = BASE_DIR / "feature_names.pkl"
PRESETS_PATH = BASE_DIR / "asteroid_presets.json"
BACKGROUND_PATH = BASE_DIR / "assets" / "asteroid_background.png"
CONFUSION_MATRIX_IMAGE_PATH = BASE_DIR / "assets" / "xgboost_confusion_matrix.png"
FEATURE_IMPORTANCE_IMAGE_PATH = BASE_DIR / "assets" / "feature_importance.png"


FEATURE_METADATA = {
    "Est Dia in KM(min)": {
        "tr": "Tahmini çap (min.)",
        "unit": "km",
    },
    "Est Dia in KM(max)": {
        "tr": "Tahmini çap (maks.)",
        "unit": "km",
    },
    "Epoch Date Close Approach": {
        "tr": "Yakın geçiş epoch tarihi",
        "unit": "ms",
    },
    "Relative Velocity km per sec": {
        "tr": "Bağıl hız",
        "unit": "km/s",
    },
    "Miss Dist.(kilometers)": {
        "tr": "Dünya'ya en yakın mesafe",
        "unit": "km",
    },
    "Orbit ID": {
        "tr": "Yörünge kimliği",
        "unit": "ID",
    },
    "Orbit Uncertainity": {
        "tr": "Yörünge belirsizliği",
        "unit": "U",
    },
    "Jupiter Tisserand Invariant": {
        "tr": "Jüpiter Tisserand değişmezi",
        "unit": "Tj",
    },
    "Epoch Osculation": {
        "tr": "Oskülasyon epoch değeri",
        "unit": "epoch",
    },
    "Eccentricity": {
        "tr": "Dışmerkezlik",
        "unit": "e",
    },
    "Semi Major Axis": {
        "tr": "Yarı büyük eksen",
        "unit": "AU",
    },
    "Inclination": {
        "tr": "Yörünge eğimi",
        "unit": "deg",
    },
    "Asc Node Longitude": {
        "tr": "Çıkış düğümü boylamı",
        "unit": "deg",
    },
    "Orbital Period": {
        "tr": "Yörünge periyodu",
        "unit": "gün",
    },
    "Perihelion Distance": {
        "tr": "Günberi mesafesi",
        "unit": "AU",
    },
    "Perihelion Arg": {
        "tr": "Günberi argümanı",
        "unit": "deg",
    },
    "Aphelion Dist": {
        "tr": "Günöte mesafesi",
        "unit": "AU",
    },
    "Perihelion Time": {
        "tr": "Günberi zamanı",
        "unit": "epoch",
    },
    "Mean Anomaly": {
        "tr": "Ortalama anomali",
        "unit": "deg",
    },
    "Mean Motion": {
        "tr": "Ortalama hareket",
        "unit": "deg/gün",
    },
}


DATASET_SOURCE_TEXT = "Kaggle - NASA Asteroids Classification"
DATASET_SOURCE_URL = "https://www.kaggle.com/datasets/shrutimehta/nasa-asteroids-classification"
NASA_SOURCE_TEXT = "NASA CNEOS / JPL"
NASA_SOURCE_URL = "https://cneos.jpl.nasa.gov/"
MODEL_METRICS = {
    "Model": "XGBoost Classifier",
    "Özellik sayısı": "20",
    "Çıktı": "Sınıf 1: Tehlikeli olma olasılığı",
}
MODEL_PERFORMANCE = {
    "Accuracy": "0.94",
    "Precision": "0.80",
    "Recall": "0.86",
    "F1-score": "0.83",
    "ROC-AUC": "0.98",
    "PR-AUC": "0.90",
    "True Negative": "755",
    "False Positive": "32",
    "False Negative": "21",
    "True Positive": "130",
    "Test veri sayısı": "938",
    "Test dağılımı": "787 tehlikesiz / 151 tehlikeli",
    "Eğitim/test": "80/20",
    "Dengeleme": "Eğitim verisine SMOTE uygulandı",
}


st.set_page_config(
    page_title="Asteroit Tehlike Analizi",
    page_icon="☄️",
    layout="wide",
)


def image_to_base64(path):
    if not path.exists():
        return ""
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def apply_theme():
    background_b64 = image_to_base64(BACKGROUND_PATH)
    background_css = ""
    if background_b64:
        background_css = f"""
        .stApp {{
            background:
                linear-gradient(90deg, rgba(5, 10, 18, 0.94), rgba(5, 10, 18, 0.76)),
                url("data:image/png;base64,{background_b64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        """

    st.markdown(
        f"""
        <style>
        {background_css}

        [data-testid="stHeader"] {{
            background: rgba(0, 0, 0, 0);
        }}

        .block-container {{
            max-width: 1220px;
            padding-top: 2.2rem;
            padding-bottom: 2.5rem;
        }}

        h1, h2, h3, p, label, span, div {{
            letter-spacing: 0;
        }}

        h1 {{
            color: #f8fbff;
            font-size: 2.55rem;
            line-height: 1.08;
        }}

        h2, h3 {{
            color: #edf5ff;
        }}

        .hero-copy {{
            color: #b9c7d8;
            font-size: 1rem;
            max-width: 760px;
            margin-top: -0.35rem;
        }}

        .hero-image-shell {{
            margin-top: 1rem;
            max-width: 780px;
        }}

        .glass-panel {{
            background: rgba(8, 17, 31, 0.78);
            border: 1px solid rgba(148, 197, 255, 0.20);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 18px 60px rgba(0, 0, 0, 0.28);
            backdrop-filter: blur(14px);
        }}

        .preset-title {{
            color: #f7fbff;
            font-weight: 700;
            font-size: 1rem;
            margin-bottom: 0.2rem;
        }}

        .preset-note, .meta-line {{
            color: #b9c7d8;
            font-size: 0.86rem;
            line-height: 1.35;
        }}

        .unit-pill {{
            min-height: 2.45rem;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-top: 0;
            color: #d7e9ff;
            background: rgba(90, 148, 210, 0.16);
            border: 1px solid rgba(148, 197, 255, 0.24);
            border-radius: 7px;
            font-weight: 700;
            font-size: 0.82rem;
            white-space: nowrap;
        }}

        .feature-label {{
            min-height: 3.15rem;
            margin-bottom: 0.2rem;
            display: flex;
            flex-direction: column;
            justify-content: flex-end;
            gap: 0.1rem;
        }}

        .feature-en {{
            color: #f3f8ff;
            font-size: 0.88rem;
            font-weight: 680;
            line-height: 1.16;
            overflow-wrap: anywhere;
        }}

        .feature-tr {{
            color: #9fb4c9;
            font-size: 0.78rem;
            line-height: 1.15;
            overflow-wrap: anywhere;
        }}

        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.7rem;
            margin-top: 0.6rem;
        }}

        .metric-card {{
            background: rgba(3, 11, 22, 0.66);
            border: 1px solid rgba(148, 197, 255, 0.18);
            border-radius: 8px;
            padding: 0.8rem;
        }}

        .metric-label {{
            color: #95a9bd;
            font-size: 0.78rem;
            margin-bottom: 0.2rem;
        }}

        .metric-value {{
            color: #f8fbff;
            font-weight: 750;
            font-size: 1rem;
        }}

        .performance-grid {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.55rem;
            margin-top: 0.65rem;
        }}

        .performance-card {{
            background: rgba(3, 11, 22, 0.58);
            border: 1px solid rgba(148, 197, 255, 0.16);
            border-radius: 8px;
            padding: 0.68rem;
        }}

        .performance-value {{
            color: #f8fbff;
            font-weight: 800;
            font-size: 1.05rem;
            line-height: 1.1;
        }}

        .performance-note {{
            color: #b9c7d8;
            font-size: 0.82rem;
            line-height: 1.35;
            margin-top: 0.72rem;
        }}

        .result-box {{
            background: rgba(8, 17, 31, 0.82);
            border: 1px solid rgba(148, 197, 255, 0.22);
            border-radius: 8px;
            padding: 1rem 1.1rem;
            margin-top: 1rem;
        }}

        .result-percentage {{
            color: #f8fbff;
            font-size: 2.4rem;
            font-weight: 800;
            line-height: 1.05;
            margin-bottom: 0.25rem;
        }}

        .result-subtitle {{
            color: #b9c7d8;
            font-size: 0.95rem;
        }}

        .stButton > button {{
            border-radius: 8px;
            font-weight: 750;
        }}

        div[data-testid="stNumberInput"] input {{
            background: rgba(4, 12, 24, 0.86);
            color: #f8fbff;
            border-color: rgba(148, 197, 255, 0.22);
        }}

        @media (max-width: 760px) {{
            .metric-grid {{
                grid-template-columns: 1fr;
            }}

            h1 {{
                font-size: 2rem;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    return model, scaler, feature_names


@st.cache_data(show_spinner=False)
def load_presets(presets_mtime):
    if not PRESETS_PATH.exists():
        return {}

    with PRESETS_PATH.open("r", encoding="utf-8") as preset_file:
        data = json.load(preset_file)

    if not isinstance(data, dict):
        return {}

    return data


def get_scaler_feature_count(scaler):
    return getattr(scaler, "n_features_in_", None)


def get_model_feature_count(model):
    feature_count = getattr(model, "n_features_in_", None)
    if feature_count is not None:
        return feature_count

    get_booster = getattr(model, "get_booster", None)
    if callable(get_booster):
        try:
            booster = get_booster()
            num_features = getattr(booster, "num_features", None)
            if callable(num_features):
                return num_features()
        except Exception:
            return None

    return None


def validate_artifacts(model, scaler, feature_names):
    errors = []

    if not isinstance(feature_names, (list, tuple)) or not feature_names:
        errors.append("Özellik isimleri dosyası boş veya liste formatında değil.")
        return errors

    if len(feature_names) != 20:
        errors.append(
            f"Özellik listesi 20 kolon içermeli, mevcut kolon sayısı: {len(feature_names)}."
        )

    if not hasattr(scaler, "transform"):
        errors.append("Scaler objesinde transform() metodu bulunamadı.")

    if not hasattr(model, "predict_proba"):
        errors.append("Model objesinde predict_proba() metodu bulunamadı.")

    scaler_feature_count = get_scaler_feature_count(scaler)
    if scaler_feature_count is not None and scaler_feature_count != len(feature_names):
        errors.append(
            "Scaler ile özellik listesi uyumsuz: "
            f"scaler {scaler_feature_count}, liste {len(feature_names)} özellik bekliyor."
        )

    model_feature_count = get_model_feature_count(model)
    if model_feature_count is not None and model_feature_count != len(feature_names):
        errors.append(
            "Model ile özellik listesi uyumsuz: "
            f"model {model_feature_count}, liste {len(feature_names)} özellik bekliyor."
        )

    return errors


def get_feature_label(feature_name):
    metadata = FEATURE_METADATA.get(feature_name, {})
    translation = metadata.get("tr", "Türkçe karşılık eklenecek")
    return f"{feature_name} ({translation})"


def render_feature_label(feature_name):
    metadata = FEATURE_METADATA.get(feature_name, {})
    translation = metadata.get("tr", "Türkçe karşılık eklenecek")
    st.markdown(
        f"""
        <div class="feature-label">
            <div class="feature-en">{feature_name}</div>
            <div class="feature-tr">({translation})</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_feature_unit(feature_name):
    return FEATURE_METADATA.get(feature_name, {}).get("unit", "-")


def get_hazard_probability(model, scaled_data):
    probabilities = model.predict_proba(scaled_data)

    if len(probabilities) == 0 or len(probabilities[0]) == 0:
        raise ValueError("Model olasılık sonucu boş geldi.")

    classes = getattr(model, "classes_", None)
    if classes is not None:
        class_list = list(classes)
        if 1 in class_list:
            hazard_index = class_list.index(1)
        elif "1" in class_list:
            hazard_index = class_list.index("1")
        else:
            raise ValueError("Model sınıfları içinde tehlikeli sınıf (1) bulunamadı.")
    elif len(probabilities[0]) >= 2:
        hazard_index = 1
    else:
        raise ValueError("Tehlikeli sınıf olasılığı okunamadı.")

    return float(probabilities[0][hazard_index])


def render_probability(probability):
    percentage = probability * 100

    st.markdown(
        f"""
        <div class="result-box">
            <div class="result-percentage">{percentage:.2f}%</div>
            <div class="result-subtitle">Tehlikeli olma olasılığı</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if probability >= 0.70:
        st.error("Yüksek risk: Bu asteroit tehlikeli sınıfa yakın görünüyor.")
    elif probability >= 0.40:
        st.warning("Orta risk: Sonuç dikkatli incelenmeli.")
    else:
        st.success("Düşük risk: Bu asteroit tehlikeli sınıftan uzak görünüyor.")


def render_model_info(model, feature_names):
    classes = getattr(model, "classes_", "Bilinmiyor")
    model_name = type(model).__name__
    now_text = datetime.now().strftime("%d.%m.%Y %H:%M:%S")

    st.markdown(
        f"""
        <div class="glass-panel">
            <div class="preset-title">🛰️ Kaynak ve sistem bilgisi</div>
            <div class="meta-line"><strong>Veri seti:</strong> <a href="{DATASET_SOURCE_URL}" target="_blank">{DATASET_SOURCE_TEXT}</a></div>
            <div class="meta-line"><strong>Kurumsal kaynak:</strong> <a href="{NASA_SOURCE_URL}" target="_blank">{NASA_SOURCE_TEXT}</a></div>
            <div class="meta-line"><strong>Sistem zamanı:</strong> {now_text}</div>
            <div class="meta-line"><strong>Model nesnesi:</strong> {model_name}</div>
            <div class="meta-line"><strong>Sınıflar:</strong> {classes}</div>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-label">Model</div>
                    <div class="metric-value">{MODEL_METRICS["Model"]}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Özellik sayısı</div>
                    <div class="metric-value">{len(feature_names)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Çıktı</div>
                    <div class="metric-value">Sınıf 1 olasılığı</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="glass-panel" style="margin-top: 0.8rem;">
            <div class="preset-title">📈 Model performansı</div>
            <div class="performance-grid">
                <div class="performance-card">
                    <div class="metric-label">Accuracy</div>
                    <div class="performance-value">{MODEL_PERFORMANCE["Accuracy"]}</div>
                </div>
                <div class="performance-card">
                    <div class="metric-label">Precision</div>
                    <div class="performance-value">{MODEL_PERFORMANCE["Precision"]}</div>
                </div>
                <div class="performance-card">
                    <div class="metric-label">Recall</div>
                    <div class="performance-value">{MODEL_PERFORMANCE["Recall"]}</div>
                </div>
                <div class="performance-card">
                    <div class="metric-label">F1-score</div>
                    <div class="performance-value">{MODEL_PERFORMANCE["F1-score"]}</div>
                </div>
                <div class="performance-card">
                    <div class="metric-label">ROC-AUC</div>
                    <div class="performance-value">{MODEL_PERFORMANCE["ROC-AUC"]}</div>
                </div>
                <div class="performance-card">
                    <div class="metric-label">PR-AUC</div>
                    <div class="performance-value">{MODEL_PERFORMANCE["PR-AUC"]}</div>
                </div>
            </div>
            <div class="performance-note">
                <strong>Confusion matrix:</strong> [[755, 32], [21, 130]]
                · TN: {MODEL_PERFORMANCE["True Negative"]}
                · FP: {MODEL_PERFORMANCE["False Positive"]}
                · FN: {MODEL_PERFORMANCE["False Negative"]}
                · TP: {MODEL_PERFORMANCE["True Positive"]}
            </div>
            <div class="performance-note">
                <strong>Test verisi:</strong> {MODEL_PERFORMANCE["Test veri sayısı"]}
                kayıt ({MODEL_PERFORMANCE["Test dağılımı"]}).
                <strong>Eğitim/test:</strong> {MODEL_PERFORMANCE["Eğitim/test"]}.
                {MODEL_PERFORMANCE["Dengeleme"]}.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if CONFUSION_MATRIX_IMAGE_PATH.exists():
        with st.expander("Confusion matrix görseli"):
            st.image(
                str(CONFUSION_MATRIX_IMAGE_PATH),
                caption="XGBoost confusion matrix",
                width="stretch",
            )


def normalize_preset_values(raw_values, feature_names):
    if isinstance(raw_values, dict) and isinstance(raw_values.get("features"), dict):
        raw_values = raw_values["features"]

    if isinstance(raw_values, dict):
        return {
            feature_name: float(raw_values.get(feature_name, 0.0))
            for feature_name in feature_names
        }

    if isinstance(raw_values, list) and len(raw_values) == len(feature_names):
        return {
            feature_name: float(raw_values[index])
            for index, feature_name in enumerate(feature_names)
        }

    raise ValueError("Preset 20 özellik içeren liste veya kolon adı/değer sözlüğü olmalı.")


def initialize_input_state(feature_names):
    for index, feature_name in enumerate(feature_names):
        key = f"feature_{index}"
        if key not in st.session_state:
            st.session_state[key] = 0.0


def apply_preset_to_state(preset_name, presets, feature_names):
    preset_values = normalize_preset_values(presets[preset_name], feature_names)
    for index, feature_name in enumerate(feature_names):
        st.session_state[f"feature_{index}"] = preset_values[feature_name]
    st.session_state["selected_preset_name"] = preset_name


def reset_input_state(feature_names):
    for index in range(len(feature_names)):
        st.session_state[f"feature_{index}"] = 0.0
    st.session_state["selected_preset_name"] = "-"


def get_numbered_presets(presets, prefix):
    numbered = []
    for name in presets:
        if not name.startswith(f"{prefix}-"):
            continue
        suffix = name.rsplit("-", 1)[-1]
        if suffix.isdigit():
            numbered.append((int(suffix), name))
    return [name for _, name in sorted(numbered)]


def render_preset_loader(label, preset_names, presets, feature_names):
    st.markdown(f'<div class="preset-note">{label}: {len(preset_names)} kayıt</div>', unsafe_allow_html=True)
    if not preset_names:
        st.caption("Bu sınıfta kayıt yok.")
        return

    max_index = len(preset_names)
    selected_index = st.number_input(
        f"{label} örnek no",
        min_value=1,
        max_value=max_index,
        value=1,
        step=1,
        key=f"{label}_preset_index",
    )

    preset_name = preset_names[int(selected_index) - 1]
    load_col, random_col = st.columns(2)
    with load_col:
        if st.button(f"{label} yükle", width="stretch", key=f"{label}_load"):
            apply_preset_to_state(preset_name, presets, feature_names)
            st.toast(f"{preset_name} yüklendi.", icon="✅")
    with random_col:
        if st.button("Rastgele", width="stretch", key=f"{label}_random"):
            random_name = random.choice(preset_names)
            apply_preset_to_state(random_name, presets, feature_names)
            st.toast(f"{random_name} yüklendi.", icon="🎲")


def render_preset_panel(feature_names):
    presets_mtime = PRESETS_PATH.stat().st_mtime if PRESETS_PATH.exists() else None
    presets = load_presets(presets_mtime)

    with st.container(border=True):
        st.markdown('<div class="preset-title">🚀 Hazır asteroit seçimi</div>', unsafe_allow_html=True)

        if presets:
            tehlikeli_presets = get_numbered_presets(presets, "tehlikeli")
            tehlikesiz_presets = get_numbered_presets(presets, "tehlikesiz")
            render_preset_loader("Tehlikeli", tehlikeli_presets, presets, feature_names)
            render_preset_loader("Tehlikesiz", tehlikesiz_presets, presets, feature_names)
            st.caption(f"Son yüklenen: {st.session_state.get('selected_preset_name', '-')}")
        else:
            st.info("Hazır örnekler temizlendi. Yeni pkl/örnek dosyasını verdiğinde burası dolacak.")

        st.button(
            "Alanları sıfırla",
            width="stretch",
            on_click=reset_input_state,
            args=(feature_names,),
        )


def get_current_input_values(feature_names):
    return {
        feature_name: st.session_state[f"feature_{index}"]
        for index, feature_name in enumerate(feature_names)
    }


apply_theme()

try:
    model, scaler, feature_names = load_artifacts()
except Exception as exc:
    st.error("Model, scaler veya özellik dosyası yüklenemedi.")
    st.exception(exc)
    st.stop()

artifact_errors = validate_artifacts(model, scaler, feature_names)
if artifact_errors:
    st.error("Uygulama başlatılamadı. Dosyalar arasında uyumsuzluk var.")
    for error in artifact_errors:
        st.write(f"- {error}")
    st.stop()

initialize_input_state(feature_names)

header_left, header_right = st.columns([2.05, 1])
with header_left:
    st.title("☄️ Asteroit Tehlike Analizi")
    st.markdown(
        '<p class="hero-copy">XGBoost modeli ile yörünge özelliklerinden tehlikeli asteroit olasılığını hesaplayan analiz paneli.</p>',
        unsafe_allow_html=True,
    )
    if FEATURE_IMPORTANCE_IMAGE_PATH.exists():
        st.markdown('<div class="hero-image-shell">', unsafe_allow_html=True)
        st.image(
            str(FEATURE_IMPORTANCE_IMAGE_PATH),
            caption="XGBoost modelinde en kritik ilk 10 yörünge özniteliği",
            width="stretch",
        )
        st.markdown("</div>", unsafe_allow_html=True)

with header_right:
    render_preset_panel(feature_names)

st.divider()

st.subheader("Yörünge Özellikleri")

columns = st.columns(3)
for index, feature_name in enumerate(feature_names):
    with columns[index % 3]:
        render_feature_label(feature_name)
        input_col, unit_col = st.columns([0.76, 0.24])
        with input_col:
            st.number_input(
                label=feature_name,
                step=0.01,
                format="%.6f",
                key=f"feature_{index}",
                label_visibility="collapsed",
            )
        with unit_col:
            st.markdown(
                f'<div class="unit-pill">{get_feature_unit(feature_name)}</div>',
                unsafe_allow_html=True,
            )

st.divider()

action_col, info_col = st.columns([1, 1.15])

with action_col:
    st.subheader("🔎 Analiz")
    st.write("Girdi değerlerini kontrol ettikten sonra modeli çalıştır.")

    if st.button("Tehlike Analizi Yap", type="primary", width="stretch"):
        input_values = get_current_input_values(feature_names)
        input_data = pd.DataFrame([input_values], columns=feature_names)

        try:
            scaled_data = scaler.transform(input_data)
            hazard_probability = get_hazard_probability(model, scaled_data)
        except Exception as exc:
            st.error("Tahmin yapılırken bir hata oluştu.")
            st.exception(exc)
        else:
            render_probability(hazard_probability)
            st.session_state["last_probability"] = hazard_probability
            st.session_state["last_prediction_time"] = datetime.now().strftime(
                "%d.%m.%Y %H:%M:%S"
            )

    if "last_probability" in st.session_state:
        st.caption(
            f"Son tahmin zamanı: {st.session_state.get('last_prediction_time', '-')}"
        )

with info_col:
    render_model_info(model, feature_names)
