import os
import json
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from catboost import CatBoostClassifier

# SHAP bersifat opsional. Jika belum ter-install, web tetap bisa prediksi.
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# =====================================================
# KONFIGURASI HALAMAN
# =====================================================
st.set_page_config(
    page_title="Prediksi Churn Nasabah Kartu Kredit",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# PATH FILE
# =====================================================
DATA_PATH = "data/BankChurners.csv"
MODEL_PATH = "models/catboost_bayes_best_model.cbm"
METADATA_PATH = "models/model_metadata.json"

TARGET_COL = "Attrition_Flag"
DROP_COLS = [
    "CLIENTNUM",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2",
]

# Berdasarkan notebook penelitian
DEFAULT_THRESHOLD = 0.37
BEST_PARAMS_DEFAULT = {
    "iterations": 435,
    "learning_rate": 0.08545901744057517,
    "depth": 5,
    "l2_leaf_reg": 3.3903849118545817,
}
METRICS_DEFAULT = {
    "Accuracy": 0.9555774925962488,
    "Precision": 0.8100263852242744,
    "Recall": 0.9446153846153846,
    "F1-Score": 0.8721590909090909,
    "F2-Score": 0.9142346634901727,
    "ROC-AUC": 0.992327,
}

# Urutan fitur harus sama dengan data training
FEATURE_COLUMNS = [
    "Customer_Age",
    "Gender",
    "Dependent_count",
    "Education_Level",
    "Marital_Status",
    "Income_Category",
    "Card_Category",
    "Months_on_book",
    "Total_Relationship_Count",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
    "Credit_Limit",
    "Total_Revolving_Bal",
    "Avg_Open_To_Buy",
    "Total_Amt_Chng_Q4_Q1",
    "Total_Trans_Amt",
    "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1",
    "Avg_Utilization_Ratio",
]

CATEGORICAL_COLUMNS = [
    "Gender",
    "Education_Level",
    "Marital_Status",
    "Income_Category",
    "Card_Category",
]

NUMERICAL_COLUMNS = [col for col in FEATURE_COLUMNS if col not in CATEGORICAL_COLUMNS]

# =====================================================
# STYLE CSS
# =====================================================
st.markdown(
    """
    <style>
    .main {background-color: #f7f9fc;}
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .title-box {
        background: linear-gradient(135deg, #1f4e79 0%, #2f80ed 100%);
        color: white;
        padding: 28px;
        border-radius: 20px;
        margin-bottom: 22px;
        box-shadow: 0 8px 24px rgba(31, 78, 121, 0.18);
    }
    .title-box h1 {margin-bottom: 8px; font-size: 34px;}
    .title-box p {font-size: 17px; margin-bottom: 0px;}
    .info-card {
        background-color: white;
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.06);
        border: 1px solid #edf0f5;
        margin-bottom: 15px;
    }
    .small-muted {color: #6b7280; font-size: 14px;}
    .risk-high {background-color:#ffe5e5; color:#9f1239; padding:12px; border-radius:12px; font-weight:700; text-align:center;}
    .risk-mid {background-color:#fff5cc; color:#92400e; padding:12px; border-radius:12px; font-weight:700; text-align:center;}
    .risk-low {background-color:#e7f8ee; color:#166534; padding:12px; border-radius:12px; font-weight:700; text-align:center;}
    div[data-testid="stMetricValue"] {font-size: 25px;}
    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================
# FUNGSI DATA DAN MODEL
# =====================================================
@st.cache_data
def load_raw_data(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File dataset tidak ditemukan: {path}")
    return pd.read_csv(path)


def preprocess_dataset(df: pd.DataFrame, has_target: bool = True) -> pd.DataFrame:
    """Preprocessing sesuai notebook penelitian."""
    data = df.copy()

    # Hapus kolom yang tidak digunakan
    data = data.drop(columns=[col for col in DROP_COLS if col in data.columns], errors="ignore")

    # Transformasi target jika tersedia
    if has_target and TARGET_COL in data.columns:
        target_as_text = data[TARGET_COL].astype(str).str.strip()
        mapped_target = target_as_text.map({
            "Existing Customer": 0,
            "Attrited Customer": 1,
            "Existing": 0,
            "Attrited": 1,
            "0": 0,
            "1": 1,
            "Tidak Churn": 0,
            "Churn": 1,
        })

        # Jika target sudah numerik, tetap dipertahankan
        if mapped_target.isna().all():
            data[TARGET_COL] = pd.to_numeric(data[TARGET_COL], errors="coerce")
        else:
            data[TARGET_COL] = mapped_target

    # Rapikan tipe data kategorikal
    for col in CATEGORICAL_COLUMNS:
        if col in data.columns:
            data[col] = data[col].astype(str).str.strip()

    return data


def prepare_features_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """Menyiapkan fitur agar urutan dan kolomnya sama dengan data training."""
    data = preprocess_dataset(df, has_target=False)

    if TARGET_COL in data.columns:
        data = data.drop(columns=[TARGET_COL])

    missing_cols = [col for col in FEATURE_COLUMNS if col not in data.columns]
    if missing_cols:
        raise ValueError(
            "Kolom berikut belum ada pada data input: " + ", ".join(missing_cols)
        )

    data = data[FEATURE_COLUMNS].copy()

    for col in NUMERICAL_COLUMNS:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    for col in CATEGORICAL_COLUMNS:
        data[col] = data[col].astype(str).str.strip()

    if data[NUMERICAL_COLUMNS].isna().any().any():
        missing_summary = data[NUMERICAL_COLUMNS].isna().sum()
        missing_summary = missing_summary[missing_summary > 0]
        raise ValueError(
            "Terdapat nilai kosong/tidak valid pada kolom numerik: "
            + ", ".join(missing_summary.index.tolist())
        )

    return data


@st.cache_resource
def load_model() -> CatBoostClassifier | None:
    if not os.path.exists(MODEL_PATH):
        return None
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    return model


@st.cache_data
def load_metadata() -> dict:
    metadata = {
        "threshold": DEFAULT_THRESHOLD,
        "best_params": BEST_PARAMS_DEFAULT,
        "metrics": METRICS_DEFAULT,
    }
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                saved_metadata = json.load(f)
            metadata.update(saved_metadata)
        except Exception:
            pass
    return metadata


def predict_churn(model: CatBoostClassifier, features: pd.DataFrame, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    prob_churn = model.predict_proba(features)[:, 1]
    pred = (prob_churn >= threshold).astype(int)
    return pred, prob_churn


def risk_category(prob: float) -> str:
    if prob >= 0.70:
        return "Tinggi"
    if prob >= 0.40:
        return "Sedang"
    return "Rendah"


def risk_html(category: str) -> str:
    if category == "Tinggi":
        return '<div class="risk-high">Risiko Churn Tinggi</div>'
    if category == "Sedang":
        return '<div class="risk-mid">Risiko Churn Sedang</div>'
    return '<div class="risk-low">Risiko Churn Rendah</div>'


def format_prediction_label(pred: int) -> str:
    return "Churn" if int(pred) == 1 else "Tidak Churn"


def get_unique_or_default(df: pd.DataFrame, col: str, default_values: list[str]) -> list[str]:
    if col in df.columns:
        values = sorted([str(x) for x in df[col].dropna().unique().tolist()])
        return values if values else default_values
    return default_values


def show_header(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="title-box">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_model_warning(model):
    if model is None:
        st.warning(
            "Model belum ditemukan di folder `models/catboost_bayes_best_model.cbm`. "
            "Jalankan file `scripts/train_save_model.py` sekali untuk menyimpan model terbaik, "
            "lalu buka ulang Streamlit."
        )

# =====================================================
# LOAD DATA
# =====================================================
try:
    raw_df = load_raw_data()
    processed_df = preprocess_dataset(raw_df, has_target=True)
except Exception as e:
    st.error(f"Gagal memuat dataset: {e}")
    st.stop()

model = load_model()
metadata = load_metadata()
THRESHOLD = float(metadata.get("threshold", DEFAULT_THRESHOLD))

# =====================================================
# TOP NAVIGATION MENU
# =====================================================

st.markdown("""
<style>

/* judul */
.top-title {
    text-align: center;
    margin-bottom: 6px;
    color: #111827;
    font-weight: 800;
}

.top-subtitle {
    text-align: center;
    color: #6b7280;
    margin-top: -4px;
    margin-bottom: 28px;
    font-size: 15px;
}

/* tombol menu - default */
div[data-testid="stButton"] > button[kind="secondary"] {
    border-radius: 999px !important;
    border: none !important;
    background: #1e2a3d !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 12px 18px !important;
    min-height: 52px !important;
    transition: all 0.2s ease-in-out !important;
    box-shadow: none !important;
}

/* hover tombol biasa */
div[data-testid="stButton"] > button[kind="secondary"]:hover {
    background: #26344a !important;
    color: white !important;
    border: none !important;
}

/* tombol aktif */
div[data-testid="stButton"] > button[kind="primary"] {
    border-radius: 999px !important;
    border: none !important;
    background: linear-gradient(90deg, #78aefb, #5a9cff) !important;
    color: white !important;
    font-weight: 700 !important;
    padding: 12px 18px !important;
    min-height: 52px !important;
    box-shadow: 0 0 0 2px rgba(120,174,251,0.25) !important;
}

/* hover tombol aktif */
div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: linear-gradient(90deg, #78aefb, #5a9cff) !important;
    color: white !important;
    border: none !important;
}

/* hilangkan outline aneh */
div[data-testid="stButton"] > button:focus {
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(120,174,251,0.25) !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 class="top-title">💳 Prediksi Churn Nasabah Kartu Kredit</h1>
<p class="top-subtitle">
Prediksi Churn Nasabah Kartu Kredit Menggunakan Algoritma CatBoost dengan Bayesian Optimization dan Interpretasi Model Berbasis Shapley Additive Explanations
</p>
""", unsafe_allow_html=True)

# =====================================================
# MENU BUTTON
# =====================================================
if "menu" not in st.session_state:
    st.session_state.menu = "Beranda"

left_space, menu_area, right_space = st.columns([0.8, 5, 0.8])

with menu_area:
    c1, c2, c3, c4 = st.columns(4, gap="medium")

    with c1:
        if st.button(
            "💳 Beranda",
            key="btn_beranda",
            use_container_width=True,
            type="primary" if st.session_state.menu == "Beranda" else "secondary"
        ):
            st.session_state.menu = "Beranda"

    with c2:
        if st.button(
            "📊 Informasi Dataset",
            key="btn_dataset",
            use_container_width=True,
            type="primary" if st.session_state.menu == "Informasi Dataset" else "secondary"
        ):
            st.session_state.menu = "Informasi Dataset"

    with c3:
        if st.button(
            "🔎 Prediksi Manual",
            key="btn_manual",
            use_container_width=True,
            type="primary" if st.session_state.menu == "Prediksi Manual" else "secondary"
        ):
            st.session_state.menu = "Prediksi Manual"

    with c4:
        if st.button(
            "📁 Prediksi Batch CSV",
            key="btn_batch",
            use_container_width=True,
            type="primary" if st.session_state.menu == "Prediksi Batch CSV" else "secondary"
        ):
            st.session_state.menu = "Prediksi Batch CSV"

menu = st.session_state.menu

# =====================================================
# MENU: BERANDA
# =====================================================
if menu == "Beranda":
    show_header(
        "Prediksi Churn Nasabah Kartu Kredit",
        "Sistem prediksi churn menggunakan CatBoost yang dioptimasi dengan Bayesian Optimization dan didukung interpretasi SHAP."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Jumlah Data", f"{processed_df.shape[0]:,}".replace(",", "."))
    with col2:
        st.metric("Jumlah Fitur Model", len(FEATURE_COLUMNS))
    with col3:
        churn_rate = processed_df[TARGET_COL].mean() * 100 if TARGET_COL in processed_df.columns else 0
        st.metric("Persentase Churn", f"{churn_rate:.2f}%")

    st.markdown("""
    <div class="info-card">
    <h3>Deskripsi Sistem</h3>
    <p>
    Sistem ini dirancang untuk memprediksi kemungkinan nasabah kartu kredit mengalami churn berdasarkan data karakteristik dan aktivitas nasabah.
    Model yang digunakan adalah <b>CatBoost Classifier</b> yang telah dioptimasi menggunakan <b>Bayesian Optimization</b> untuk memperoleh kombinasi hyperparameter terbaik.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Ringkasan Metode")
    method_df = pd.DataFrame({
        "Komponen": ["Algoritma", "Optimasi", "Interpretasi", "Target", "Output"],
        "Keterangan": [
            "CatBoost Classifier",
            "Bayesian Optimization",
            "SHAP untuk interpretasi kontribusi fitur",
            "Attrition_Flag",
            "Churn / Tidak Churn dan probabilitas churn",
        ]
    })
    st.dataframe(method_df, use_container_width=True, hide_index=True)

    show_model_warning(model)

# =====================================================
# MENU: DATASET
# =====================================================
elif menu == "Informasi Dataset":
    show_header(
        "Informasi Dataset",
        "Halaman ini menampilkan dataset utama penelitian yang sudah melewati tahap preprocessing."
    )

    st.markdown("""
    <div class="info-card">
    Dataset yang ditampilkan pada halaman ini merupakan dataset hasil preprocessing. Tahap preprocessing meliputi penghapusan kolom <b>CLIENTNUM</b> dan dua kolom <b>Naive_Bayes_Classifier</b>, serta transformasi target <b>Attrition_Flag</b> menjadi nilai numerik, yaitu 0 untuk nasabah tidak churn dan 1 untuk nasabah churn.
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Jumlah Baris", f"{processed_df.shape[0]:,}".replace(",", "."))
    with c2:
        st.metric("Jumlah Kolom", processed_df.shape[1])
    with c3:
        st.metric("Jumlah Fitur Model", len(FEATURE_COLUMNS))
    with c4:
        st.metric("Kolom Target", TARGET_COL)

    st.subheader("Distribusi Variabel Target")
    target_counts = processed_df[TARGET_COL].value_counts().rename(index={0: "Tidak Churn", 1: "Churn"}).reset_index()
    target_counts.columns = ["Status Nasabah", "Jumlah"]
    target_counts["Persentase"] = target_counts["Jumlah"] / target_counts["Jumlah"].sum() * 100

    col_chart1, col_chart2 = st.columns([1.3, 1])
    with col_chart1:
        fig_bar = px.bar(
            target_counts,
            x="Status Nasabah",
            y="Jumlah",
            text="Jumlah",
            title="Jumlah Nasabah Berdasarkan Status Churn",
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(yaxis_title="Jumlah Nasabah", xaxis_title="Status Nasabah")
        st.plotly_chart(fig_bar, use_container_width=True)
    with col_chart2:
        fig_pie = px.pie(
            target_counts,
            names="Status Nasabah",
            values="Jumlah",
            title="Persentase Status Churn",
            hole=0.45,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.dataframe(target_counts, use_container_width=True, hide_index=True)

    st.subheader("Preview Dataset Setelah Preprocessing")
    st.dataframe(processed_df.head(20), use_container_width=True)

    with st.expander("Lihat daftar fitur yang digunakan model"):
        fitur_df = pd.DataFrame({
            "No": range(1, len(FEATURE_COLUMNS) + 1),
            "Nama Fitur": FEATURE_COLUMNS,
            "Tipe": ["Kategorikal" if col in CATEGORICAL_COLUMNS else "Numerik" for col in FEATURE_COLUMNS],
        })
        st.dataframe(fitur_df, use_container_width=True, hide_index=True)

    with st.expander("Statistik deskriptif fitur numerik"):
        st.dataframe(processed_df[NUMERICAL_COLUMNS].describe().T, use_container_width=True)

# =====================================================
# MENU: PREDIKSI MANUAL
# =====================================================
elif menu == "Prediksi Manual":
    show_header(
        "Prediksi Nasabah Manual",
        "Masukkan nilai karakteristik dan aktivitas nasabah untuk memperoleh hasil prediksi churn."
    )

    show_model_warning(model)

    st.markdown("Isi form berikut sesuai data nasabah yang ingin diprediksi.")

    gender_options = get_unique_or_default(processed_df, "Gender", ["F", "M"])
    edu_options = get_unique_or_default(processed_df, "Education_Level", ["Unknown", "Uneducated", "High School", "College", "Graduate", "Post-Graduate", "Doctorate"])
    marital_options = get_unique_or_default(processed_df, "Marital_Status", ["Unknown", "Single", "Married", "Divorced"])
    income_options = get_unique_or_default(processed_df, "Income_Category", ["Unknown", "Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +"])
    card_options = get_unique_or_default(processed_df, "Card_Category", ["Blue", "Silver", "Gold", "Platinum"])

    with st.form("manual_prediction_form"):
        st.subheader("Data Demografi Nasabah")
        col1, col2, col3 = st.columns(3)
        with col1:
            customer_age = st.number_input("Customer Age", min_value=18, max_value=100, value=45)
            gender = st.selectbox("Gender", gender_options)
            dependent_count = st.number_input("Dependent Count", min_value=0, max_value=10, value=3)
        with col2:
            education_level = st.selectbox("Education Level", edu_options)
            marital_status = st.selectbox("Marital Status", marital_options)
            income_category = st.selectbox("Income Category", income_options)
        with col3:
            card_category = st.selectbox("Card Category", card_options)
            months_on_book = st.number_input("Months on Book", min_value=0, max_value=100, value=36)
            total_relationship_count = st.number_input("Total Relationship Count", min_value=1, max_value=10, value=4)

        st.subheader("Data Aktivitas dan Transaksi")
        col4, col5, col6 = st.columns(3)
        with col4:
            months_inactive = st.number_input("Months Inactive 12 Mon", min_value=0, max_value=12, value=2)
            contacts_count = st.number_input("Contacts Count 12 Mon", min_value=0, max_value=10, value=2)
            credit_limit = st.number_input("Credit Limit", min_value=0.0, value=5000.0, step=100.0)
        with col5:
            total_revolving_bal = st.number_input("Total Revolving Bal", min_value=0.0, value=1000.0, step=100.0)
            avg_open_to_buy = st.number_input("Avg Open To Buy", min_value=0.0, value=4000.0, step=100.0)
            total_amt_chng = st.number_input("Total Amt Chng Q4 Q1", min_value=0.0, value=0.75, step=0.01, format="%.3f")
        with col6:
            total_trans_amt = st.number_input("Total Trans Amt", min_value=0.0, value=4000.0, step=100.0)
            total_trans_ct = st.number_input("Total Trans Ct", min_value=0, max_value=200, value=60)
            total_ct_chng = st.number_input("Total Ct Chng Q4 Q1", min_value=0.0, value=0.70, step=0.01, format="%.3f")
            avg_utilization_ratio = st.number_input("Avg Utilization Ratio", min_value=0.0, max_value=1.0, value=0.30, step=0.01, format="%.3f")

        submitted = st.form_submit_button("Prediksi Churn")

    if submitted:
        if model is None:
            st.error("Prediksi belum bisa dilakukan karena file model belum tersedia.")
        else:
            input_df = pd.DataFrame([{
                "Customer_Age": customer_age,
                "Gender": gender,
                "Dependent_count": dependent_count,
                "Education_Level": education_level,
                "Marital_Status": marital_status,
                "Income_Category": income_category,
                "Card_Category": card_category,
                "Months_on_book": months_on_book,
                "Total_Relationship_Count": total_relationship_count,
                "Months_Inactive_12_mon": months_inactive,
                "Contacts_Count_12_mon": contacts_count,
                "Credit_Limit": credit_limit,
                "Total_Revolving_Bal": total_revolving_bal,
                "Avg_Open_To_Buy": avg_open_to_buy,
                "Total_Amt_Chng_Q4_Q1": total_amt_chng,
                "Total_Trans_Amt": total_trans_amt,
                "Total_Trans_Ct": total_trans_ct,
                "Total_Ct_Chng_Q4_Q1": total_ct_chng,
                "Avg_Utilization_Ratio": avg_utilization_ratio,
            }])

            try:
                features = prepare_features_for_prediction(input_df)
                pred, prob = predict_churn(model, features, THRESHOLD)
                prob_churn = float(prob[0])
                pred_label = format_prediction_label(pred[0])
                category = risk_category(prob_churn)

                st.subheader("Hasil Prediksi")
                r1, r2, r3 = st.columns(3)
                with r1:
                    st.metric("Hasil Prediksi", pred_label)
                with r2:
                    st.metric("Probabilitas Churn", f"{prob_churn * 100:.2f}%")
                with r3:
                    st.markdown(risk_html(category), unsafe_allow_html=True)

                if pred_label == "Churn":
                    st.error("Nasabah diprediksi berpotensi churn. Nasabah ini dapat diprioritaskan untuk strategi retensi.")
                else:
                    st.success("Nasabah diprediksi tidak churn. Risiko kehilangan nasabah relatif lebih rendah.")

                # Interpretasi SHAP lokal sederhana
                st.subheader("Interpretasi Singkat")
                if SHAP_AVAILABLE:
                    try:
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(features)
                        if isinstance(shap_values, list):
                            shap_values = shap_values[1]
                        contrib = pd.DataFrame({
                            "Fitur": FEATURE_COLUMNS,
                            "Nilai Input": features.iloc[0].values,
                            "Kontribusi SHAP": np.array(shap_values)[0],
                        })
                        contrib["Arah Pengaruh"] = np.where(
                            contrib["Kontribusi SHAP"] > 0,
                            "Meningkatkan peluang churn",
                            "Menurunkan peluang churn",
                        )
                        contrib["Abs"] = contrib["Kontribusi SHAP"].abs()
                        top_contrib = contrib.sort_values("Abs", ascending=False).head(5).drop(columns="Abs")
                        st.write("Fitur berikut merupakan faktor yang paling memengaruhi hasil prediksi nasabah ini.")
                        st.dataframe(top_contrib, use_container_width=True, hide_index=True)
                    except Exception as shap_error:
                        st.info(f"Interpretasi SHAP belum dapat ditampilkan: {shap_error}")
                else:
                    st.info("Library SHAP belum ter-install. Prediksi tetap berjalan, tetapi interpretasi SHAP tidak ditampilkan.")

            except Exception as e:
                st.error(f"Terjadi kesalahan saat prediksi: {e}")

# =====================================================
# MENU: PREDIKSI BATCH CSV
# =====================================================
elif menu == "Prediksi Batch CSV":
    show_header(
        "Prediksi Batch CSV",
        "Unggah file CSV berisi data nasabah baru untuk memprediksi banyak nasabah sekaligus."
    )

    show_model_warning(model)

    st.markdown("""
    File CSV yang diunggah harus memiliki kolom fitur yang sama dengan dataset training. Kolom target `Attrition_Flag`, `CLIENTNUM`, dan kolom `Naive_Bayes_Classifier` boleh ada di file, karena sistem akan menghapus kolom yang tidak digunakan saat preprocessing.
    """)

    with st.expander("Lihat format kolom yang dibutuhkan"):
        st.dataframe(pd.DataFrame({"Kolom Wajib": FEATURE_COLUMNS}), use_container_width=True, hide_index=True)

    uploaded_file = st.file_uploader("Upload file CSV data nasabah", type=["csv"])

    # Session state dipakai agar hasil prediksi tetap terlihat setelah Streamlit melakukan rerun.
    if "batch_result_df" not in st.session_state:
        st.session_state.batch_result_df = None
    if "batch_summary_df" not in st.session_state:
        st.session_state.batch_summary_df = None
    if "batch_file_name" not in st.session_state:
        st.session_state.batch_file_name = None

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)

            # Jika user mengunggah file baru, hapus hasil lama agar tidak membingungkan.
            if st.session_state.batch_file_name != uploaded_file.name:
                st.session_state.batch_result_df = None
                st.session_state.batch_summary_df = None
                st.session_state.batch_file_name = uploaded_file.name

            st.subheader("Preview Data Upload")
            st.dataframe(batch_df.head(20), use_container_width=True)

            col_btn, col_info = st.columns([1, 2])
            with col_btn:
                run_batch = st.button("Prediksi Data CSV", type="primary", use_container_width=True)
            with col_info:
                st.info(f"Data yang diunggah berisi {len(batch_df)} baris dan {batch_df.shape[1]} kolom.")

            if run_batch:
                if model is None:
                    st.error("Prediksi belum bisa dilakukan karena file model belum tersedia.")
                else:
                    features = prepare_features_for_prediction(batch_df)
                    pred, prob = predict_churn(model, features, THRESHOLD)

                    result_df = batch_df.copy()
                    result_df["Prediksi"] = [format_prediction_label(x) for x in pred]
                    result_df["Probabilitas_Churn"] = prob
                    result_df["Probabilitas_Churn_%"] = (prob * 100).round(2)
                    result_df["Kategori_Risiko"] = [risk_category(float(p)) for p in prob]

                    summary_df = result_df["Prediksi"].value_counts().reset_index()
                    summary_df.columns = ["Prediksi", "Jumlah"]

                    st.session_state.batch_result_df = result_df
                    st.session_state.batch_summary_df = summary_df

            if st.session_state.batch_result_df is not None:
                result_df = st.session_state.batch_result_df
                summary_df = st.session_state.batch_summary_df

                st.success("Prediksi batch berhasil dilakukan. Hasil prediksi ditampilkan pada tabel di bawah ini.")
                st.subheader("Hasil Prediksi Batch")

                total_data = len(result_df)
                total_churn = int((result_df["Prediksi"] == "Churn").sum())
                total_not_churn = int((result_df["Prediksi"] == "Tidak Churn").sum())

                b1, b2, b3 = st.columns(3)
                with b1:
                    st.metric("Total Data", total_data)
                with b2:
                    st.metric("Diprediksi Churn", total_churn)
                with b3:
                    st.metric("Diprediksi Tidak Churn", total_not_churn)

                fig = px.bar(
                    summary_df,
                    x="Prediksi",
                    y="Jumlah",
                    text="Jumlah",
                    title="Ringkasan Hasil Prediksi Batch",
                )
                fig.update_traces(textposition="outside")
                st.plotly_chart(fig, use_container_width=True)

                # Tabel ringkas diletakkan di atas agar hasil prediksi langsung terlihat.
                result_priority_cols = ["Prediksi", "Probabilitas_Churn_%", "Kategori_Risiko"]
                id_cols = [col for col in ["CLIENTNUM"] if col in result_df.columns]
                preview_cols = id_cols + result_priority_cols
                other_cols = [col for col in result_df.columns if col not in preview_cols]

                st.markdown("**Tabel hasil prediksi ringkas:**")
                st.dataframe(result_df[preview_cols], use_container_width=True, hide_index=True)

                with st.expander("Lihat hasil lengkap beserta seluruh kolom input"):
                    st.dataframe(result_df[preview_cols + other_cols], use_container_width=True, hide_index=True)

                csv_buffer = io.StringIO()
                result_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="Download Hasil Prediksi CSV",
                    data=csv_buffer.getvalue(),
                    file_name="hasil_prediksi_churn.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        except Exception as e:
            st.error(f"Gagal memproses file CSV: {e}")
            st.warning("Pastikan file CSV memiliki kolom yang sama dengan fitur model dan nilai numeriknya tidak kosong.")
