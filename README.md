# Streamlit Churn Prediction App

Aplikasi web prediksi churn nasabah kartu kredit menggunakan CatBoost hasil optimasi Bayesian Optimization.

## Struktur Folder

```text
streamlit_churn_app/
├── app.py
├── requirements.txt
├── data/
│   └── BankChurners.csv
├── models/
│   ├── catboost_bayes_best_model.cbm        # dibuat setelah menjalankan train_save_model.py
│   └── model_metadata.json                  # dibuat setelah menjalankan train_save_model.py
└── scripts/
    └── train_save_model.py
```

## Cara Menjalankan

1. Install library:

```bash
pip install -r requirements.txt
```

2. Jalankan script training sekali saja untuk menyimpan model terbaik:

```bash
python scripts/train_save_model.py
```

3. Jalankan web Streamlit:

```bash
streamlit run app.py
```

## Catatan

- Web tidak melakukan training ulang.
- Web hanya memuat model terbaik dari folder `models`.
- Dataset utama otomatis dibaca dari folder `data` untuk halaman Dataset.
- Prediksi batch menggunakan CSV baru yang diunggah user.
