# Amazon Video Games — Sentiment Classification

**Tujuan:** klasifikasi biner ulasan (negative = rating 1–2, positive = rating 4–5; rating 3 dibuang) dengan metrik utama **Macro-F1**.

## Data
Amazon Reviews 2023, kategori `Video_Games`, hanya ulasan dengan timestamp ≤ 31 Des 2019. Input = judul + teks ulasan.
File mentah dibaca **per baris via streaming** (tidak dimuat utuh); baris valid disampel acak (seed 42, rate 0.15) dan ditulis per chunk ke CSV, lalu dibaca kembali per chunk. Duplikat judul+teks dibuang sebelum split.

## Metode
`prepare_texts` (gabung `judul . teks`, buang HTML/URL, rapikan karakter berulang) → `HashingVectorizer` (uni+bigram, 2^21 fitur) → `TfidfTransformer` (sublinear) → `LogisticRegression(class_weight="balanced")`, ambang keputusan disetel untuk Macro-F1.

**Alasan:** cepat, hemat memori (tanpa kosakata, cocok data besar), baseline kuat untuk teks, reproducible (seed tetap), dan class weight + threshold mencegah model hanya unggul di kelas positif yang mayoritas.

## Pembagian Data
**Split temporal:** 85% ulasan terlama → train (332,060; 1998-12-18 s/d 2019-01-08), 15% terbaru → validasi (58,599; 2019-01-08 s/d 2019-12-31).
**Alasan:** meniru skenario nyata (melatih dari masa lalu, memprediksi ulasan baru), mencegah kebocoran dari masa depan, distribusi kelas validasi dibiarkan alami sehingga evaluasi jujur. Duplikat dibuang & diverifikasi tidak ada teks identik antar-split; idf, `C` (=8.0) dan threshold (=0.48) hanya dipelajari dari train (tuning memakai 10% terbaru dari train), validasi dipakai sekali untuk pelaporan.

## Hasil (data validasi)
**Macro-F1 = 0.9405** (threshold 0.50: 0.9399)

| Kelas | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| negative | 0.8768 | 0.9313 | 0.9032 | 10,630 |
| positive | 0.9846 | 0.9710 | 0.9777 | 47,969 |

Confusion matrix [[TN, FP],[FN, TP]] (baris=aktual neg/pos): [[9900, 730], [1391, 46578]] — gambar: `model/confusion_matrix.png`.

## Menjalankan
1. Buka notebook di Colab → Run all (hasil dibuat otomatis: `model/sentiment_model.joblib`, `model/metrics.json`).
2. Muat model tanpa training ulang (butuh `review_preprocess.py` di folder yang sama):
```python
import joblib, review_preprocess as rp
b = joblib.load("model/sentiment_model.joblib")
rp.predict_sentiment(b, ["Great"], ["Loved every minute."])
```
`pip install -r requirements.txt` bila di luar Colab.
