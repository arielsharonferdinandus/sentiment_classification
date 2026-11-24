"""Preprocessing & inference helpers untuk Amazon Video Games sentiment model.

Modul ini HARUS tersedia (satu folder dengan notebook / di PYTHONPATH) saat
memuat file model .joblib, karena pipeline mereferensikan `prepare_texts`.
"""
import re

import pandas as pd

_HTML = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+|www\.\S+")
_REPEAT = re.compile(r"(.)\1{3,}")
_WS = re.compile(r"\s+")

LABELS = {0: "negative", 1: "positive"}


def clean_text(s):
    """Bersihkan satu string: HTML, URL, karakter berulang, spasi."""
    if s is None or (isinstance(s, float) and s != s):
        return ""
    s = str(s)
    s = _HTML.sub(" ", s)          # <br />, <b>, dst.
    s = _URL.sub(" url ", s)       # URL -> token generik
    s = _REPEAT.sub(r"\1\1\1", s)  # "sooooo" -> "sooo"
    s = _WS.sub(" ", s)
    return s.strip()


def prepare_texts(X):
    """Gabungkan judul + teks ulasan menjadi satu string per baris.

    X: DataFrame dengan kolom 'title' dan 'text'.
    Return: list[str] -> "<judul> . <teks>" (lowercase dilakukan vectorizer).
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X harus DataFrame dengan kolom 'title' dan 'text'")
    titles = X["title"].tolist()
    texts = X["text"].tolist()
    return [f"{clean_text(t)} . {clean_text(x)}" for t, x in zip(titles, texts)]


def predict_sentiment(bundle, titles, texts, return_proba=False):
    """Prediksi sentimen dari bundle hasil joblib.load().

    titles/texts: str tunggal atau list[str].
    Return: list label ('negative'/'positive'); jika return_proba=True,
    tambahan array P(positive).
    """
    if isinstance(titles, str):
        titles = [titles]
    if isinstance(texts, str):
        texts = [texts]
    df = pd.DataFrame({"title": list(titles), "text": list(texts)})
    proba = bundle["pipeline"].predict_proba(df)[:, 1]
    pred = (proba >= bundle["threshold"]).astype(int)
    labels = [LABELS[int(p)] for p in pred]
    return (labels, proba) if return_proba else labels
