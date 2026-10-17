# -*- coding: utf-8 -*-
"""Bai tap 3: khao sat anh huong cua tham so lam tron alpha."""

import sys
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("data/tin_nhan.csv")
vec = CountVectorizer(token_pattern=r"(?u)\S+")
X = vec.fit_transform(df["noi_dung"])
y = df["la_rac"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

cac_alpha = [0.001, 0.1, 1.0, 10.0, 100.0]

print(f"{'alpha':>8s} | {'Accuracy':>10s} | {'Precision':>10s} | {'Recall':>10s}")
print("-" * 46)

for alpha in cac_alpha:
    mo_hinh = MultinomialNB(alpha=alpha).fit(X_train, y_train)
    y_pred = mo_hinh.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    print(f"{alpha:8.3f} | {acc:10.4f} | {prec:10.4f} | {rec:10.4f}")

print()
print("Nhan xet:")
print("  Khi alpha rat lon (vi du tien toi vo cung), cac xac suat co dieu kien")
print("  P(tu | lop) = (count + alpha) / (tong + alpha*V) se deu xich lai gan nhau")
print("  va tien ve phan phoi deu 1/V. Khi do, cac tu khong con suc phan loai nua,")
print("  va mo hinh bi chi phoi hoan toan boi xac suat tien nghiem P(lop). Do lop tin")
print("  thuong chiem da so (60%), mo hinh se doan tat ca la tin thuong, khien Recall")
print("  ve tin rac bi giam sut nghiem trong (co the ve 0 khi alpha du lon).")

