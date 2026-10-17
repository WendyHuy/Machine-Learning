# -*- coding: utf-8 -*-
"""Bai tap 6: lop hiem va bay accuracy khi du lieu mat can bang."""

import sys
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# 1. Tao tap du lieu mat can bang nang: 180 tin thuong + 20 tin rac dau tien
df = pd.read_csv("data/tin_nhan.csv")
df_thuong = df[df["la_rac"] == 0]
df_rac = df[df["la_rac"] == 1].head(20)
df_moi = pd.concat([df_thuong, df_rac], ignore_index=True)

print("Kich thuoc bo du lieu moi:", df_moi.shape)
print("Phan bo nhan:")
print(df_moi["la_rac"].value_counts())
print()

# 2. Vector hoa va chia tap hoc / kiem tra
vec = CountVectorizer(token_pattern=r"(?u)\S+")
X = vec.fit_transform(df_moi["noi_dung"])
y = df_moi["la_rac"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

# 3. Huan luyen lai va du doan
mo_hinh = MultinomialNB(alpha=1.0).fit(X_train, y_train)
y_pred = mo_hinh.predict(X_test)

# 4. Tinh toan cac chi so danh gia
acc = accuracy_score(y_test, y_pred)
rec_rac = recall_score(y_test, y_pred, pos_label=1)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print(f"Accuracy               = {acc:.4f}")
print(f"Recall (lop rac = 1)   = {rec_rac:.4f}")
print(f"Ma tran nham lan       : TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print()

print("Nhan xet ve 'Bay Accuracy':")
print("  - Accuracy dat toi 0.9800 (98%), tao cam giac mo hinh gan nhu hoan hao.")
print("  - Tuy nhien, tren tap kiem tra chi co 5 tin rac, thi mo hinh da de lot 1 tin")
print("    (FN = 1), khien Recall cua lop rac chi con 0.8000 (80%).")
print("  - Khi du lieu mat can bang nang (tin thuong chiem 90%), mot bo loc 'ngu ngoc' chi can")
print("    doan tat ca la tin thuong thi Accuracy van dat toi 90.00% du khong loc duoc bat ky")
print("    tin rac nao.")
print("  - Do do, Accuracy KHONG CON phan anh dung chat luong bo loc trong truong hop lop hiem.")
print("    De danh gia bo loc thu rac, can phai dua vao Recall (kha nang bat sach rac) va")
print("    Precision (tranh xoa nham thu quan trong).")

