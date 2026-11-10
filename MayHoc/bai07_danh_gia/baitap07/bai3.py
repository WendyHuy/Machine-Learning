# -*- coding: utf-8 -*-
"""Bài tập 3: Chọn ngưỡng theo yêu cầu recall >= 0.8.

Dùng xác suất từ cross_val_predict trên tập học để tìm ngưỡng lớn nhất
đảm bảo recall >= 0.8.
Áp ngưỡng tìm được cho tập kiểm tra một lần duy nhất và đánh giá.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/sinh_vien_bo_hoc.csv")
X = df.drop(columns="bo_hoc")
y = df["bo_hoc"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

lr = make_pipeline(StandardScaler(), LogisticRegression())
lr.fit(X_tr, y_tr)

# 1. Dự đoán xác suất trên tập học qua kiểm định chéo (5-fold)
xs_hoc = cross_val_predict(lr, X_tr, y_tr, cv=5, method="predict_proba")[:, 1]

# Thử các ngưỡng từ 0.05 đến 0.95 với bước nhảy 0.01
cac_nguong = np.round(np.arange(0.01, 0.96, 0.01), 2)
nguong_thoa_man = []

for t in cac_nguong:
    du_hoc = (xs_hoc >= t).astype(int)
    rec_hoc = recall_score(y_tr, du_hoc)
    if rec_hoc >= 0.8:
        nguong_thoa_man.append((t, rec_hoc))

# Tìm ngưỡng lớn nhất trong số các ngưỡng thỏa mãn recall >= 0.8
t_chon, rec_hoc_chon = max(nguong_thoa_man, key=lambda x: x[0])
print(f"Nguong lon nhat dat recall >= 0.8 tren tap hoc: {t_chon:.2f} (recall = {rec_hoc_chon:.4f})")

# 2. Áp dụng ngưỡng đó lên tập kiểm tra một lần duy nhất
xs_te = lr.predict_proba(X_te)[:, 1]
du_te = (xs_te >= t_chon).astype(int)

rec_te = recall_score(y_te, du_te)
prec_te = precision_score(y_te, du_te)
so_bao = du_te.sum()
tp = int(((du_te == 1) & (y_te == 1)).sum())

print("\n=== KET QUA TREN TAP KIEM TRA ===")
print(f"Nguong ap dung       : {t_chon:.2f}")
print(f"Recall tap kiem tra  : {rec_te:.4f} ({tp}/{y_te.sum()} ban bo hoc)")
print(f"Precision            : {prec_te:.4f}")
print(f"So ban bi bao nguy co: {so_bao}")

print("\n=== TRA LOI CAU HOI ===")
print(f"- Recall tren tap kiem tra co dat 0.8 khong? -> {'Co' if rec_te >= 0.8 else 'Khong'}, dat {rec_te:.4f}.")
print(f"- Precision tren tap kiem tra dat: {prec_te:.4f}.")
print(f"- So ban bi bao nguy co la: {so_bao} ban.")
print("- Giai thich: Nguong duoc chon bang cross_val_predict tren tap hoc dam bao tinh khach quan,")
print("  khong dung tap kiem tra de ra quyet dinh (tranh ro ri thong tin / data leakage).")
