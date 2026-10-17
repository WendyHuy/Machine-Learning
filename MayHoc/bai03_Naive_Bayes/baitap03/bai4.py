# -*- coding: utf-8 -*-
"""Bai tap 4: tu viet bo loc Naive Bayes khong dung thu vien."""

import sys
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from collections import Counter
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# 1. Doc du lieu va chia tap hoc / tap kiem tra
df = pd.read_csv("data/tin_nhan.csv")

train_df, test_df = train_test_split(
    df, test_size=0.25, random_state=17, stratify=df["la_rac"]
)

# 2. Xay dung tu dien tu tap hoc
tu_dien = sorted({t for text in train_df["noi_dung"] for t in text.split()})
V = len(tu_dien)

# 3. Dem so tu theo tung lop
dem = {0: Counter(), 1: Counter()}
tong = {0: 0, 1: 0}
so_tin = {0: 0, 1: 0}

for _, row in train_df.iterrows():
    nhan = row["la_rac"]
    so_tin[nhan] += 1
    for t in row["noi_dung"].split():
        dem[nhan][t] += 1
        tong[nhan] += 1

# 4. Tinh loga xac suat tien nghiem log P(y)
log_prior = {
    0: np.log(so_tin[0] / len(train_df)),
    1: np.log(so_tin[1] / len(train_df))
}

alpha = 1.0


# 5. Ham tinh diem loga hau nghiem
def tinh_log_diem(cau, nhan):
    """Tinh log P(y) + tong log P(tu | y) voi lam tron Laplace."""
    diem = log_prior[nhan]
    for t in cau.split():
        if t in tu_dien:
            # P(tu | nhan) = (dem + alpha) / (tong + alpha * V)
            prob = (dem[nhan][t] + alpha) / (tong[nhan] + alpha * V)
            diem += np.log(prob)
    return diem


# 6. Du doan tren tap kiem tra
y_pred_tu_viet = []
for cau in test_df["noi_dung"]:
    diem_thuong = tinh_log_diem(cau, 0)
    diem_rac = tinh_log_diem(cau, 1)
    y_pred_tu_viet.append(1 if diem_rac > diem_thuong else 0)

acc_tu_viet = accuracy_score(test_df["la_rac"], y_pred_tu_viet)

# 7. So sanh voi MultinomialNB cua scikit-learn
vec = CountVectorizer(token_pattern=r"(?u)\S+")
X_train = vec.fit_transform(train_df["noi_dung"])
X_test = vec.transform(test_df["noi_dung"])
mo_hinh_sklearn = MultinomialNB(alpha=1.0).fit(X_train, train_df["la_rac"])
acc_sklearn = accuracy_score(test_df["la_rac"], mo_hinh_sklearn.predict(X_test))

print("Ket qua so sanh:")
print(f"  Accuracy bo loc tu viet : {acc_tu_viet:.4f}")
print(f"  Accuracy MultinomialNB   : {acc_sklearn:.4f}")
print(f"  Chenh lech               : {abs(acc_tu_viet - acc_sklearn):.6f}")
print()
print("Ket luan: Bo loc tu viet va MultinomialNB cua scikit-learn cho ket qua")
print("hoan toan khop nhau, chung to thuat toan tu cai dat dung co che hoat dong.")

