# -*- coding: utf-8 -*-
"""Bai tap 5: do nguong cho F1 cao nhat."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/sinh_vien.csv")
X = df[["gio_on"]]
y = df["qua_mon"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
mo_hinh = LogisticRegression().fit(X_train, y_train)
p = mo_hinh.predict_proba(X_test)[:, 1]

print("Nguong   F1")
nguong_tot_nhat, f1_cao_nhat = None, -1.0
for nguong in np.arange(0.05, 1.0, 0.05):
    y_pred = (p >= nguong).astype(int)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    print(f"  {nguong:.2f}   {f1:.4f}")
    # Dung dau > (khong phai >=) de khi hoa thi giu nguong nho hon
    if f1 > f1_cao_nhat:
        f1_cao_nhat = f1
        nguong_tot_nhat = nguong

print()
print(f"Nguong cho F1 cao nhat: {nguong_tot_nhat:.2f} (F1 = {f1_cao_nhat:.4f})")