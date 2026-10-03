# -*- coding: utf-8 -*-
"""Bai tap 6: cham diem cho lop rot mon (lop 0) roi so voi lop qua mon (lop 1)."""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/sinh_vien.csv")
X = df[["gio_on"]]
y = df["qua_mon"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
mo_hinh = LogisticRegression().fit(X_train, y_train)
y_pred = mo_hinh.predict(X_test)

pre_qua = precision_score(y_test, y_pred, pos_label=1)
rec_qua = recall_score(y_test, y_pred, pos_label=1)
pre_rot = precision_score(y_test, y_pred, pos_label=0)
rec_rot = recall_score(y_test, y_pred, pos_label=0)

print("Lop duong = qua mon (1):")
print(f"  Precision = {pre_qua:.4f}")
print(f"  Recall    = {rec_qua:.4f}")
print()
print("Lop duong = rot mon (0):")
print(f"  Precision = {pre_rot:.4f}")
print(f"  Recall    = {rec_rot:.4f}")