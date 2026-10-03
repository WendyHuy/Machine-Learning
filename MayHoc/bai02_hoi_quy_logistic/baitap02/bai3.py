# -*- coding: utf-8 -*-
"""Bai tap 3: ham du_doan(gio) cho mo hinh mot bien."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/sinh_vien.csv")
X = df[["gio_on"]]
y = df["qua_mon"]

mo_hinh = LogisticRegression().fit(X, y)
w = float(mo_hinh.coef_[0][0])
b = float(mo_hinh.intercept_[0])


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def du_doan(gio):
    """In ra z, xac suat qua mon va nhan theo nguong 0.5."""
    z = w * gio + b
    xac_suat = sigmoid(z)
    nhan = 1 if xac_suat >= 0.5 else 0
    print(f"{gio:6.2f} gio | z = {z:7.3f} | xac suat qua = {xac_suat:.4f} | nhan = {nhan}")


print(f"w = {w:.6f}, b = {b:.6f}")
print(f"Moc 50-50: {-b / w:.2f} gio")
print()
for gio in [3, 8, 12.89, 18, 26]:
    du_doan(gio)