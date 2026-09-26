# -*- coding: utf-8 -*-
"""Bài tập 4: thêm số phòng vào mô hình hồi quy tuyến tính."""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/gia_nha.csv")
X = df[["dien_tich", "so_phong"]]
y = df["gia"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mo_hinh = LinearRegression()
mo_hinh.fit(X_train, y_train)
r2 = r2_score(y_test, mo_hinh.predict(X_test))

print(f"R2 tren tap kiem tra = {r2:.4f}")
print("R2 mo hinh mot bien trong muc 6 = 0.9622")
if r2 > 0.9622:
    print("Nhan xet: Them so_phong co ich vi R2 tang.")
else:
    print("Nhan xet: Them so_phong khong co ich vi R2 khong tang.")
