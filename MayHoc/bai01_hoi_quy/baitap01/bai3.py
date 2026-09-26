# -*- coding: utf-8 -*-
"""Bài tập 3: hồi quy giá theo tuổi nhà bằng công thức bình phương tối thiểu."""

import pandas as pd

df = pd.read_csv("data/gia_nha.csv")
x = df["tuoi_nha"].to_numpy()
y = df["gia"].to_numpy()

x_tb = x.mean()
y_tb = y.mean()
w = ((x - x_tb) * (y - y_tb)).sum() / ((x - x_tb) ** 2).sum()
b = y_tb - w * x_tb

print(f"He so goc w = {w:.6f}")
print(f"He so chan b = {b:.6f}")
print("Nhan xet: w am, nghia la tuoi nha tang thi gia du doan co xu huong giam.")
print("Cac diem du lieu phan tan kha rong, nen rieng tuoi nha chua cho thay moi quan he chat.")
