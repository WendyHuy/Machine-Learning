# -*- coding: utf-8 -*-
"""Bài tập 1: lọc các căn hộ có diện tích lớn hơn 100 m2."""

import pandas as pd

df = pd.read_csv("data/gia_nha.csv")
nhom_lon = df[df["dien_tich"] > 100]

print("So can co dien tich lon hon 100 m2:", len(nhom_lon))
print(f"Gia trung binh cua nhom: {nhom_lon['gia'].mean():.3f} ty dong")
