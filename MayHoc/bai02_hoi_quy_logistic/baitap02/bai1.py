# -*- coding: utf-8 -*-
"""Bai tap 1: thong ke ty le qua mon theo diem giua ky."""

import pandas as pd

df = pd.read_csv("data/sinh_vien.csv")

# Phep loc theo dieu kien: True/False cho tung dong
nhom_cao = df[df["diem_giua_ky"] >= 7]
nhom_con_lai = df[df["diem_giua_ky"] < 7]

print("So ban co diem giua ky tu 7 tro len:", len(nhom_cao))
print("Ty le qua mon cua nhom diem >= 7   :", round(nhom_cao["qua_mon"].mean(), 4))
print("Ty le qua mon cua nhom con lai     :", round(nhom_con_lai["qua_mon"].mean(), 4))