# -*- coding: utf-8 -*-
"""Bài tập 2: Tự tìm câu hỏi gốc trên dữ liệu thật."""

import sys
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Đọc dữ liệu khách vay
df = pd.read_csv("data/khach_vay.csv")
X = df.drop(columns="tra_dung_han")
y = df["tra_dung_han"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)


def gini(nhan):
    """Tính chỉ số Gini cho một tập nhãn nhị phân (0 và 1)."""
    if len(nhan) == 0:
        return 0.0
    p1 = np.mean(nhan)
    return 1.0 - p1 ** 2 - (1.0 - p1) ** 2


def tim_cau_hoi(X_df, y_s):
    """Thử mọi cột và mọi ngưỡng để tìm câu hỏi làm Gini giảm nhiều nhất.

    X_df : DataFrame chứa các đặc trưng của tập học
    y_s  : Series chứa nhãn của tập học
    """
    y_arr = y_s.to_numpy(dtype=float)
    g_goc = gini(y_arr)
    n = len(y_arr)

    tot_giam = -1.0
    tot_cot = None
    tot_nguong = None

    ket_qua_tung_cot = {}

    for cot in X_df.columns:
        vals = X_df[cot].to_numpy(dtype=float)
        cac_muc = np.sort(np.unique(vals))

        giam_max_cot = -1.0
        nguong_max_cot = None

        # Thử các ngưỡng nằm giữa 2 giá trị phân biệt liền nhau
        for a, b in zip(cac_muc[:-1], cac_muc[1:]):
            nguong = (a + b) / 2.0
            di_trai = vals <= nguong
            trai = y_arr[di_trai]
            phai = y_arr[~di_trai]

            g_chia = (len(trai) / n) * gini(trai) + (len(phai) / n) * gini(phai)
            g_giam = g_goc - g_chia

            if g_giam > giam_max_cot:
                giam_max_cot = g_giam
                nguong_max_cot = nguong

            if g_giam > tot_giam:
                tot_giam = g_giam
                tot_cot = cot
                tot_nguong = nguong

        ket_qua_tung_cot[cot] = (nguong_max_cot, giam_max_cot)

    return tot_cot, tot_nguong, tot_giam, g_goc, ket_qua_tung_cot


print("=== BAI TAP 2: TU TIM CAU HOI GOC TREN DU LIEU THAT (450 KHACH) ===")

cot_tot, nguong_tot, giam_tot, g_goc, ds_cot = tim_cau_hoi(X_tr, y_tr)

print(f"Gini cua nut goc (450 khach): {g_goc:.4f}")
print()
print("Cau hoi tot nhat cua tung cot:")
for cot, (nguong, giam) in ds_cot.items():
    print(f"  - Cot {cot:18s} : nguong {nguong:6.3f}, Gini giam = {giam:.4f}")

print()
print(f"-> CAU HOI GOC TOT NHAT DUOC CHON: '{cot_tot} <= {nguong_tot:.2f}' voi Gini giam {giam_tot:.4f}")

# Đối chiếu với DecisionTreeClassifier của scikit-learn
cay_1_tang = DecisionTreeClassifier(max_depth=1, random_state=0)
cay_1_tang.fit(X_tr, y_tr)
nguong_sklearn = cay_1_tang.tree_.threshold[0]
cot_sklearn = X_tr.columns[cay_1_tang.tree_.feature[0]]

print()
print(f"Cau hoi goc do scikit-learn chon : '{cot_sklearn} <= {nguong_sklearn:.2f}'")
if cot_tot == cot_sklearn and np.isclose(nguong_tot, nguong_sklearn):
    print("Nhan xet: Ket qua tu tinh HOAN TOAN TRUNG KHOP voi scikit-learn (thu_nhap <= 15.05)!")
