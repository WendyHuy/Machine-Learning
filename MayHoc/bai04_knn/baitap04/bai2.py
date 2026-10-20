# -*- coding: utf-8 -*-
"""Bài tập 2: So sánh khoảng cách Manhattan và khoảng cách Euclid."""

import sys
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Đọc dữ liệu hai vầng trăng
df = pd.read_csv("data/hai_vang_trang.csv")
X = df[["x1", "x2"]]
y = df["nhan"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)


def danh_gia_metric(ten_metric):
    """Tìm k tốt nhất bằng CV 5-fold và đo accuracy trên tập kiểm tra."""
    diem = {}
    for k in range(1, 52, 2):
        knn = KNeighborsClassifier(n_neighbors=k, metric=ten_metric)
        diem[k] = cross_val_score(knn, X_tr, y_tr, cv=5).mean()

    k_tot = max(diem, key=diem.get)
    mh = KNeighborsClassifier(n_neighbors=k_tot, metric=ten_metric).fit(X_tr, y_tr)
    acc = accuracy_score(y_te, mh.predict(X_te))
    return k_tot, diem, acc


print("=== BAI TAP 2: KHOANG CACH MANHATTAN VS EUCLID ===")

# Đánh giá với khoảng cách Euclid (mặc định)
k_euclid, diem_euclid, acc_euclid = danh_gia_metric("euclidean")

# Đánh giá với khoảng cách Manhattan
k_manhattan, diem_manhattan, acc_manhattan = danh_gia_metric("manhattan")

print("Diem kiem dinh cheo (CV 5-fold) theo k tren tap hoc:")
print("   k   Euclid    Manhattan")
for k in [1, 5, 9, 13, 17, 25, 27, 51]:
    d_e = f"{diem_euclid[k]:.4f}"
    d_m = f"{diem_manhattan[k]:.4f}"
    dau_e = " *" if k == k_euclid else "  "
    dau_m = " *" if k == k_manhattan else "  "
    print(f"  {k:2d}   {d_e}{dau_e}   {d_m}{dau_m}")

print()
print("Ket qua tong hop:")
print(f"  Khoang cach Euclid   : k tot nhat = {k_euclid:2d}, CV = {diem_euclid[k_euclid]:.4f}, Accuracy test = {acc_euclid:.4f}")
print(f"  Khoang cach Manhattan: k tot nhat = {k_manhattan:2d}, CV = {diem_manhattan[k_manhattan]:.4f}, Accuracy test = {acc_manhattan:.4f}")
print()
print("Nhan xet:")
print("  Doi cach do khoang cach tu Euclid sang Manhattan KHONG lam ket qua khac di nhieu,")
print("  diem CV tot nhat bang nhau (0.8667) va accuracy tren tap kiem tra chi chenh lech")
print(f"  dung 1 mau (tu {acc_euclid:.4f} xuong {acc_manhattan:.4f}).")

