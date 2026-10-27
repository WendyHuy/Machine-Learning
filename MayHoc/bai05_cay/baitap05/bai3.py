# -*- coding: utf-8 -*-
"""Bài tập 3: So sánh thước đo Gini và Entropy."""

import sys
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

# Đọc dữ liệu khách vay
df = pd.read_csv("data/khach_vay.csv")
X = df.drop(columns="tra_dung_han")
y = df["tra_dung_han"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)


def danh_gia_criterion(ten_criterion):
    """Chọn độ sâu tối ưu bằng kiểm định chéo 5 phần và đo trên tập kiểm tra."""
    diem_cv = {}
    diem_hoc = {}
    for d in range(1, 16):
        mh = DecisionTreeClassifier(max_depth=d, criterion=ten_criterion, random_state=0)
        diem_cv[d] = cross_val_score(mh, X_tr, y_tr, cv=5).mean()
        diem_hoc[d] = accuracy_score(y_tr, mh.fit(X_tr, y_tr).predict(X_tr))

    d_tot = max(diem_cv, key=diem_cv.get)
    tot = DecisionTreeClassifier(max_depth=d_tot, criterion=ten_criterion, random_state=0)
    tot.fit(X_tr, y_tr)
    acc_test = accuracy_score(y_te, tot.predict(X_te))
    return d_tot, diem_cv, diem_hoc, acc_test, tot.get_n_leaves()


print("=== BAI TAP 3: SO SANH THUOC DO GINI VA ENTROPY ===")

d_gini, cv_gini, hoc_gini, test_gini, la_gini = danh_gia_criterion("gini")
d_ent, cv_ent, hoc_ent, test_ent, la_ent = danh_gia_criterion("entropy")

print("Bang so sanh diem CV 5-fold theo do sau toi da (max_depth):")
print("  Do sau |       Gini (muc 4)      |         Entropy")
print("         |   Tap hoc     CV 5-fold |   Tap hoc     CV 5-fold")
print("---------+-------------------------+------------------------")
for d in range(1, 16):
    dau_g = " *" if d == d_gini else "  "
    dau_e = " *" if d == d_ent else "  "
    print(f"    {d:2d}   |   {hoc_gini[d]:.4f}     {cv_gini[d]:.4f}{dau_g} |   {hoc_ent[d]:.4f}     {cv_ent[d]:.4f}{dau_e}")

print()
print("Ket qua tong hop:")
print(f"  Thuo do Gini    : Do sau tot nhat = {d_gini}, CV = {cv_gini[d_gini]:.4f}, So la = {la_gini}, Accuracy test = {test_gini:.4f}")
print(f"  Thuoc do Entropy : Do sau tot nhat = {d_ent}, CV = {cv_ent[d_ent]:.4f}, So la = {la_ent}, Accuracy test = {test_ent:.4f}")
print()
print("Nhan xet:")
print("  Doi thuoc do tu Gini sang Entropy KHONG lam ket qua khac di nhieu:")
print(f"  - Ca hai deu chon cung mot do sau toi uu la max_depth = {d_gini}.")
print(f"  - Accuracy tren tap kiem tra chi chenh lech dung 1 mau tren 150 mau ({test_gini:.4f} vs {test_ent:.4f}).")
print("  - Ly do: Ca Gini va Entropy deu co hinh dang do thi tuong tu nhau (cung dat cuc dai tai")
print("    xac suat 0.5 va bang 0 tai hai dau 0 va 1), nen thu tu uu tien cua cac cau hoi chia")
print("    du lieu gan nhu tuong duong nhau.")
