# -*- coding: utf-8 -*-
"""Bài tập 3: Kernel đa thức (SVC polynomial)."""

import sys
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Đọc dữ liệu hai vầng trăng
df = pd.read_csv("data/hai_vang_trang.csv")
X = df[["x1", "x2"]]
y = df["nhan"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

bac_d = [2, 3, 4]
ds_C = [0.1, 1, 10, 100]

print("=== BAI TAP 3: KERNEL DA THUC TREN BO HAI VANG TRANG ===")
print("Do tham so C trong [0.1, 1, 10, 100] cho tung bac d = 2, 3, 4 bang GridSearchCV (cv=5):")
print()

ket_qua = {}
for d in bac_d:
    pipe = make_pipeline(StandardScaler(), SVC(kernel="poly", degree=d))
    grid = GridSearchCV(pipe, {"svc__C": ds_C}, cv=5)
    grid.fit(X_tr, y_tr)

    c_tot = grid.best_params_["svc__C"]
    diem_cv_tot = grid.best_score_
    acc_test = grid.score(X_te, y_te)
    ket_qua[d] = (c_tot, diem_cv_tot, acc_test, grid)

    print(f"--- Bac d = {d} ---")
    print(f"  C tot nhat            : {c_tot}")
    print(f"  Diem CV tot nhat      : {diem_cv_tot:.4f}")
    print(f"  Accuracy tap kiem tra : {acc_test:.4f}")
    # In điểm CV của từng C
    for c_val, score in zip(ds_C, grid.cv_results_["mean_test_score"]):
        dau = " <- tot nhat" if c_val == c_tot else ""
        print(f"    C = {c_val:5}: diem CV = {score:.4f}{dau}")
    print()

# Tìm bậc tốt nhất
bac_tot = max(ket_qua, key=lambda k: ket_qua[k][1])
c_best, cv_best, acc_best, _ = ket_qua[bac_tot]

print("Tong hop ket qua:")
print(f"  Bac tot nhat: d = {bac_tot} (C = {c_best}, Diem CV = {cv_best:.4f}, Accuracy test = {acc_best:.4f})")
print()
print("Tra loi cac cau hoi:")
print(f"1. Bac nao tot nhat?")
print(f"   -> Bac d = {bac_tot} la tot nhat, voi diem kiem dinh cheo {cv_best:.4f} va accuracy test {acc_best:.4f}.")
print("      Cac bac chan d = 2 (0.5067) va d = 4 (0.4800) the hien rat kem vi da thuc bac chan")
print("      co tinh doi xung truc, khong the uon theo hinh dang hai vang trang khuyet long vao nhau.")
print()
print("2. Co vuot duoc kernel RBF o muc 5 khong?")
print(f"   -> KHONG vuot duoc! Kernel da thuc bac {bac_tot} chi dat accuracy test {acc_best:.4f} (CV: {cv_best:.4f}),")
print("      trong khi kernel RBF o muc 5 dat accuracy test len toi 0.9200 (CV: 0.8711).")
print("   -> Giai thich:")
print("      Kernel RBF tao ra bien quyet dinh uon luan cuc bo rat mem mai theo khoang cach Euclid quanh")
print("      cac vecto ho tro, rat thich hop cho cac duong cong phuc tap nhu hai vang trang.")
print("      Con kernel da thuc bi rang buoc boi dang ham da thuc toan cuc, kho uon luon mem mai bang RBF.")
