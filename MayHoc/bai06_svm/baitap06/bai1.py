# -*- coding: utf-8 -*-
"""Bài tập 1: Tính tay khoảng cách tới ranh giới."""

import sys
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

# Tạo 24 điểm như ở mục 2
X, y = make_blobs(n_samples=24, centers=[[2, 2], [6, 6]],
                  cluster_std=0.9, random_state=0)
y = 1 - y

svm = SVC(kernel="linear", C=1000)
svm.fit(X, y)

w = svm.coef_[0]
b = svm.intercept_[0]
w1, w2 = w[0], w[1]
do_dai_w = np.sqrt(w1 ** 2 + w2 ** 2)
do_rong_le = 2.0 / do_dai_w

print("=== BAI TAP 1: TINH TAY KHOANG CACH TOI RANH GIOI ===")
print(f"Tham so mo hinh tu muc 2: w1 = {w1:.4f}, w2 = {w2:.4f}, b = {b:.4f}")
print(f"Do dai vecto w: ||w|| = {do_dai_w:.4f}")
print(f"Do rong cua le: 2 / ||w|| = {do_rong_le:.4f}")
print()


def f_tay(x1, x2):
    """Tính giá trị f(x) = w1*x1 + w2*x2 + b bằng phép nhân và cộng."""
    return w1 * x1 + w2 * x2 + b


def khoang_cach_tay(x1, x2):
    """Tính khoảng cách từ điểm (x1, x2) tới ranh giới: d = |f(x)| / ||w||."""
    return abs(f_tay(x1, x2)) / do_dai_w


# 1. Tính cho hai điểm mới (3, 3) và (5, 4.5)
diem_moi = [(3.0, 3.0), (5.0, 4.5)]
print("1. Gia tri f(x) va khoang cach cua hai diem moi (tinh tay):")
for p in diem_moi:
    f_val = f_tay(p[0], p[1])
    d_val = khoang_cach_tay(p[0], p[1])
    lop_doan = 1 if f_val > 0 else 0
    print(f"   Diem ({p[0]}, {p[1]}): f(x) = {f_val:+.4f}, khoang cach d = {d_val:.4f} -> doan lop {lop_doan}")

print()

# 2. Tính cho ba vectơ hỗ trợ
print(f"2. Khoang cach cua {len(svm.support_)} vecto ho tro toi ranh gioi:")
for i in svm.support_:
    pt = X[i]
    f_val = f_tay(pt[0], pt[1])
    d_val = khoang_cach_tay(pt[0], pt[1])
    print(f"   Diem {i:2d} ({pt[0]:.2f}, {pt[1]:.2f}): f(x) = {f_val:+.4f}, khoang cach d = {d_val:.4f}")

print()
print("Tra loi cau hoi:")
print("Vi sao khoang cach cua ba vecto ho tro bang dung mot nua do rong le?")
print(f"  - Do rong cua le la khoang cach giua hai mep le f(x) = +1 va f(x) = -1: le = 2 / ||w|| = {do_rong_le:.4f}.")
print("  - Duong ranh gioi f(x) = 0 nam chinh giua hai mep le nay.")
print("  - Cac vecto ho tro nam dung tren hai mep le, nghia la |f(x_sv)| = 1.")
print(f"  - Do do, khoang cach tu moi vecto ho tro toi duong ranh gioi la:")
print(f"      d(x_sv) = |f(x_sv)| / ||w|| = 1 / ||w|| = le / 2 = {do_rong_le / 2.0:.4f}.")
print("  -> Chinh vi vay, khoang cach nay luon bang dung mot nua do rong le!")
