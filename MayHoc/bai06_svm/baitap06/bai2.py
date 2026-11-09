# -*- coding: utf-8 -*-
"""Bài tập 2: Thêm một điểm và sự thay đổi của vectơ hỗ trợ."""

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

print("=== BAI TAP 2: THEM MOT DIEM VA SU THAY DOI RANH GIOI ===")

# Mô hình gốc trên 24 điểm
svm_goc = SVC(kernel="linear", C=1000).fit(X, y)
w_goc = svm_goc.coef_[0]
b_goc = svm_goc.intercept_[0]
print("0. Mo hinh goc (24 diem):")
print(f"   w1 = {w_goc[0]:.4f}, w2 = {w_goc[1]:.4f}, b = {b_goc:.4f}")
print(f"   Chi so cac vecto ho tro: {[int(i) for i in svm_goc.support_]}")
print()

# Lần 1: Thêm điểm lớp 1 tại (-1, -1)
p1 = np.array([[-1.0, -1.0]])
X1 = np.vstack([X, p1])
y1 = np.append(y, 1)

svm1 = SVC(kernel="linear", C=1000).fit(X1, y1)
w1 = svm1.coef_[0]
b1 = svm1.intercept_[0]
print("1. Lan 1: Them diem lop 1 tai (-1, -1) vao 24 diem:")
print(f"   w1 = {w1[0]:.4f}, w2 = {w1[1]:.4f}, b = {b1:.4f}")
print(f"   Chi so cac vecto ho tro: {[int(i) for i in svm1.support_]}")
print(f"   f((-1, -1)) = {svm1.decision_function(p1)[0]:+.4f} (nam rat xa ngoai le)")
print()

# Lần 2: Bỏ (-1, -1), thay bằng điểm lớp 1 tại (4.5, 4)
p2 = np.array([[4.5, 4.0]])
X2 = np.vstack([X, p2])
y2 = np.append(y, 1)

svm2 = SVC(kernel="linear", C=1000).fit(X2, y2)
w2 = svm2.coef_[0]
b2 = svm2.intercept_[0]
print("2. Lan 2: Thay bang diem lop 1 tai (4.5, 4):")
print(f"   w1 = {w2[0]:.4f}, w2 = {w2[1]:.4f}, b = {b2:.4f}")
print(f"   Chi so cac vecto ho tro: {[int(i) for i in svm2.support_]}")
print(f"   f((4.5, 4)) = {svm2.decision_function(p2)[0]:+.4f} (chinh la vecto ho tro moi tai chi so 24)")
print()

print("Tra loi cac cau hoi:")
print("1. Lan nao duong ranh gioi thay doi?")
print("   - Lan 1 (them diem (-1, -1)): Duong ranh gioi HOAN TOAN KHONG THAY DOI.")
print("     (w1, w2, b giu nguyen tuyet doi: w1=-0.8314, w2=-0.3549, b=4.8204).")
print("   - Lan 2 (them diem (4.5, 4)): Duong ranh gioi THAY DOI RO RET.")
print("     (w1=-2.6030, w2=-0.8068, b=15.9409).")
print()
print("2. Giai thich bang khai niem vecto ho tro:")
print("   - Duong ranh gioi va hai mep le cua SVM chi duoc quyet dinh DUY NHAT boi cac vecto ho tro.")
print("     Cac diem khong phai vecto ho tro hoan toan khong co tieng noi trong viec xac dinh ranh gioi.")
print("   - O Lan 1: Diem (-1, -1) nam rat sau ve phia cum lop 1, f((-1, -1)) = +6.006 > 1 (ngoai le).")
print("     Do do diem nay KHONG phai la vecto ho tro, khong lam anh huong den ranh gioi da co.")
print("   - O Lan 2: Diem (4.5, 4.0) nam sau vao vung lanh tho cua cum lop 0 (truoc do f(4.5, 4) = -0.499 < 0).")
print("     Khi duoc gan nhan 1, diem nay buoc SVM phai xoay va dich chuyen ranh gioi de thoa man;")
print("     no tro thanh mot trong cac vecto ho tro moi (chi so 24), lam thay doi hoan toan duong ranh gioi.")
