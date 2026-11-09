# -*- coding: utf-8 -*-
"""Bài tập 6: SVM trên dữ liệu lớn (Đo thời gian fit của SVC và LinearSVC)."""

import sys
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import time
from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC, SVC

co_mau = [2000, 5000, 10000, 20000]
ket_qua = []

print("=== BAI TAP 6: SVM TREN DU LIEU LON ===")
print("Dang tao du lieu va do thoi gian .fit() cho cac co mau...")
print()

for n in co_mau:
    X, y = make_classification(n_samples=n, n_features=20, random_state=0)

    # 1. Đo thời gian SVC()
    t0 = time.perf_counter()
    SVC().fit(X, y)
    t_svc = time.perf_counter() - t0

    # 2. Đo thời gian LinearSVC()
    t0 = time.perf_counter()
    LinearSVC(random_state=0, max_iter=2000).fit(X, y)
    t_lsvc = time.perf_counter() - t0

    ket_qua.append((n, t_svc, t_lsvc))

# In bảng kết quả
print("Bang thoi gian huan luyen .fit():")
print(f"{'Co mau n':>10} | {'Thoi gian SVC (s)':>20} | {'Thoi gian LinearSVC (s)':>24}")
print("-" * 60)
for n, t_svc, t_lsvc in ket_qua:
    print(f"{n:>10d} | {t_svc:>20.4f} | {t_lsvc:>24.4f}")

print()

# Tính tỷ số tăng khi dữ liệu tăng gấp đôi
print("Phan tich toc do tang thoi gian khi du lieu tang:")
print("1. Tu n=5.000 len n=10.000 (tang 2 lan):")
ty_le_svc_1 = ket_qua[2][1] / max(1e-6, ket_qua[1][1])
ty_le_lsvc_1 = ket_qua[2][2] / max(1e-6, ket_qua[1][2])
print(f"   - SVC       tang : {ty_le_svc_1:.2f} lan")
print(f"   - LinearSVC tang : {ty_le_lsvc_1:.2f} lan")
print()

print("2. Tu n=10.000 len n=20.000 (tang 2 lan):")
ty_le_svc_2 = ket_qua[3][1] / max(1e-6, ket_qua[2][1])
ty_le_lsvc_2 = ket_qua[3][2] / max(1e-6, ket_qua[2][2])
print(f"   - SVC       tang : {ty_le_svc_2:.2f} lan")
print(f"   - LinearSVC tang : {ty_le_lsvc_2:.2f} lan")
print()

print("Nhan xet va giai thich:")
print("1. Khi du lieu tang gap doi:")
print("   - Thoi gian cua SVC() tang khoang 3 den 8 lan (trung binh khoang 4 lan hoac hon).")
print("   - Thoi gian cua LinearSVC() chi tang khoang 1.5 den 2.5 lan (xap xi 2 lan).")
print()
print("2. Giai thich nguyen nhan:")
print("   - SVC() dung thuat toan giai quyet bai toan quy hoach phi tuyen dua tren ma tran kernel.")
print("     Do phuc tap tinh toan cua no la O(n^2) den O(n^3). Khi n tang gap 2, chi phi tinh toan")
print("     tang theo cap so mu 2^2 = 4 den 2^3 = 8 lan, khien SVC rat cham tren du lieu lon.")
print("   - LinearSVC() su dung thuat toan Liblinear toi uu truc tiep tren khong gian nguyen thuy,")
print("     khong can ma tran kernel, co do phuc tap xap xi tuyen tinh O(n). Khi n tang gap 2,")
print("     thoi gian hoc chi tang xap xi gap doi.")
print("   -> Loi khuyen: Voi tap du lieu lon co hang chuc nghin den hang trieu mau, neu dung mo hinh")
print("      tuyen tinh thi nen uu tien LinearSVC() thay vi SVC(kernel='linear').")
