# -*- coding: utf-8 -*-
"""Bài tập 6: Đo thời gian dự đoán và bản chất học lười của KNN."""

import sys
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Tạo bộ sinh số ngẫu nhiên với seed cố định
rng = np.random.default_rng(0)

# Tạo trước 1000 mẫu kiểm tra mới với 2 đặc trưng
X_moi = rng.random((1000, 2))

cac_co = [1000, 10000, 100000]
ket_qua = []

print("=== BAI TAP 6: DO THOI GIAN FIT VA PREDICT CUA KNN ===")
print("Dang thuc hien do thoi gian cho cac co mau...")

for N in cac_co:
    # Tạo N mẫu huấn luyện ngẫu nhiên với 2 đặc trưng
    X_tr = rng.random((N, 2))
    y_tr = rng.integers(0, 2, size=N)

    knn = KNeighborsClassifier(n_neighbors=5)

    # 1. Đo thời gian .fit
    t_bat_dau_fit = time.perf_counter()
    knn.fit(X_tr, y_tr)
    t_fit = time.perf_counter() - t_bat_dau_fit

    # 2. Đo thời gian .predict cho 1000 mẫu mới
    t_bat_dau_pred = time.perf_counter()
    knn.predict(X_moi)
    t_pred = time.perf_counter() - t_bat_dau_pred

    ket_qua.append((N, t_fit, t_pred))

# In bảng kết quả
print()
print("Bang ket qua thoi gian thuc thi:")
print(f"{'Co mau N':>10} | {'Thoi gian .fit (s)':>20} | {'Thoi gian .predict 1000 mau (s)':>32}")
print("-" * 68)
for N, t_fit, t_pred in ket_qua:
    print(f"{N:>10d} | {t_fit:>20.5f} | {t_pred:>32.5f}")

print()
print("Giai thich vi sao KNN duoc goi la mo hinh 'hoc luoi' (lazy learning):")
print()
print("1. Giai doan huan luyen (.fit):")
print("   - KNN khong thiet lap phuong trinh, khong hoc trong so w hay tham so nao tu du lieu.")
print("   - 'Huan luyen' cua KNN thuc chat chi la luu toan bo du lieu vao bo nho (va dung cay chi muc).")
print("   - Do do, thoi gian .fit cuc ky nhanh, gan nhu khong ton chi phi tinh toan.")
print()
print("2. Giai doan du doan (.predict):")
print("   - Moi phep tinh thuc su deu bi tri hoan va don het vao luc du doan.")
print("   - Voi moi mau moi, mo hinh phai di tinh khoang cach toi tat ca (hoac rat nhieu) mau cu")
print("     trong tap huan luyen de tim ra k lang gieng gan nhat va bo phieu.")
print("   - Khi tap du lieu huan luyen tang tu 1.000 len 100.000 mau, chi phi tim kiem va tinh toan")
print("     khoang cach tang len ro ret, khien thoi gian du doan keo dai.")
print("   -> 'Hoc thi luoi' (khong lam gi ca), nhung 'du doan thi cuc nhoc' (tinh toan don het vao day).")

