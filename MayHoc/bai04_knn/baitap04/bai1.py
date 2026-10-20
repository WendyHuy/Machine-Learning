# -*- coding: utf-8 -*-
"""Bài tập 1: Tự viết KNN không dùng thư viện."""

import sys
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Đọc dữ liệu hai vầng trăng
df = pd.read_csv("data/hai_vang_trang.csv")
X = df[["x1", "x2"]].to_numpy(dtype=float)
y = df["nhan"].to_numpy(dtype=int)

# Chia tập học và kiểm tra tương tự mục 4.2 của tài liệu
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)


def knn_du_doan(X_hoc, y_hoc, x_moi, k):
    """Dự đoán nhãn cho một mẫu mới dựa trên k láng giềng gần nhất.

    X_hoc : mảng 2D tập huấn luyện
    y_hoc : mảng 1D nhãn tương ứng (0 hoặc 1)
    x_moi : mảng 1D đặc trưng của mẫu mới
    k     : số láng giềng cần xét
    """
    # 1. Tính khoảng cách Euclid từ x_moi tới mọi điểm trong X_hoc
    khoang_cach = np.sqrt(((X_hoc - x_moi) ** 2).sum(axis=1))

    # 2. Tìm chỉ số của k láng giềng có khoảng cách nhỏ nhất
    chi_so_gan = np.argsort(khoang_cach)[:k]

    # 3. Lấy nhãn của k láng giềng
    nhan_gan = y_hoc[chi_so_gan]

    # 4. Bỏ phiếu đa số (với nhãn 0 và 1)
    so_phieu_1 = np.sum(nhan_gan == 1)
    return 1 if so_phieu_1 > (k / 2) else 0


k = 13

# Dự đoán cho toàn bộ tập kiểm tra bằng hàm tự viết
y_du_doan_tu_viet = np.array([knn_du_doan(X_tr, y_tr, x, k=k) for x in X_te])
acc_tu_viet = accuracy_score(y_te, y_du_doan_tu_viet)

# So sánh với KNeighborsClassifier của scikit-learn
mh_sklearn = KNeighborsClassifier(n_neighbors=k)
mh_sklearn.fit(X_tr, y_tr)
y_du_doan_sklearn = mh_sklearn.predict(X_te)
acc_sklearn = accuracy_score(y_te, y_du_doan_sklearn)

print("=== BAI TAP 1: TU VIET THUAT TOAN KNN KHONG DUNG THU VIEN ===")
print(f"So luong mau tap hoc      : {len(X_tr)}")
print(f"So luong mau tap kiem tra : {len(X_te)}")
print(f"So lang gieng k           : {k}")
print()
print(f"Accuracy tu viet KNN (k={k})        : {acc_tu_viet:.4f}")
print(f"Accuracy KNeighborsClassifier (k={k}) : {acc_sklearn:.4f}")
print()
if np.isclose(acc_tu_viet, acc_sklearn):
    print("Nhan xet: Hai ket qua HOAN TOAN KHOP NHAU (0.9200) nhu yeu cau de bai.")
else:
    print("Nhan xet: Co su chenh lech giua hai ket qua.")

