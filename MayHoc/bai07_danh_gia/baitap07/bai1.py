# -*- coding: utf-8 -*-
"""Bài tập 1: Tự viết hàm đánh giá mô hình bằng numpy.

Hàm danh_gia(y_that, y_doan) trả về accuracy, precision, recall, F1.
Xử lý trường hợp chia cho 0 khi tính precision của mô hình lười.
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def danh_gia(y_that, y_doan):
    """Tính accuracy, precision, recall, f1 chỉ bằng numpy."""
    y_that = np.asarray(y_that)
    y_doan = np.asarray(y_doan)

    tp = np.sum((y_that == 1) & (y_doan == 1))
    fp = np.sum((y_that == 0) & (y_doan == 1))
    fn = np.sum((y_that == 1) & (y_doan == 0))
    tn = np.sum((y_that == 0) & (y_doan == 0))

    acc = (tp + tn) / len(y_that)

    # Xử lý chia cho 0: nếu TP + FP == 0 thì precision gán bằng 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Xử lý chia cho 0: nếu TP + FN == 0 thì recall gán bằng 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Xử lý chia cho 0: nếu prec + rec == 0 thì F1 gán bằng 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return acc, prec, rec, f1


df = pd.read_csv("data/sinh_vien_bo_hoc.csv")
X = df.drop(columns="bo_hoc")
y = df["bo_hoc"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

# 1. Mô hình lười
luoi = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
du_luoi = luoi.predict(X_te)

# 2. Hồi quy logistic
lr = make_pipeline(StandardScaler(), LogisticRegression())
lr.fit(X_tr, y_tr)
du_lr = lr.predict(X_te)

print("=== SO SANH TU TINH VOI SCIKIT-LEARN ===")
print("1. Mo hinh luoi (DummyClassifier):")
acc_l, prec_l, rec_l, f1_l = danh_gia(y_te, du_luoi)
print(f"  Tu tinh        : Acc={acc_l:.4f}, Prec={prec_l:.4f}, Rec={rec_l:.4f}, F1={f1_l:.4f}")
print(f"  scikit-learn   : Acc={accuracy_score(y_te, du_luoi):.4f}, "
      f"Prec={precision_score(y_te, du_luoi, zero_division=0):.4f}, "
      f"Rec={recall_score(y_te, du_luoi):.4f}, "
      f"F1={f1_score(y_te, du_luoi, zero_division=0):.4f}")

print("\n2. Hoi quy logistic:")
acc_lr, prec_lr, rec_lr, f1_lr = danh_gia(y_te, du_lr)
print(f"  Tu tinh        : Acc={acc_lr:.4f}, Prec={prec_lr:.4f}, Rec={rec_lr:.4f}, F1={f1_lr:.4f}")
print(f"  scikit-learn   : Acc={accuracy_score(y_te, du_lr):.4f}, "
      f"Prec={precision_score(y_te, du_lr):.4f}, "
      f"Rec={recall_score(y_te, du_lr):.4f}, "
      f"F1={f1_score(y_te, du_lr):.4f}")

print("\n=== TRA LOI CAU HOI ===")
print("- Voi mo hinh luoi, precision khong tinh duoc theo phep chia thong thuong")
print("  vi mo hinh khong doan mau nao la lop 1 (TP = 0, FP = 0 -> mau so TP + FP = 0).")
print("- Ham xu ly truong hop chia cho 0 bang cach kiem tra dieu kien mau so > 0,")
print("  neu mau so bang 0 thi tra ve 0.0 (tuong duong tham so zero_division=0 cua scikit-learn).")
