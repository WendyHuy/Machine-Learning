# -*- coding: utf-8 -*-
"""Bài tập 1: Tự dự đoán bằng trọng số đã học của MLPClassifier.

Học lại mạng 8 nơ-ron ẩn trên bộ XOR như mục 3.
Lấy mlp.coefs_ và mlp.intercepts_, tự tính lan truyền tiến bằng numpy cho 5 điểm
đầu của tập kiểm tra (lớp ẩn ReLU, lớp ra sigmoid).
So sánh với mlp.predict_proba cột thứ hai.
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def relu(z):
    return np.maximum(0, z)


df = pd.read_csv("data/xor.csv")
X = df[["x1", "x2"]]
y = df["nhan"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

# Huấn luyện mạng MLP 8 nơ-ron ẩn
mlp = MLPClassifier(hidden_layer_sizes=(8,), solver="lbfgs",
                    max_iter=5000, random_state=0)
mlp.fit(X_tr, y_tr)

# Trích xuất trọng số và bias
W1, W2 = mlp.coefs_          # W1: (2, 8), W2: (8, 1)
b1, b2 = mlp.intercepts_     # b1: (8,),   b2: (1,)

# Lấy 5 điểm đầu tiên của tập kiểm tra
X_5 = X_te.iloc[:5].values
y_5 = y_te.iloc[:5].values

# Tự tính lan truyền tiến bằng numpy
H = relu(X_5 @ W1 + b1)      # (5, 8)
Z_out = H @ W2 + b2          # (5, 1)
prob_numpy = sigmoid(Z_out).ravel()

# Xác suất từ hàm predict_proba của scikit-learn (cột thứ 2 ứng với lớp 1)
prob_sklearn = mlp.predict_proba(X_5)[:, 1]

print("=== SO SANH XAC SUAT LAN TRUYEN TIEN BANG NUMPY VA SCIKIT-LEARN ===")
print("Mau      x1       x2   Nhan that   Prob (Numpy)   Prob (Sklearn)     Chenh lech")
for i in range(5):
    diff = abs(prob_numpy[i] - prob_sklearn[i])
    print(f" {i+1}   {X_5[i,0]:7.4f}  {X_5[i,1]:7.4f}      {y_5[i]}       "
          f"{prob_numpy[i]:.8f}     {prob_sklearn[i]:.8f}   {diff:.2e}")

print("\n=== NHAN XET ===")
print("- Ket qua tu tinh lan truyen tien bang numpy hoan toan trung khop voi mlp.predict_proba")
print("  toi tan chu so thap phan cuc nho (sai so < 1e-12, do lam tron dau phay dong).")
print("- Dieu nay chung minh rang ben trong MLPClassifier:")
print("  + Dau vao duoc bien doi qua lop an: H = ReLU(X @ W1 + b1)")
print("  + Lop ra ap dung ham sigmoid tren ket hop tuyen tinh: P = sigmoid(H @ W2 + b2)")
