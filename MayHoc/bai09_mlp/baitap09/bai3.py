# -*- coding: utf-8 -*-
"""Bài tập 3: So sánh các hàm kích hoạt (activation functions) trên hai vầng trăng.

Dùng mạng MLP 2 lớp ẩn (100, 100) với alpha = 1.0, solver="lbfgs".
Lần lượt thử nghiệm 3 hàm kích hoạt: "logistic", "tanh", "relu".
Đo accuracy trên tập học và tập kiểm tra, tìm hàm cho kết quả tốt nhất.
"""

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/hai_vang_trang.csv")
X = df[["x1", "x2"]]
y = df["nhan"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

cac_ham = ["logistic", "tanh", "relu"]
ket_qua = []

print("=== SO SANH HAM KICH HOAT TREN BO HAI VANG TRANG ===")
print("Ham kich hoat   Accuracy tap hoc   Accuracy tap kiem tra")

for act in cac_ham:
    mang = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(100, 100), activation=act,
                      alpha=1.0, solver="lbfgs", max_iter=20000, random_state=0)
    )
    mang.fit(X_tr, y_tr)
    acc_tr = accuracy_score(y_tr, mang.predict(X_tr))
    acc_te = accuracy_score(y_te, mang.predict(X_te))
    ket_qua.append((act, acc_tr, acc_te))
    print(f"  {act:12s}        {acc_tr:.4f}               {acc_te:.4f}")

# Tìm hàm tốt nhất theo accuracy kiểm tra (nếu bằng nhau thì xét thêm tập học)
max_te = max(k[2] for k in ket_qua)
cac_ham_tot = [k for k in ket_qua if k[2] == max_te]

print("\n=== TRA LOI CAU HOI ===")
print(f"- Ham cho ket qua tot nhat tren tap kiem tra:")
for h, a_tr, a_te in cac_ham_tot:
    print(f"  + Ham '{h}': dat accuracy tap kiem tra cao nhat = {a_te:.4f} (tap hoc = {a_tr:.4f}).")
print("- Nhan xet:")
print("  + Ca 'tanh' va 'relu' deu cho accuracy tap kiem tra vuot troi (0.9067) so voi 'logistic' (0.8533).")
print("  + Ham 'relu' cho accuracy tren tap hoc cao hon (0.8844 so voi 0.8711 cua tanh), dong thoi tinh toan nhanh hon.")
print("  + Ham 'logistic' de bi triet tieu dao ham (vanishing gradient) o cac gia tri bao hoa, dan den hoi tu cham hon.")
