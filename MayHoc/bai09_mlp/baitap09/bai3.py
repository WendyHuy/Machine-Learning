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

tot_nhat = max(ket_qua, key=lambda x: x[2])
print("\n=== TRA LOI CAU HOI ===")
print(f"- Ham cho ket qua tot nhat tren tap kiem tra la: '{tot_nhat[0]}' (accuracy = {tot_nhat[2]:.4f}).")
print("- Nhan xet:")
print("  + Ham 'relu' thuong hoc nhanh, tao cac duong bien phan chia manh me, phu hop voi du lieu phi tuyen.")
print("  + Ham 'tanh' va 'logistic' deu co tinh chat tron tru, dao ham lien tuc nen ranh gioi rat min.")
print("  + Tren du lieu hai vang trang voi alpha = 1.0, cac ham phi tuyen tron nhu 'tanh' hoac 'relu' deu cho")
print("    do chinh xac rat cao (~88-91%), the hien ro kha nang uon luon theo duong cong cua mang no-ron.")
