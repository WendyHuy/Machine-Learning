# -*- coding: utf-8 -*-
"""Bài tập 6: Mạng nơ-ron cho bài toán hồi quy (MLPRegressor).

Dùng MLPRegressor(hidden_layer_sizes=(20,), max_iter=5000, random_state=0) có chuẩn hoá.
So sánh MAE trên tập kiểm tra với Hồi quy tuyến tính.
Dự đoán giá căn nhà diện tích 250 m2 và nhận xét về khả năng ngoại suy của MLP.
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/gia_nha.csv")
X = df[["dien_tich"]]
y = df["gia"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 1. Hồi quy tuyến tính
lr = LinearRegression().fit(X_tr, y_tr)
du_lr = lr.predict(X_te)
mae_lr = mean_absolute_error(y_te, du_lr)

# 2. Mạng nơ-ron hồi quy có chuẩn hoá
mlp = make_pipeline(
    StandardScaler(),
    MLPRegressor(hidden_layer_sizes=(20,), max_iter=5000, random_state=0)
).fit(X_tr, y_tr)
du_mlp = mlp.predict(X_te)
mae_mlp = mean_absolute_error(y_te, du_mlp)

print("=== SO SANH MAE TREN TAP KIEM TRA ===")
print(f"MAE Hoi quy tuyen tinh : {mae_lr:.4f} ty dong")
print(f"MAE MLPRegressor       : {mae_mlp:.4f} ty dong")

# 3. Dự đoán giá căn nhà 250 m2 (ngoại suy)
can_250 = pd.DataFrame({"dien_tich": [250.0]})
gia_lr_250 = lr.predict(can_250)[0]
gia_mlp_250 = mlp.predict(can_250)[0]

print("\n=== DU DOAN GIA CAN NHA 250 m2 (NGOAI SUY) ===")
print(f"Gia nha du doan boi Hoi quy tuyen tinh : {gia_lr_250:.2f} ty dong")
print(f"Gia nha du doan boi MLPRegressor       : {gia_mlp_250:.2f} ty dong")

print("\n=== TRA LOI CAU HOI ===")
print("- So sanh MAE tren tap kiem tra:")
if mae_mlp < mae_lr:
    print(f"  MLPRegressor dat MAE tot hon hoi quy tuyen tinh ({mae_mlp:.4f} vs {mae_lr:.4f}).")
else:
    print(f"  Hoi quy tuyen tinh dat MAE tot hon hoac tuong duong MLPRegressor ({mae_lr:.4f} vs {mae_mlp:.4f}).")

print("- Mang no-ron co ngoai suy tot nhu duong thang khong?")
print("  KHONG! Mang no-ron khong ngoai suy tot bang hoi quy tuyen tinh.")
print("  Giai thich:")
print("  + Duong thang co tinh chat tuyen tinh toan cuc, voi can nha 250 m2 (gap doi pham vi du lieu train),")
print("    no tiep tuc tang theo he so goc w va cho ra muc gia hop ly theo xu huong thi truong.")
print("  + Mang no-ron voi cac ham kich hoat phi tuyen (nhu ReLU hay Sigmoid) chi hoc duoc hinh dang ben trong")
print("    mien du lieu huan luyen (noi suy). Khi gap diem nam xa ngoai vung hoc (ngoai suy 250 m2), cac no-ron")
print("    bi bao hoa hoac di theo doan thang co do doc cuoi cung, de dan den gia tri du doan bi 'bet' ngang")
print("    hoac lech lac vo ly so voi xu huong thuc te.")
