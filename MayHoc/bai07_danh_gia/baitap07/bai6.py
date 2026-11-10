# -*- coding: utf-8 -*-
"""Bài tập 6: MAE và RMSE với một lần đoán hỏng.

Trên dữ liệu giá nhà của bài 1, tính MAE, RMSE và R^2 của hồi quy tuyến tính trên tập kiểm tra.
Sau đó cộng thêm 5 tỷ vào dự đoán của đúng một căn, rồi tính lại ba con số.
Đánh giá con số nào thay đổi nhiều nhất và giải thích vì sao RMSE phạt nặng sai số lớn hơn MAE.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/gia_nha.csv")
X = df[["dien_tich"]]
y = df["gia"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 1. Hồi quy tuyến tính thông thường
lr = LinearRegression()
lr.fit(X_tr, y_tr)
y_pred = lr.predict(X_te)

mae_goc = mean_absolute_error(y_te, y_pred)
rmse_goc = np.sqrt(mean_squared_error(y_te, y_pred))
r2_goc = r2_score(y_te, y_pred)

print("=== 1. KET QUA KHI CHUA CO DIEM DOAN SAI LECH ===")
print(f"MAE  = {mae_goc:.4f} ty dong")
print(f"RMSE = {rmse_goc:.4f} ty dong")
print(f"R^2  = {r2_goc:.4f}")

# 2. Thêm sai số 5 tỷ vào đúng một căn trong tập kiểm tra
y_pred_hong = y_pred.copy()
y_pred_hong[0] += 5.0  # cong them 5 ty vao can dau tien

mae_hong = mean_absolute_error(y_te, y_pred_hong)
rmse_hong = np.sqrt(mean_squared_error(y_te, y_pred_hong))
r2_hong = r2_score(y_te, y_pred_hong)

print("\n=== 2. KET QUA SAU KHI CONG THEM 5 TY VAO DUNG MOT CAN ===")
print(f"MAE  = {mae_hong:.4f} ty dong (tang {mae_hong - mae_goc:+.4f})")
print(f"RMSE = {rmse_hong:.4f} ty dong (tang {rmse_hong - rmse_goc:+.4f})")
print(f"R^2  = {r2_hong:.4f} (thay doi {r2_hong - r2_goc:+.4f})")

print("\n=== 3. SO SANH PHAN TRAM THAY DOI ===")
pct_mae = (mae_hong - mae_goc) / mae_goc * 100
pct_rmse = (rmse_hong - rmse_goc) / rmse_goc * 100
pct_r2 = abs(r2_hong - r2_goc) / abs(r2_goc) * 100
print(f"Phan tram thay doi MAE : {pct_mae:+.2f}%")
print(f"Phan tram thay doi RMSE: {pct_rmse:+.2f}%")
print(f"Phan tram thay doi R^2 : {pct_r2:+.2f}%")

print("\n=== TRA LOI CAU HOI ===")
print("- Con so thay doi manh nhat ve do lon sai so la RMSE va R^2:")
print(f"  RMSE tang tu {rmse_goc:.4f} len {rmse_hong:.4f} (tang {pct_rmse:.1f}%),")
print(f"  trong khi MAE chi tang tu {mae_goc:.4f} len {mae_hong:.4f} (tang {pct_mae:.1f}%).")
print(f"  Diem R^2 bi tut giam nghiem trong tu {r2_goc:.4f} xuong {r2_hong:.4f}.")
print("- Giai thich vi sao RMSE phat nang sai so lon hon MAE:")
print("  Trong cong thuc cua MAE, sai so e_i duoc tinh theo gia tri tuyet doi |e_i|, do do mot sai so")
print("  5 ty chi dong gop dung 5/N vao MAE trung binh.")
print("  Trong cong thuc cua RMSE, cac sai so bi binh phuong (e_i^2) truoc khi lay trung binh va can bac hai.")
print("  Vi 5^2 = 25 (gap rat nhieu lan cac sai so nho binh phuong), sai so lon bi phong dai len rat nhieu.")
print("  Vi vay, RMSE cuc ky nhay cam voi cac diem ngoai le (outliers) va phat rat nang nhung lan doan hong.")
