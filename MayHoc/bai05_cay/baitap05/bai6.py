# -*- coding: utf-8 -*-
"""Bài tập 6: Rừng ngẫu nhiên cho bài toán hồi quy (RandomForestRegressor)."""

import sys
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Đọc dữ liệu giá nhà
df = pd.read_csv("data/gia_nha.csv")
X = df[["dien_tich"]]
y = df["gia"]

# Chia tập học và kiểm tra như mục 8 (test_size=0.2, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 1. Cây hồi quy sâu 3 (mục 8)
cay_3 = DecisionTreeRegressor(max_depth=3, random_state=0).fit(X_tr, y_tr)
mae_cay_3 = mean_absolute_error(y_te, cay_3.predict(X_te))

# 2. Hồi quy tuyến tính - đường thẳng (mục 8)
duong_thang = LinearRegression().fit(X_tr, y_tr)
mae_duong = mean_absolute_error(y_te, duong_thang.predict(X_te))

# 3. Rừng ngẫu nhiên 200 cây
rung = RandomForestRegressor(n_estimators=200, random_state=0).fit(X_tr, y_tr)
mae_rung = mean_absolute_error(y_te, rung.predict(X_te))

print("=== BAI TAP 6: RUNG NGAU NHIEN CHO HOI QUY GIA NHA ===")
print("1. So sanh MAE tren tap kiem tra (ty dong):")
print(f"   - Cay quyet dinh sau 3       : {mae_cay_3:.4f} ty")
print(f"   - Rung ngau nhien 200 cay    : {mae_rung:.4f} ty")
print(f"   - Duong thang (Linear Reg)   : {mae_duong:.4f} ty")
print()

# Dự đoán cho căn mới 250 m2
can_250 = pd.DataFrame({"dien_tich": [250.0]})
gia_cay = cay_3.predict(can_250)[0]
gia_rung = rung.predict(can_250)[0]
gia_duong = duong_thang.predict(can_250)[0]

print("2. Du doan gia cho can nha rong 250 m2 (ngoai vung du lieu):")
print(f"   - Cay sau 3       : {gia_cay:.2f} ty dong")
print(f"   - Rung 200 cay    : {gia_rung:.2f} ty dong")
print(f"   - Duong thang     : {gia_duong:.2f} ty dong")
print()
print("Tra loi cac cau hoi:")
print("1. Rung co ngoai suy duoc khong?")
print("   -> KHONG. Rung ngau nhien HOAN TOAN KHONG THE NGOAI SUY DUOC ra ngoai mien du lieu.")
print(f"      Dien tich lon nhat tap hoc la {X_tr['dien_tich'].max():.1f} m2 (gia {y_tr.max():.2f} ty).")
print(f"      Voi can 250 m2, duong thang doan hop ly {gia_duong:.2f} ty, con rung chi doan duoc {gia_rung:.2f} ty.")
print()
print("2. Giai thich bang cach la cua moi cay dua ra du doan:")
print("   - Moi cay quyet dinh trong rung phan chia khong gian bang cac cau hoi dang 'dien_tich <= nguong'.")
print("   - Khi gap can 250 m2 (lon hon moi nguong trong cay), can nay deu bi re sang la tan cung ben phai.")
print("   - O moi la, du doan la GIA TRI TRUNG BINH cua cac can nha tap hoc roi vao la do (khong the vuot")
print("     qua gia tri lon nhat trong tap du lieu goc).")
print("   - Du doan cua rung ngau nhien bang TRUNG BINH CONG du doan cua 200 cay:")
print("     Trung binh cua cac con so bi chan tren (<= 9.74 ty) thi van luon bi chan tren.")
print("   -> Vi vay, du co trong bao nhieu cay di nua thi rung van chi noi lai nhung gi da thay trong")
print("      vung du lieu huan luyen, khong the tiep tuc ngoai suy duong thang di len.")
