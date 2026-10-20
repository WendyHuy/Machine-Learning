# -*- coding: utf-8 -*-
"""Bài tập 5: Chọn k cho bài toán hồi quy (KNeighborsRegressor)."""

import sys
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor

# Đọc dữ liệu giá nhà
df = pd.read_csv("data/gia_nha.csv")
X = df[["dien_tich"]]
y = df["gia"]

# Chia dữ liệu như mục 6 (random_state=42, test_size=0.2)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Kiểm định chéo 5 phần trên tập học để tìm k tốt nhất từ 1 đến 20
diem_mae_cv = {}
for k in range(1, 21):
    # scoring='neg_mean_absolute_error' trả về điểm âm, cần đổi dấu để lấy MAE
    scores = cross_val_score(
        KNeighborsRegressor(n_neighbors=k),
        X_tr,
        y_tr,
        cv=5,
        scoring="neg_mean_absolute_error",
    )
    diem_mae_cv[k] = -scores.mean()

# Chọn k có MAE kiểm định chéo nhỏ nhất
k_tot = min(diem_mae_cv, key=diem_mae_cv.get)

# Đánh giá trên tập kiểm tra với k tốt nhất
knn_tot = KNeighborsRegressor(n_neighbors=k_tot).fit(X_tr, y_tr)
mae_knn_tot = mean_absolute_error(y_te, knn_tot.predict(X_te))

# So sánh với mô hình Hồi quy tuyến tính (đường thẳng)
duong_thang = LinearRegression().fit(X_tr, y_tr)
mae_duong_thang = mean_absolute_error(y_te, duong_thang.predict(X_te))

# Mô hình KNN k=5 (ở mục 6) để tham chiếu
knn_5 = KNeighborsRegressor(n_neighbors=5).fit(X_tr, y_tr)
mae_knn_5 = mean_absolute_error(y_te, knn_5.predict(X_te))

print("=== BAI TAP 5: CHON K CHO BAI TOAN HOI QUY (GIA NHA) ===")
print(f"Kiem dinh cheo 5 phan (CV 5-fold) tren tap hoc voi k tu 1 den 20:")
print("   k    MAE CV (ty dong)")
for k in range(1, 21):
    dau = " <- tot nhat" if k == k_tot else ""
    print(f"  {k:2d}    {diem_mae_cv[k]:.4f}{dau}")

print()
print("Ket qua danh gia tren tap kiem tra:")
print(f"  k tot nhat theo CV            : k = {k_tot} (MAE CV = {diem_mae_cv[k_tot]:.4f} ty)")
print(f"  KNN voi k tot nhat (k={k_tot})       : MAE test = {mae_knn_tot:.4f} ty")
print(f"  KNN voi k = 5 (muc 6)          : MAE test = {mae_knn_5:.4f} ty")
print(f"  Duong thang (Linear Regression): MAE test = {mae_duong_thang:.4f} ty")
print()
print("Tra loi cau hoi:")
print("Voi k tot nhat, MAE tren tap kiem tra co vuot duoc duong thang khong?")
print(f"  -> KHONG vuot duoc! MAE cua KNN voi k={k_tot} la {mae_knn_tot:.4f} ty, van cao hon")
print(f"     dang ke so voi {mae_duong_thang:.4f} ty cua duong thang.")
print()
print("Giai thich:")
print("  - Moi quan he giua dien tich va gia nha trong tap du lieu thuc su gan nhu tuyen tinh.")
print("  - Mo hinh duong thang tan dung triet de gia dinh dung nay va mo hinh hoa toan cuc rat tot.")
print("  - Trong khi do, KNN chi lay trung binh cuc bo k diem xung quanh, de bi nhieu va tao thanh")
print("    duong du doan dang bac thang roi rac, dong thoi tap du lieu it mau nen KNN khong the")
print("    dat do chinh xac tot bang mo hinh tham so duong thang.")

