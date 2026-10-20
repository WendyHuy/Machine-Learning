# -*- coding: utf-8 -*-
"""Bài tập 3: Đổi đơn vị đo và vai trò của chuẩn hoá."""

import sys
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Đọc dữ liệu sinh viên mở rộng
df = pd.read_csv("data/sinh_vien_mo_rong.csv")
y = df["qua_mon"]

# Cột chi tiêu đơn vị nghìn đồng và đơn vị triệu đồng
df["chi_tieu_trieu"] = df["chi_tieu"] / 1000.0

cot_nghin = ["gio_on", "diem_giua_ky", "chi_tieu"]
cot_trieu = ["gio_on", "diem_giua_ky", "chi_tieu_trieu"]

# Chia tập học và kiểm tra (cùng random_state=17, stratify=y)
X_tr, X_te, y_tr, y_te = train_test_split(
    df[cot_nghin + ["chi_tieu_trieu"]], y, test_size=0.25, random_state=17, stratify=y
)

# 1. Không chuẩn hoá - đơn vị nghìn đồng (như mục 3)
knn_nghin = KNeighborsClassifier(n_neighbors=7)
knn_nghin.fit(X_tr[cot_nghin], y_tr)
acc_nghin = accuracy_score(y_te, knn_nghin.predict(X_te[cot_nghin]))

# 2. Không chuẩn hoá - đơn vị triệu đồng
knn_trieu = KNeighborsClassifier(n_neighbors=7)
knn_trieu.fit(X_tr[cot_trieu], y_tr)
acc_trieu = accuracy_score(y_te, knn_trieu.predict(X_te[cot_trieu]))

# 3. Có chuẩn hoá - đơn vị nghìn đồng
pipe_nghin = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7))
pipe_nghin.fit(X_tr[cot_nghin], y_tr)
acc_chuan_nghin = accuracy_score(y_te, pipe_nghin.predict(X_te[cot_nghin]))

# 4. Có chuẩn hoá - đơn vị triệu đồng
pipe_trieu = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7))
pipe_trieu.fit(X_tr[cot_trieu], y_tr)
acc_chuan_trieu = accuracy_score(y_te, pipe_trieu.predict(X_te[cot_trieu]))

print("=== BAI TAP 3: DOI DON VI DO VA VAI TRO CUA CHUAN HOA ===")
print(f"1. Khong chuan hoa (chi tieu: nghin dong) : {acc_nghin:.4f}")
print(f"2. Khong chuan hoa (chi tieu: trieu dong): {acc_trieu:.4f}")
print(f"3. Co chuan hoa    (chi tieu: nghin dong) : {acc_chuan_nghin:.4f}")
print(f"4. Co chuan hoa    (chi tieu: trieu dong): {acc_chuan_trieu:.4f}")
print()
print("Tra loi cac cau hoi:")
print("1. Accuracy thay doi the nao so voi 0.5333 o muc 3?")
print(f"   -> Khi doi sang trieu dong, accuracy vot tu {acc_nghin:.4f} len {acc_trieu:.4f} (tang ro ret).")
print()
print("2. Vi sao chi doi don vi ma ket qua lai khac han?")
print("   -> Khi tinh bang 'nghin dong', chi tieu co gia tri tu 2000 den 8000, do lech chuan ~1692.")
print("      Con gio on chi 0-30 va diem chi 0-10. Chenh lech chi tieu ap dao hoan toan khoang cach,")
print("      khien mo hinh bo qua hai dac trung quan trong, lam accuracy te hon ca doan bua (0.6000).")
print("   -> Khi chia 1000 sang 'trieu dong', chi tieu chi con 2.0 den 8.0, do lech chuan ~1.69.")
print("      Thang do luc nay tuong dong voi gio on va diem giua ky, khoang cach Euclid khong con bi")
print("      thau tom boi chi tieu, giup accuracy tang vot len 0.8667.")
print()
print("3. Neu co chuan hoa thi doi don vi co lam ket qua thay doi khong?")
print(f"   -> KHONG thay doi! Ca hai truong hop co chuan hoa deu dat chinh xac {acc_chuan_nghin:.4f}.")
print("      Ly do: StandardScaler tinh z = (x - mean) / std. Khi nhan/chia mot bien voi hang so c > 0,")
print("      ca mean va std deu nhan/chia c, ty so z khong he bi anh huong.")

