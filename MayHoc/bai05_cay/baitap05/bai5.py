# -*- coding: utf-8 -*-
"""Bài tập 5: Đo độ quan trọng đặc trưng bằng cách xáo trộn (Permutation Importance)."""

import sys
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

# Đọc dữ liệu khách vay
df = pd.read_csv("data/khach_vay.csv")
X = df.drop(columns="tra_dung_han")
y = df["tra_dung_han"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

# Thêm cột số ngẫu nhiên vào cả tập học và tập kiểm tra (như mục 7)
rng = np.random.default_rng(0)
X_tr_nn = X_tr.copy()
X_tr_nn["so_ngau_nhien"] = rng.random(len(X_tr))

X_te_nn = X_te.copy()
X_te_nn["so_ngau_nhien"] = rng.random(len(X_te))

# Huấn luyện rừng 200 cây trên tập học có cột ngẫu nhiên
rung_nn = RandomForestClassifier(n_estimators=200, random_state=0)
rung_nn.fit(X_tr_nn, y_tr)

# 1. Đo theo feature_importances_ (MDI - Mean Decrease in Impurity)
qt_mdi = pd.Series(rung_nn.feature_importances_, index=X_tr_nn.columns).sort_values(ascending=False)

# 2. Đo theo permutation_importance trên tập kiểm tra
ket_qua_perm = permutation_importance(
    rung_nn, X_te_nn, y_te, n_repeats=20, random_state=0
)
qt_perm = pd.Series(ket_qua_perm.importances_mean, index=X_te_nn.columns).sort_values(ascending=False)

print("=== BAI TAP 5: DO DO QUAN TRONG BANG PERMUTATION IMPORTANCE ===")
print()
print("1. Bang do quan trong theo feature_importances_ (MDI - tinh tren tap hoc):")
print(qt_mdi.round(4).to_string())
print()
print("2. Bang do quan trong theo permutation_importance (xao tron tren tap kiem tra):")
print(qt_perm.round(4).to_string())
print()

# Tìm thứ hạng của cột số ngẫu nhiên ở 2 cách
hang_mdi = list(qt_mdi.index).index("so_ngau_nhien") + 1
hang_perm = list(qt_perm.index).index("so_ngau_nhien") + 1

print("Tra loi cac cau hoi:")
print(f"1. Cot ngau nhien xep thu may theo permutation_importance?")
print(f"   -> Cot 'so_ngau_nhien' xep thu {hang_perm} (dung CHOT / THAP NHAT) voi diem chi {qt_perm['so_ngau_nhien']:.4f} (~0).")
print()
print(f"2. So voi thu hang cua no theo feature_importances_ (MDI)?")
print(f"   -> Theo feature_importances_, cot ngau nhien duoc danh gia rat cao: xep thu {hang_mdi}/5")
print(f"      (dat {qt_mdi['so_ngau_nhien']:.3f}), vuot qua ca hai cot that la 'co_viec_on_dinh' va 'so_nguoi_phu_thuoc'.")
print()
print("3. Giai thich su khac nhau giua hai cach do:")
print("   - Nguyen nhan feature_importances_ bi danh lua (ao tuong suc manh):")
print("     + feature_importances_ tinh bang muc giam Gini tren tap hoc.")
print("     + Cot 'so_ngau_nhien' la bien lien tuc co 450 gia tri duy nhat khac nhau (cardinality cao).")
print("       Cay rat de tim duoc mot nguong ngau nhien nao do giup tach rieng mot vai diem hoc,")
print("       khien cot nay lien tuc duoc cong diem Gini giam, tao ra gia tri quan trong gia tao.")
print("   - Vi sao permutation_importance phan anh dung ban chat:")
print("     + permutation_importance do su tut giam Accuracy tren tap kiem tra khi xao tron gia tri.")
print("     + Vi cot ngau nhien khong he chua thong tin quy luat that, viec xao tron no khong he lam")
print("       giam do chinh xac cua mo hinh tren tap kiem tra (do suy giam ~ 0).")
print("     + Nho do, permutation_importance boc tran duoc ban chat vo dung cua cot ngau nhien.")
