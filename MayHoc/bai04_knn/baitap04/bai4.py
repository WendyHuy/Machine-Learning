# -*- coding: utf-8 -*-
"""Bài tập 4: Láng giềng có trọng số (weights='distance')."""

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

# Đọc dữ liệu hai vầng trăng
df = pd.read_csv("data/hai_vang_trang.csv")
X = df[["x1", "x2"]]
y = df["nhan"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

ds_k = [1, 13, 101]

print("=== BAI TAP 4: LANG GIENG CO TRONG SO (WEIGHTS='DISTANCE') ===")
print("So sanh giua weights='uniform' (muc 4) va weights='distance':")
print()
print("      |        weights='uniform'       |       weights='distance'")
print("  k   |   Tap hoc       Tap kiem tra   |   Tap hoc       Tap kiem tra")
print("------+--------------------------------+-------------------------------")

ket_qua = {}
for k in ds_k:
    # Mô hình không trọng số (uniform)
    mh_u = KNeighborsClassifier(n_neighbors=k, weights="uniform").fit(X_tr, y_tr)
    acc_tr_u = accuracy_score(y_tr, mh_u.predict(X_tr))
    acc_te_u = accuracy_score(y_te, mh_u.predict(X_te))

    # Mô hình có trọng số nghịch đảo khoảng cách (distance)
    mh_d = KNeighborsClassifier(n_neighbors=k, weights="distance").fit(X_tr, y_tr)
    acc_tr_d = accuracy_score(y_tr, mh_d.predict(X_tr))
    acc_te_d = accuracy_score(y_te, mh_d.predict(X_te))

    ket_qua[k] = (acc_tr_u, acc_te_u, acc_tr_d, acc_te_d)
    print(f" {k:3d}  |    {acc_tr_u:.4f}          {acc_te_u:.4f}      |    {acc_tr_d:.4f}          {acc_te_d:.4f}")

print()
print("Tra loi cau hoi:")
print("Voi k = 101, trong so theo khoang cach co giup mo hinh bot cau tha khong? Vi sao?")
print()
print(f"1. Co, mo hinh bot cau tha ro ret:")
print(f"   Accuracy tren tap kiem tra tang tu {ket_qua[101][1]:.4f} (uniform) len {ket_qua[101][3]:.4f} (distance).")
print()
print("2. Vi sao?")
print("   - Khi dung weights='uniform', tat ca 101 lang gieng (chiem gan mot nua tap hoc 225 mau)")
print("     deu co quyen bo phieu ngang nhau. Cac diem o rat xa se ap dao va lam bien quyet dinh")
print("     bi san phang thanh mot duong gan nhu thang, khong bam duoc hinh hai vang trang (cau tha).")
print("   - Khi dung weights='distance', moi phieu duoc nhan voi trong so w = 1 / d. Nhung lang gieng")
print("     o rat gan co khoang cach nho -> trong so rat lon; cac diem o xa co khoang cach lon -> trong")
print("     so rat nho, gan nhu khong anh huong. Nho do, mo hinh van giu duoc tinh cuc bo va uon luan")
print("     theo ranh gioi cong cua du lieu, giup tranh hien tuong cau tha khi k lon.")

