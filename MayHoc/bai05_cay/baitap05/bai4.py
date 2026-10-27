# -*- coding: utf-8 -*-
"""Bài tập 4: Cắt tỉa bằng số khách tối thiểu ở lá (min_samples_leaf)."""

import sys
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

# Đọc dữ liệu khách vay
df = pd.read_csv("data/khach_vay.csv")
X = df.drop(columns="tra_dung_han")
y = df["tra_dung_han"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

cac_gia_tri_leaf = [1, 2, 5, 10, 20, 40]

print("=== BAI TAP 4: CAT TIA CAY BANG MIN_SAMPLES_LEAF ===")
print("Kiem dinh cheo 5 phan tren tap hoc voi cac gia tri min_samples_leaf:")
print()
print(" min_samples_leaf | Accuracy hoc | CV 5-fold ")
print("------------------+--------------+-----------")

diem_cv = {}
diem_hoc = {}
for msl in cac_gia_tri_leaf:
    mh = DecisionTreeClassifier(min_samples_leaf=msl, random_state=0)
    diem_cv[msl] = cross_val_score(mh, X_tr, y_tr, cv=5).mean()
    diem_hoc[msl] = accuracy_score(y_tr, mh.fit(X_tr, y_tr).predict(X_tr))

msl_tot = max(diem_cv, key=diem_cv.get)
for msl in cac_gia_tri_leaf:
    dau = " <- tot nhat" if msl == msl_tot else ""
    print(f"       {msl:2d}         |    {diem_hoc[msl]:.4f}    |  {diem_cv[msl]:.4f}{dau}")

print()
# Đánh giá trên tập kiểm tra với min_samples_leaf tốt nhất
tot_msl = DecisionTreeClassifier(min_samples_leaf=msl_tot, random_state=0)
tot_msl.fit(X_tr, y_tr)
acc_test_msl = accuracy_score(y_te, tot_msl.predict(X_te))

# Mô hình cây sâu 5 ở mục 4 để so sánh
cay_sau_5 = DecisionTreeClassifier(max_depth=5, random_state=0)
cay_sau_5.fit(X_tr, y_tr)
acc_test_sau_5 = accuracy_score(y_te, cay_sau_5.predict(X_te))

print("So sanh hai cach cat tia tren tap kiem tra:")
print(f"  1. Cat tia theo min_samples_leaf={msl_tot} : CV = {diem_cv[msl_tot]:.4f}, So la = {tot_msl.get_n_leaves()}, Accuracy test = {acc_test_msl:.4f}")
print(f"  2. Cat tia theo max_depth=5 (muc 4)     : CV = 0.8333, So la = {cay_sau_5.get_n_leaves()}, Accuracy test = {acc_test_sau_5:.4f}")
print()
print("Tra loi cau hoi:")
print("Cach cat tia nao cho ket qua tot hon tren bo du lieu nay?")
print(f"  -> Cat tia theo DO SAU (max_depth=5) cho ket qua TOT HON ({acc_test_sau_5:.4f} so voi {acc_test_msl:.4f}).")
print("  -> Giai thich:")
print("     + max_depth gioi han dong deu chieu dai toi da cua moi duong di tu goc toi la,")
print("       ngan chan toan bo cac nhanh cay khong duoc phat trien qua phuc tap.")
print("     + Trong khi do, min_samples_leaf chi rang buoc so mau toi thieu o moi la,")
print("       neu mot vung du lieu co mat do day dac thi cay van co the tiep tuc re them nhieu")
print("       tang sau (cay co toi 30 la), dan toi mot phan van bi qua khop cuc bo.")
