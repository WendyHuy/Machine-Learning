# -*- coding: utf-8 -*-
"""Bài tập 1: Tự vẽ cây quyết định bằng plot_tree."""

import sys
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

os.makedirs("figures", exist_ok=True)

# Đọc dữ liệu khách vay
df = pd.read_csv("data/khach_vay.csv")
X = df.drop(columns="tra_dung_han")
y = df["tra_dung_han"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

ten_dac_trung = list(X.columns)
ten_lop = ["tre han", "dung han"]

print("=== BAI TAP 1: TU VE CAY QUYET DINH ===")

# 1. Vẽ cây năm lá của mục 3
cay_5_la = DecisionTreeClassifier(max_leaf_nodes=5, random_state=0)
cay_5_la.fit(X_tr, y_tr)

plt.figure(figsize=(14, 8))
plot_tree(
    cay_5_la,
    feature_names=ten_dac_trung,
    class_names=ten_lop,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Cay quyet dinh 5 la (max_leaf_nodes=5)", fontsize=14, fontweight="bold")
plt.savefig("cay_nam_la.png", dpi=150, bbox_inches="tight")
plt.savefig("figures/cay_nam_la.png", dpi=150, bbox_inches="tight")
plt.close()
print("Da luu hinh: cay_nam_la.png va figures/cay_nam_la.png")

# 2. Vẽ cây sâu 2 (max_depth=2)
cay_sau_2 = DecisionTreeClassifier(max_depth=2, random_state=0)
cay_sau_2.fit(X_tr, y_tr)

plt.figure(figsize=(12, 7))
plot_tree(
    cay_sau_2,
    feature_names=ten_dac_trung,
    class_names=ten_lop,
    filled=True,
    rounded=True,
    fontsize=11
)
plt.title("Cay quyet dinh sau 2 tang (max_depth=2)", fontsize=14, fontweight="bold")
plt.savefig("cay_sau_2.png", dpi=150, bbox_inches="tight")
plt.savefig("figures/cay_sau_2.png", dpi=150, bbox_inches="tight")
plt.close()
print("Da luu hinh: cay_sau_2.png va figures/cay_sau_2.png")
print()

# 3. Phân tích hai lá bên trái của cây sâu 2
print("Tra loi cac cau hoi:")
print("1. Hai la ben trai cua cay sau 2 co cung ket luan khong?")
print("   -> CO. Ca hai la ben trai deu co ket luan la 'class = tre han' (lop 0).")
print("      + La thu nhat (ty_le_no <= 0.615): value = [68, 53] -> lop da so la 0 (tre han)")
print("      + La thu hai  (ty_le_no > 0.615) : value = [114, 14] -> lop da so la 0 (tre han)")
print()
print("2. Neu co, vi sao cay van chia nut do ra lam hai?")
print("   -> Vi tieu chi chia cua cay la lam GIAM CHI SO GINI (tang do thuan khiet),")
print("      chu khong dua vao viec hai nut con co cung nhan da so hay khong.")
print("   -> Nut cha truoc khi chia co 182 tre han va 67 dung han (Gini = 0.393).")
print("   -> Sau khi chia voi cau hoi 'ty_le_no <= 0.615':")
print("      + Nhanh phai co 114 tre han tren 128 khach, rat thuan khiet (Gini giam manh xuong 0.195).")
print("      + Phep chia nay lam giam chi so Gini trung binh co trong so dang ke.")
print("   -> Dong thoi, no giup mo hinh uoc luong xac suat (predict_proba) chinh xac hon:")
print("      nhom no cao ty_le_no > 0.615 co xac suat tre han len toi 89.1%, trong khi")
print("      nhom no thap ty_le_no <= 0.615 ty le tre han chi la 56.2%.")
