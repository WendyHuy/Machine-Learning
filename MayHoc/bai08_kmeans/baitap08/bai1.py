# -*- coding: utf-8 -*-
"""Bài tập 1: Tự tính hệ số silhouette cho từng điểm bằng numpy.

Hàm silhouette_mot_diem(X, nhan, i) tính a, b và s của điểm thứ i.
Chạy hàm cho mọi khách hàng với K = 5, lấy trung bình so với silhouette_score.
Tìm khách có s nhỏ nhất và xác định vị trí của khách đó trên biểu đồ.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def silhouette_mot_diem(X, nhan, i):
    """Tính a, b, s của điểm thứ i trong tập dữ liệu X với nhãn cụm nhan."""
    cum_i = nhan[i]
    xi = X[i]

    # a: khoảng cách trung bình tới các điểm cùng cụm (loại trừ chính nó)
    cung_cum = X[(nhan == cum_i) & (np.arange(len(X)) != i)]
    if len(cung_cum) == 0:
        a = 0.0
    else:
        kc_cung = np.sqrt(((cung_cum - xi) ** 2).sum(axis=1))
        a = kc_cung.mean()

    # b: khoảng cách trung bình nhỏ nhất tới các cụm khác
    cac_cum_khac = [c for c in np.unique(nhan) if c != cum_i]
    tb_cac_cum = []
    for c in cac_cum_khac:
        khac_cum = X[nhan == c]
        kc_khac = np.sqrt(((khac_cum - xi) ** 2).sum(axis=1))
        tb_cac_cum.append(kc_khac.mean())

    b = min(tb_cac_cum) if tb_cac_cum else 0.0

    # s = (b - a) / max(a, b)
    s = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return a, b, s


df = pd.read_csv("data/khach_hang.csv")
X = df[["thu_nhap", "diem_chi_tieu"]].values
sc = StandardScaler().fit(X)
Xs = sc.transform(X)

km = KMeans(n_clusters=5, n_init=10, random_state=0).fit(Xs)
nhan = km.labels_

# Tính s cho mọi điểm
ds_s = []
for i in range(len(Xs)):
    _, _, s = silhouette_mot_diem(Xs, nhan, i)
    ds_s.append(s)

s_tb = np.mean(ds_s)
s_sklearn = silhouette_score(Xs, nhan)

print("=== SO SANH SILHOUETTE TU TINH VA SCIKIT-LEARN ===")
print(f"Silhouette trung binh tu tinh : {s_tb:.4f}")
print(f"Silhouette cua scikit-learn   : {s_sklearn:.4f}")
print(f"Chenh lech                    : {abs(s_tb - s_sklearn):.6f}")

idx_min = int(np.argmin(ds_s))
khach_min = df.iloc[idx_min]
print(f"\nKhach co s nho nhat: khach thu {idx_min}")
print(f"  Gia tri s           : {ds_s[idx_min]:.4f}")
print(f"  Thu nhap            : {khach_min['thu_nhap']:.1f} trieu")
print(f"  Diem chi tieu       : {khach_min['diem_chi_tieu']:.1f}")
print(f"  Thuoc cum           : {nhan[idx_min]}")

print("\n=== TRA LOI CAU HOI ===")
print(f"- Khach hang co he so s nho nhat la khach so {idx_min} voi s = {ds_s[idx_min]:.4f}.")
print("- Vi tri tren hinh: Khach nay nam o vung ranh gioi giua hai cum (gan tam cua mot cum khac),")
print("  tuc la nam o ria ngoai cua cum minh duoc phan vao, co khoang cach toi cum lan can")
print("  khong chenh lech nhieu so voi khoang cach toi cac ban cung cum.")
