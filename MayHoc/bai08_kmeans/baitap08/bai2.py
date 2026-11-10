# -*- coding: utf-8 -*-
"""Bài tập 2: Phân cụm K-Means khi thêm cột tuổi.

Phân cụm trên cả ba cột: tuoi, thu_nhap, diem_chi_tieu (có chuẩn hoá).
Chọn lại K tối ưu bằng chỉ số silhouette với K từ 2 tới 10.
So sánh xem K tốt nhất có còn là 5 và tâm các cụm thay đổi ra sao.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/khach_hang.csv")
X = df[["tuoi", "thu_nhap", "diem_chi_tieu"]]
sc = StandardScaler().fit(X)
Xs = sc.transform(X)

print("=== TIM K TOI UU TREN CA 3 COT (TUOI, THU NHAP, CHI TIEU) ===")
print(" K   Inertia    Silhouette")
diem_sil = {}
for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xs)
    sil = silhouette_score(Xs, km.labels_)
    diem_sil[k] = sil
    print(f"{k:2d}   {km.inertia_:7.2f}     {sil:.4f}")

k_tot = max(diem_sil, key=diem_sil.get)
print(f"\nK cho silhouette cao nhat: K = {k_tot} (silhouette = {diem_sil[k_tot]:.4f})")

# Khớp với K tốt nhất và đổi tâm cụm về đơn vị gốc
km_tot = KMeans(n_clusters=k_tot, n_init=10, random_state=0).fit(Xs)
tam_goc = pd.DataFrame(sc.inverse_transform(km_tot.cluster_centers_),
                       columns=["tuoi", "thu_nhap", "diem_chi_tieu"])
tam_goc["so_khach"] = np.bincount(km_tot.labels_)

print(f"\n=== TAM CUM VOI K = {k_tot} (DON VI GOC) ===")
print(tam_goc.round(1).to_string())

print("\n=== TRA LOI CAU HOI ===")
print(f"- K tot nhat co con la 5 khong? -> K tot nhat hien tai la K = {k_tot}.")
print("- Tam cac cum thay doi the nao so voi muc 3:")
print("  Khi them bien 'tuoi', thuoc do khoang cach bay gio phu thuoc ca vao do chenh lech tuoi.")
print("  Cac nhom khach hang duoc tach them theo yeu to do tuoi (vi du: nguoi tre thu nhap thap vs nguoi lon tuoi thu nhap thap).")
print("  Cac tam cum bay gio mo ta chi tiet hon ca 3 khia canh (tuoi, thu nhap, muc do chi tieu),")
print("  giup phan khuc khach hang phan anh ro net hon hanh vi theo lua tuoi.")
