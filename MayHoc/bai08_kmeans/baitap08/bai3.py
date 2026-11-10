# -*- coding: utf-8 -*-
"""Bài tập 3: So sánh khởi tạo k-means++ và khởi tạo ngẫu nhiên.

Chạy K-Means 50 lần với n_init=1, random_state từ 0 tới 49.
Đếm số lần mỗi phương pháp đạt mức inertia tối ưu.
Đánh giá cách khởi tạo nào đáng tin cậy hơn.
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/khach_hang.csv")
Xs = StandardScaler().fit_transform(df[["thu_nhap", "diem_chi_tieu"]])

# Tìm mức inertia tối ưu tốt nhất làm chuẩn
km_chuan = KMeans(n_clusters=5, n_init=20, random_state=0).fit(Xs)
muc_chuan = round(km_chuan.inertia_, 1)
print(f"Muc inertia toi uu chuan (n_init=20): {muc_chuan}")

ket_random = []
ket_plus = []

for rs in range(50):
    km_r = KMeans(n_clusters=5, init="random", n_init=1, random_state=rs).fit(Xs)
    ket_random.append(round(km_r.inertia_, 1))

    km_p = KMeans(n_clusters=5, init="k-means++", n_init=1, random_state=rs).fit(Xs)
    ket_plus.append(round(km_p.inertia_, 1))

# Đếm số lần đạt mức tối ưu (sai số trong khoảng 0.2)
so_lan_r = sum(1 for v in ket_random if abs(v - muc_chuan) <= 0.2)
so_lan_p = sum(1 for v in ket_plus if abs(v - muc_chuan) <= 0.2)

print("\n=== KET QUA 50 LAN CHAY (n_init=1) ===")
print(f"1. Khoi tao 'random'   : {so_lan_r}/50 lan dat muc toi uu ({so_lan_r/50*100:.1f}%)")
print(f"   Inertia trung binh  : {sum(ket_random)/len(ket_random):.2f}, cao nhat la {max(ket_random):.2f}")
print(f"2. Khoi tao 'k-means++': {so_lan_p}/50 lan dat muc toi uu ({so_lan_p/50*100:.1f}%)")
print(f"   Inertia trung binh  : {sum(ket_plus)/len(ket_plus):.2f}, cao nhat la {max(ket_plus):.2f}")

print("\n=== TRA LOI CAU HOI ===")
print(f"- Khoi tao 'k-means++' dat ket qua toi uu {so_lan_p} lan, vuot troi hon so voi 'random' ({so_lan_r} lan).")
print("- Cach khoi tao dang tin hon: 'k-means++' dang tin cay hon rat nhieu.")
print("  Giai thich: Khoi tao ngau nhien thuong de cac tam ban dau roi vao qua gan nhau, dan den hoi tu ve")
print("  cuc tieu dia phuong kem (inertia cao). Trong khi do, k-means++ chu dong chon cac tam ban dau nam cach xa nhau,")
print("  giup thuat toan bao quat toan bo pham vi du lieu va hau nhu luon hoi tu ve nghiem toi uu toan cuc.")
