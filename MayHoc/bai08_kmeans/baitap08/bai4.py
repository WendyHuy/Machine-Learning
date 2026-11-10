# -*- coding: utf-8 -*-
"""Bài tập 4: Nén màu ảnh bằng K-Means và lưu ảnh.

Tạo ảnh 16 màu từ ảnh mẫu china.jpg, lưu thành figures/anh_16_mau.png.
Lưu ảnh gốc thành figures/anh_goc.png và so sánh dung lượng hai tệp.
Thực hiện tiếp với ảnh flower.jpg có sẵn trong scikit-learn.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image

os.makedirs("figures", exist_ok=True)

# 1. Thực hiện với bức ảnh china.jpg
anh = load_sample_image("china.jpg")
cao, rong, _ = anh.shape
diem = anh.reshape(-1, 3) / 255.0

# Lưu ảnh gốc
plt.imsave("figures/anh_goc.png", anh)

# Phân cụm 16 màu trên 5000 điểm ngẫu nhiên
rng = np.random.default_rng(0)
mau = diem[rng.choice(len(diem), 5000, replace=False)]

km16 = KMeans(n_clusters=16, n_init=4, random_state=0).fit(mau)
nhan = km16.predict(diem)
anh_moi = km16.cluster_centers_[nhan].reshape(cao, rong, 3)

# Lưu ảnh 16 màu
plt.imsave("figures/anh_16_mau.png", anh_moi)

# So sánh dung lượng
sz_goc = os.path.getsize("figures/anh_goc.png")
sz_16 = os.path.getsize("figures/anh_16_mau.png")

print("=== 1. SO SANH DUNG LUONG ANH CHINA.JPG ===")
print(f"Kich thuoc anh goc (PNG)    : {sz_goc:,} bytes ({sz_goc / 1024:.1f} KB)")
print(f"Kich thuoc anh 16 mau (PNG) : {sz_16:,} bytes ({sz_16 / 1024:.1f} KB)")
print(f"Ty le giam dung luong       : {sz_goc / sz_16:.2f} lan (giam {(1 - sz_16 / sz_goc) * 100:.1f}%)")

# 2. Làm lại với ảnh flower.jpg
anh_hoa = load_sample_image("flower.jpg")
cao_h, rong_h, _ = anh_hoa.shape
diem_hoa = anh_hoa.reshape(-1, 3) / 255.0

plt.imsave("figures/anh_hoa_goc.png", anh_hoa)

mau_hoa = diem_hoa[rng.choice(len(diem_hoa), 5000, replace=False)]
km_hoa16 = KMeans(n_clusters=16, n_init=4, random_state=0).fit(mau_hoa)
nhan_hoa = km_hoa16.predict(diem_hoa)
anh_hoa_16 = km_hoa16.cluster_centers_[nhan_hoa].reshape(cao_h, rong_h, 3)

plt.imsave("figures/anh_hoa_16_mau.png", anh_hoa_16)

sz_hoa_goc = os.path.getsize("figures/anh_hoa_goc.png")
sz_hoa_16 = os.path.getsize("figures/anh_hoa_16_mau.png")

print("\n=== 2. SO SANH DUNG LUONG ANH FLOWER.JPG ===")
print(f"Kich thuoc anh hoa goc (PNG)   : {sz_hoa_goc:,} bytes ({sz_hoa_goc / 1024:.1f} KB)")
print(f"Kich thuoc anh hoa 16 mau (PNG): {sz_hoa_16:,} bytes ({sz_hoa_16 / 1024:.1f} KB)")
print(f"Ty le giam dung luong          : {sz_hoa_goc / sz_hoa_16:.2f} lan (giam {(1 - sz_hoa_16 / sz_hoa_goc) * 100:.1f}%)")

print("\n=== NHAN XET ===")
print("- Anh 16 mau giam dang ke so luong mau duy nhat, tao thanh cac mang mau dong nhat giup thuat toan nen PNG")
print("  nen hieu qua hon rat nhieu (giam dung luong file).")
print("- Ve thi giac, anh 16 mau van giu duoc hau het cac duong net va mau sac chinh cua buc anh.")
