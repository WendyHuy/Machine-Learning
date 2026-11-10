# -*- coding: utf-8 -*-
"""Bài tập 6: Phân cụm chữ số viết tay bằng K-Means (K = 10).

Chạy K-Means với K = 10 trên load_digits không dùng nhãn.
Dùng pd.crosstab(nhan_that, cum) để kiểm tra độ phân bổ.
Xác định chữ số nào được gom gọn và những chữ số nào hay bị gom chung.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits

digits = load_digits()
X = digits.data
y_that = digits.target

km = KMeans(n_clusters=10, n_init=10, random_state=0).fit(X)
cum = km.labels_

bang = pd.crosstab(pd.Series(y_that, name="Chu so that"),
                   pd.Series(cum, name="Cum K-Means"))
print("=== BANG PHAN BO CHU SO THAT VA CUM K-MEANS (pd.crosstab) ===")
print(bang.to_string())

# Phân tích độ thuần nhất của từng chữ số
print("\n=== DO THUAN NHAT CUA TUNG CHU SO ===")
gom_gon = []
bi_phan_tan = []
for so in range(10):
    phan_bo = bang.loc[so]
    cum_max = phan_bo.idxmax()
    so_max = phan_bo.max()
    tong_so = phan_bo.sum()
    ty_le = so_max / tong_so
    print(f"Chu so {so}: tap trung o cum {cum_max} ({so_max}/{tong_so} mau = {ty_le*100:.1f}%)")
    if ty_le >= 0.85:
        gom_gon.append((so, ty_le))
    else:
        bi_phan_tan.append((so, ty_le))

print("\n=== TRA LOI CAU HOI ===")
print(f"- Nhung chu so duoc gom gon nhat vao mot cum (do thuan nhat cao):")
for so, tl in sorted(gom_gon, key=lambda x: x[1], reverse=True):
    print(f"  + Chu so {so} ({tl*100:.1f}% mau nam trong cung 1 cum)")

print("\n- Nhung chu so hay bi gom chung hoac phan tan qua nhieu cum:")
for so, tl in sorted(bi_phan_tan, key=lambda x: x[1]):
    print(f"  + Chu so {so} (chi co {tl*100:.1f}% mau o cum dong nhat)")
print("- Giai thich: Cac chu so nhu 0, 4, 6 co hinh dang net viet rat dac trung (vong tron kin, goc vuong),")
print("  it bi nham lan nen duoc gom rat gon.")
print("  Nguoc lai, cac cap chu so nhu (1, 8), (3, 9), (7, 1) co nhieu net tuong dong trong khong gian 64 chieu,")
print("  nen de bi K-Means phan vao chung mot cum hoac bi xet ra cac cum khac nhau.")
