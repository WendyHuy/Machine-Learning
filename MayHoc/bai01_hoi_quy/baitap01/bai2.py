# -*- coding: utf-8 -*-
"""Bài tập 2: vẽ giá căn hộ theo số phòng."""

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("data/gia_nha.csv")

plt.figure(figsize=(7, 5))
plt.scatter(df["so_phong"], df["gia"], alpha=0.75)
plt.xlabel("So phong")
plt.ylabel("Gia (ty dong)")
plt.title("Gia can ho theo so phong")
plt.xticks(sorted(df["so_phong"].unique()))
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("baitap01/bai2.png", dpi=150)
plt.show()

print("Nhan xet: Can ho co nhieu phong hon nhin chung co gia cao hon.")