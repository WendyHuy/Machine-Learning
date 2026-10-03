# -*- coding: utf-8 -*-
"""Bai tap 2: ve ham sigmoid va luu thanh sigmoid.png."""

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


z = np.linspace(-8, 8, 400)   # 400 diem deu nhau tu -8 toi 8

plt.figure(figsize=(7, 4))
plt.plot(z, sigmoid(z), linewidth=2, label="sigmoid(z)")
plt.axhline(0.5, color="gray", linestyle=":")   # duong ngang tai 0.5
plt.axvline(0, color="gray", linestyle=":")     # duong doc tai z = 0
plt.xlabel("z")
plt.ylabel("sigmoid(z)")
plt.title("Ham sigmoid")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("sigmoid.png", dpi=150, bbox_inches="tight")
print("Da luu sigmoid.png")