# -*- coding: utf-8 -*-
"""Bài tập 2: Kiểm chứng một mạng nơ-ron khác giải bài toán XOR.

Mạng gồm 2 nơ-ron ẩn:
h1 = max(0, x1 - x2)
h2 = max(0, x2 - x1)
y = h1 + h2

Kiểm chứng trên cả 4 điểm của bảng chân trị XOR: (0,0), (0,1), (1,0), (1,1).
Giải thích bằng lời vì sao mạng này đúng.
"""

import numpy as np


def relu(z):
    return np.maximum(0, z)


# 4 trường hợp đầu vào của XOR
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype=float)

print("=== KIEM CHUNG MANG NƠ-RON XOR MOI ===")
print(" x1   x2    h1 = max(0, x1-x2)   h2 = max(0, x2-x1)   y = h1 + h2   XOR chuan")
for x in X:
    x1, x2 = x[0], x[1]
    h1 = relu(x1 - x2)
    h2 = relu(x2 - x1)
    y = h1 + h2
    xor_chuan = int(x1) ^ int(x2)
    print(f"  {int(x1)}    {int(x2)}            {h1:.1f}                  {h2:.1f}             {int(y)}            {xor_chuan}")

print("\n=== GIAI THICH BANG LOI VI SAO MANG NAY DUNG ===")
print("1. Ban chat cua phep XOR giua hai bien nhi phan x1, x2 la:")
print("   - Neu x1 == x2 (cung la 0 hoac cung la 1): dau ra phai bang 0.")
print("   - Neu x1 != x2 (mot ben la 1, ben kia la 0): dau ra phai bang 1.")
print("2. Hoat dong cua tung no-ron an:")
print("   - h1 = max(0, x1 - x2):")
print("     + Khi (x1=1, x2=0): x1 - x2 = 1 > 0 -> h1 = 1.")
print("     + Trong tat ca truong hop con lai (x1 <= x2): x1 - x2 <= 0 -> h1 bi ReLU cat thanh 0.")
print("     => Nhu vay, h1 dong vai tro la bo phat hien trang thai (1, 0).")
print("   - h2 = max(0, x2 - x1):")
print("     + Khi (x1=0, x2=1): x2 - x1 = 1 > 0 -> h2 = 1.")
print("     + Trong tat ca truong hop con lai (x2 <= x1): x2 - x1 <= 0 -> h2 bi ReLU cat thanh 0.")
print("     => Nhu vay, h2 dong vai tro la bo phat hien trang thai (0, 1).")
print("3. No-ron dau ra y = h1 + h2:")
print("   - Khi x1 == x2 (cung la 0 hoac 1): ca h1 va h2 deu bang 0 -> y = 0 + 0 = 0.")
print("   - Khi x1 != x2: dung mot trong hai no-ron h1 hoac h2 bat len bang 1, no-ron kia bang 0 -> y = 1.")
print("=> Vay mang tinh chinh xac 100% logic cua phep XOR ma khong can trong so phuc tap.")
