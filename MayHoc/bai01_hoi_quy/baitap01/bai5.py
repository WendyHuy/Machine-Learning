# -*- coding: utf-8 -*-
"""Bài tập 5: thử gradient descent với hai tốc độ học khác nhau."""

import pandas as pd

df = pd.read_csv("data/gia_nha.csv")
x_goc = df["dien_tich"].to_numpy()
y = df["gia"].to_numpy()
x = (x_goc - x_goc.mean()) / x_goc.std()
n = len(x)
so_vong = 200


def chay_gradient_descent(toc_do_hoc):
    """Chạy 200 vòng và trả về MSE ở vòng cuối."""
    w, b = 0.0, 0.0
    for _ in range(so_vong):
        chenh_lech = (w * x + b) - y
        grad_w = (2 / n) * (chenh_lech * x).sum()
        grad_b = (2 / n) * chenh_lech.sum()
        w -= toc_do_hoc * grad_w
        b -= toc_do_hoc * grad_b
    return (((w * x + b) - y) ** 2).mean()


for toc_do_hoc in (0.001, 1.02):
    mse = chay_gradient_descent(toc_do_hoc)
    print(f"Toc do hoc {toc_do_hoc}: MSE o vong 200 = {mse:.4e}")

print("Giai thich: Toc do 0.001 cho buoc di qua ngan nen sau 200 vong chua toi day.")
print("Toc do 1.02 cho buoc di qua dai, nhay qua day va lam MSE tang rat lon.")
