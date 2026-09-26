# -*- coding: utf-8 -*-
"""Bài tập 6: viết hàm dự đoán giá có cảnh báo ngoại suy."""

w = 0.078367
b = 0.401752


def du_doan_gia(dien_tich):
    """Trả về giá dự đoán và cảnh báo nếu diện tích ngoài dữ liệu đã học."""
    if dien_tich < 35.5 or dien_tich > 117.5:
        print(f"Canh bao: {dien_tich} m2 nam ngoai khoang du lieu 35.5-117.5 m2.")
    return w * dien_tich + b


for dien_tich in (60, 80, 200):
    gia = du_doan_gia(dien_tich)
    print(f"Can {dien_tich} m2 -> du doan {gia:.3f} ty dong")
