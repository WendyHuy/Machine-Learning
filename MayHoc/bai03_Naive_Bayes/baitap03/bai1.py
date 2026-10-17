# -*- coding: utf-8 -*-
"""Bai tap 1: Bayes voi ba nguyen nhan."""

import sys
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def bayes_ba_nhanh(tien_nghiem, kha_nang):
    """Tinh xac suat hau nghiem P(nhom_i | qua_mon) cho ba nhom.

    tien_nghiem : danh sach [P(N1), P(N2), P(N3)]
    kha_nang    : danh sach [P(qua | N1), P(qua | N2), P(qua | N3)]
    """
    # Cong thuc xac suat toan phan: P(qua) = tong(P(Ni) * P(qua | Ni))
    tu_so = [p * k for p, k in zip(tien_nghiem, kha_nang)]
    bang_chung = sum(tu_so)  # P(qua mon)
    hau_nghiem = [t / bang_chung for t in tu_so]
    return hau_nghiem, bang_chung


# Thong so de bai
ten_nhom = ["Hoc deu", "Hoc don", "It hoc"]
tien_nghiem = [0.50, 0.30, 0.20]  # Ty le sinh vien moi nhom
kha_nang = [0.95, 0.70, 0.30]     # Ty le qua mon cua tung nhom

hau_nghiem, p_qua = bayes_ba_nhanh(tien_nghiem, kha_nang)

print("Xac suat qua mon chung cua lop P(qua mon) = {:.4f}".format(p_qua))
print()
print("Xac suat ban da qua mon thuoc tung nhom:")
for ten, tn, kn, hn in zip(ten_nhom, tien_nghiem, kha_nang, hau_nghiem):
    print(f"  P({ten:7s} | qua mon) = {hn:.4f}  (P(N) = {tn:.2f}, P(qua|N) = {kn:.2f})")

tong_hau_nghiem = sum(hau_nghiem)
print()
print(f"Tong cac xac suat hau nghiem = {tong_hau_nghiem:.4f} (chinh xac bang 1)")

