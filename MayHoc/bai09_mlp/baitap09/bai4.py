# -*- coding: utf-8 -*-
"""Bài tập 4: Khảo sát ảnh hưởng của tốc độ học (learning_rate_init).

Dùng mạng MLP 1 lớp ẩn 100 nơ-ron trên bộ load_digits.
Lần lượt thử nghiệm learning_rate_init: 0.0001, 0.001, 0.01, 0.1.
Ghi lại n_iter_, loss cuối cùng và accuracy tập kiểm tra.
Vẽ 4 đường loss_curve_ lưu vào figures/toc_do_hoc.png.
Giải thích hiện tượng khi tốc độ học quá lớn hoặc quá nhỏ.
"""

import os
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

os.makedirs("figures", exist_ok=True)

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

# Chuẩn hóa dữ liệu trước để dùng trực tiếp cho MLPClassifier
sc = StandardScaler()
X_tr_s = sc.fit_transform(X_tr)
X_te_s = sc.transform(X_te)

cac_lr = [0.0001, 0.001, 0.01, 0.1]
ket_qua = []
cac_loss_curve = {}

print("=== ANH HUONG CUA TOC DO HOC (LEARNING RATE) ===")
print("Learning rate   So vong (n_iter_)   Mat mat cuoi   Accuracy tap kiem tra")

for lr_val in cac_lr:
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500,
                        learning_rate_init=lr_val, random_state=0)
    mlp.fit(X_tr_s, y_tr)
    loss_cuoi = mlp.loss_curve_[-1]
    acc_te = accuracy_score(y_te, mlp.predict(X_te_s))
    ket_qua.append((lr_val, mlp.n_iter_, loss_cuoi, acc_te))
    cac_loss_curve[lr_val] = mlp.loss_curve_
    print(f"   {lr_val:<8}           {mlp.n_iter_:3d}             {loss_cuoi:.4f}             {acc_te:.4f}")

# Vẽ đồ thị 4 đường loss_curve_
plt.figure(figsize=(8, 5))
for lr_val, duong_loss in cac_loss_curve.items():
    plt.plot(duong_loss, label=f"lr = {lr_val}")

plt.xlabel("So vong hoc (Epoch / Iteration)", fontsize=11)
plt.ylabel("Gia tri mat mat (Loss)", fontsize=11)
plt.title("Duong mat mat (Loss Curve) voi cac toc do hoc khac nhau", fontsize=12)
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("figures/toc_do_hoc.png", dpi=150)
plt.close()
print("\nDa luu bieu do vao figures/toc_do_hoc.png")

print("\n=== TRA LOI CAU HOI ===")
print("- Khi toc do hoc qua nho (e.g. 0.0001):")
print("  + Mang hoc rat cham, buoc giam do doc qua ngan.")
print("  + Phai ton rat nhieu vong lap (co the het max_iter ma van chua hoi tu ve day cuc tieu).")
print("  + Mat mat giam tu tu, mo hinh de bi ket o diem kem neu dung som.")
print("- Khi toc do hoc qua lon (e.g. 0.1):")
print("  + Buoc cap nhat trong so qua dai, de bi nhay vot qua diem cuc tieu (overshooting).")
print("  + Duong mat mat co the bi dao dong manh, tham chi phan ky hoac dung som o mot nghiem phu khong tot.")
print("- Toc do hoc mac dinh (0.001) la diem can bang hop ly nhat: hoi tu nhanh, deu dan va dat accuracy cao.")
