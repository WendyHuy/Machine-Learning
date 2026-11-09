# -*- coding: utf-8 -*-
"""Bài tập 5: Chữ số viết tay, những lỗi nào hay gặp (Confusion Matrix)."""

import sys
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

os.makedirs("figures", exist_ok=True)

# 1. Nạp dữ liệu và chia tập học / kiểm tra giống mục 6
X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

# 2. Huấn luyện SVM kernel RBF mặc định
svm = SVC(kernel="rbf").fit(X_tr, y_tr)
du_doan = svm.predict(X_te)

print("=== BAI TAP 5: CHU SO VIET TAY - MA TRAN NHAM LAN ===")
print()

# 3. Tính bảng nhầm lẫn 10 hàng 10 cột
cm = confusion_matrix(y_te, du_doan)
print("Bang nham lan 10 hang 10 cot (Hang: That, Cot: Doan):")
print("That\\Doan   0   1   2   3   4   5   6   7   8   9")
print("-------------------------------------------------")
for i, row in enumerate(cm):
    row_str = " ".join(f"{val:3d}" for val in row)
    print(f"    {i}    |{row_str}")

print()

# 4. Xác định các cặp bị nhầm lẫn
sai_idx = np.flatnonzero(du_doan != y_te)
print(f"Tong so anh doan sai: {len(sai_idx)} tren {len(y_te)} anh kiem tra.")
print()
print("Chi tiet tung anh bi nham:")
for i in sai_idx:
    that = y_te[i]
    doan = du_doan[i]
    print(f"  - Anh chi so {i:3d}: chu so that la {that}, mo hinh doan nham thanh {doan}")

# 5. Vẽ và lưu từng ảnh bị đoán sai
fig, axes = plt.subplots(1, len(sai_idx), figsize=(10, 3.2))
for ax, i in zip(axes, sai_idx):
    that = y_te[i]
    doan = du_doan[i]
    img = X_te[i].reshape(8, 8)

    # Lưu ảnh riêng lẻ
    plt.figure(figsize=(2.5, 2.5))
    plt.imshow(img, cmap="gray_r")
    plt.title(f"Anh {i}: That {that} -> Doan {doan}", fontsize=11, fontweight="bold")
    plt.axis("off")
    plt.savefig(f"figures/sai_anh_{i}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Vẽ vào ảnh ghép
    ax.imshow(img, cmap="gray_r")
    ax.set_title(f"Anh {i}\nThat: {that} | Doan: {doan}", fontsize=10)
    ax.axis("off")

fig.suptitle("Cac anh chu so bi doan sai boi SVM", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig("figures/cac_anh_doan_sai.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print()
print("Da luu cac anh doan sai vao thu muc figures/:")
for i in sai_idx:
    print(f"  + figures/sai_anh_{i}.png")
print("  + figures/cac_anh_doan_sai.png")
print()

# 6. Nhận xét nguyên nhân nhầm lẫn
print("Tra loi cac cau hoi:")
print("1. Nhung cap chu so nao bi nham?")
print("   - Cap (That 4, Doan 8): Anh chi so 161")
print("   - Cap (That 7, Doan 9): Anh chi so 428")
print("   - Cap (That 8, Doan 1): Anh chi so 18")
print("   - Cap (That 9, Doan 3): Anh chi so 6")
print()
print("2. Vi sao SVM nham o cac truong hop tren?")
print("   - Anh 6 (That 9, Doan 3): Net tren cua so 9 bi dut doan, khong kin thanh vong tron ma giong")
print("     hinh vong cung cua so 3, phan duoi cung hoi uon nhe nen SVM de nham thanh so 3.")
print("   - Anh 18 (That 8, Doan 1): Chu so 8 nay duoc viet rat hep va manh theo chieu doc, hai vong tron")
print("     gan nhu ep sat vao nhau thanh mot vach thang dung, rat giong net ke cua chu so 1.")
print("   - Anh 161 (That 4, Doan 8): Net ngang va net moc cua so 4 bi noi dinh vao nhau, tao cam giac")
print("     co hai lo kin giong hai vong cua chu so 8 o do phan giai thap 8x8.")
print("   - Anh 428 (That 7, Doan 9): Nhanh cheo cua so 7 hoi uon cong sang trai o giua than, lam cho")
print("     khong gian diem sang 8x8 trong giong nhu mot nua tren cua chu so 9.")
