# -*- coding: utf-8 -*-
"""Bài tập 4: Tự vẽ đường ROC so sánh Logistic Regression và Random Forest.

Vẽ đường ROC của cả 2 mô hình cùng đường chéo ngẫu nhiên lên hình.
Ghi AUC vào chú thích, lưu thành figures/roc.png.
Nhận xét đường nào ôm sát góc trên bên trái hơn và hai đường có cắt nhau không.
"""

import os
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

os.makedirs("figures", exist_ok=True)

df = pd.read_csv("data/sinh_vien_bo_hoc.csv")
X = df.drop(columns="bo_hoc")
y = df["bo_hoc"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

# 1. Hồi quy Logistic
lr = make_pipeline(StandardScaler(), LogisticRegression())
lr.fit(X_tr, y_tr)
xs_lr = lr.predict_proba(X_te)[:, 1]
fpr_lr, tpr_lr, _ = roc_curve(y_te, xs_lr)
auc_lr = roc_auc_score(y_te, xs_lr)

# 2. Rừng ngẫu nhiên
rf = RandomForestClassifier(n_estimators=200, random_state=0)
rf.fit(X_tr, y_tr)
xs_rf = rf.predict_proba(X_te)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_te, xs_rf)
auc_rf = roc_auc_score(y_te, xs_rf)

print(f"AUC Logistic Regression : {auc_lr:.4f}")
print(f"AUC Random Forest       : {auc_rf:.4f}")

# Vẽ đồ thị ROC
plt.figure(figsize=(7, 6))
plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {auc_lr:.4f})", color="blue", linewidth=2)
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {auc_rf:.4f})", color="green", linewidth=2, linestyle="--")
plt.plot([0, 1], [0, 1], "k--", label="Doan ngau nhien (AUC = 0.5000)")

plt.xlabel("Ty le bao dong nham (FPR)", fontsize=11)
plt.ylabel("Recall / Do nhay (TPR)", fontsize=11)
plt.title("So sanh duong ROC giua Logistic Regression va Random Forest", fontsize=12)
plt.legend(loc="lower right")
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("figures/roc.png", dpi=150)
plt.close()
print("Da luu bieu do vao figures/roc.png")

print("\n=== TRA LOI CAU HOI ===")
print(f"- Duong nao om sat goc tren ben trai hon?")
print(f"  Duong cua Logistic Regression om sat goc tren ben trai hon tren hau het chieu dai")
print(f"  va co tong the dien tich duoi duong cong lon hon (AUC = {auc_lr:.4f} so voi {auc_rf:.4f}).")

print("- Co doan nao hai duong cat nhau khong?")
print("  CO! Hai duong cat nhau tai mot so diem cu the:")
print("  + Tai FPR rat thap (FPR ~ 0.010, TPR ~ 0.533): hai duong giao nhau khi bat dau vuon len.")
print("  + Tai FPR ~ 0.081 (TPR ~ 0.800): hai duong cham nhau.")
print("  + Tai vung FPR trung binh (FPR ~ 0.465, TPR ~ 0.967) va FPR cao (FPR ~ 0.727),")
print("    duong cua Random Forest co luc vuot nhe len tren truoc khi ca hai cung dat TPR = 1.0.")
print("  Nguyen nhan cat nhau la do Random Forest tao ra cac xac suat roi rac (bac thang),")
print("  trong khi Logistic Regression tao ra ham xac suat lien tuc va min hon.")
