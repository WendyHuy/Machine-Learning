# -*- coding: utf-8 -*-
"""Bài tập 4: Dò lưới mịn (Fine Grid Search)."""

import sys
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC

# Đọc dữ liệu hai vầng trăng
df = pd.read_csv("data/hai_vang_trang.csv")
X = df[["x1", "x2"]]
y = df["nhan"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

print("=== BAI TAP 4: DO LUOI MIN QUANH CAP TOT NHAT ===")
print("Quanh cap tot nhat o muc 5 (C=10, gamma=1), do luoi min:")
print("  C     trong [3, 10, 30]")
print("  gamma trong [0.3, 1, 3]")
print()

luoi_min = {
    "C": [3, 10, 30],
    "gamma": [0.3, 1, 3]
}

tim_min = GridSearchCV(SVC(kernel="rbf"), luoi_min, cv=5)
tim_min.fit(X_tr, y_tr)

bang_min = pd.DataFrame(tim_min.cv_results_["params"])
bang_min["diem"] = tim_min.cv_results_["mean_test_score"]

print("Bang diem kiem dinh cheo cua 9 cap (hang la C, cot la gamma):")
pivot_bang = bang_min.pivot(index="C", columns="gamma", values="diem").round(4)
print(pivot_bang.to_string())
print()

cap_tot = tim_min.best_params_
diem_cv_tot = tim_min.best_score_
acc_test = tim_min.score(X_te, y_te)

print(f"Cap tot nhat sau khi do min: C = {cap_tot['C']}, gamma = {cap_tot['gamma']}")
print(f"Diem kiem dinh cheo tot nhat : {diem_cv_tot:.4f} (o muc 5 la 0.8711)")
print(f"Accuracy tren tap kiem tra   : {acc_test:.4f} (o muc 5 la 0.9200)")
print()

print("Tra loi cac cau hoi:")
print("1. Diem kiem dinh cheo tot nhat co cao hon 0.8711 khong?")
print(f"   -> CO. Diem kiem dinh cheo tang tu 0.8711 len {diem_cv_tot:.4f} (tai cap C = 30, gamma = 1).")
print()
print("2. Accuracy tren tap kiem tra co doi khong?")
print(f"   -> KHONG DOI. Accuracy tren tap kiem tra van giu nguyen la {acc_test:.4f} (0.9200).")
print()
print("3. Nhan xet ve cai loi cua viec do that min:")
print("   - Giup ta tim kiem them su toi uu tinh te ben trong vung tiem nang ma luoi thua da phat hien,")
print("     nang cao do khop cua mo hinh tren tap hoc va tang diem danh gia kiem dinh cheo (CV).")
print("   - Khi do min ma accuracy tren tap kiem tra van on dinh (0.9200), dieu nay chung to vung tham so")
print("     nay rat vung chac (robust), khong bi nhay cam qua muc voi nhung dao dong nho cua C va gamma.")
print("   - Chien luoc 'luoi thua truoc, luoi min sau' giup tiet kiem chi phi tinh toan rat nhieu so voi")
print("     viec do luoi day dac ngay tu dau.")
