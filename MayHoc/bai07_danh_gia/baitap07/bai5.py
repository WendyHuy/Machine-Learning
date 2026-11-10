# -*- coding: utf-8 -*-
"""Bài tập 5: Cân bằng lớp cho RandomForestClassifier và SVC.

Thêm class_weight="balanced" cho RandomForestClassifier và SVC.
So sánh precision, recall, F1 trên tập kiểm tra trước và sau khi cân bằng.
Đánh giá mô hình nào thay đổi nhiều nhất.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

df = pd.read_csv("data/sinh_vien_bo_hoc.csv")
X = df.drop(columns="bo_hoc")
y = df["bo_hoc"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

models = {
    "Random Forest (mac dinh)": RandomForestClassifier(n_estimators=200, random_state=0),
    "Random Forest (balanced)": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=0),
    "SVC RBF (mac dinh)"     : make_pipeline(StandardScaler(), SVC(random_state=0)),
    "SVC RBF (balanced)"     : make_pipeline(StandardScaler(), SVC(class_weight="balanced", random_state=0)),
}

results = []
for ten, mh in models.items():
    mh.fit(X_tr, y_tr)
    du = mh.predict(X_te)
    p = precision_score(y_te, du, zero_division=0)
    r = recall_score(y_te, du, zero_division=0)
    f = f1_score(y_te, du, zero_division=0)
    results.append({"Mo hinh": ten, "Precision": p, "Recall": r, "F1": f})

bang = pd.DataFrame(results)
print("=== KET QUA SO SANH TRUOC VA SAU KHI CAN BANG LOP ===")
print(bang.to_string(index=False))

# Đánh giá thay đổi
delta_rf = abs(results[1]["Recall"] - results[0]["Recall"]) + abs(results[1]["Precision"] - results[0]["Precision"])
delta_svc = abs(results[3]["Recall"] - results[2]["Recall"]) + abs(results[3]["Precision"] - results[2]["Precision"])

print("\n=== NHAN XET ===")
print(f"- Do bien doi tong cong (Precision + Recall) cua Random Forest: {delta_rf:.4f}")
print(f"- Do bien doi tong cong (Precision + Recall) cua SVC          : {delta_svc:.4f}")
if delta_svc > delta_rf:
    print("- Mo hinh SVC thay doi nhieu nhat khi them class_weight='balanced':")
    print("  Recall tang len ro ret do ranh gioi quyet dinh bi day xa hon ve phia lop da so.")
else:
    print("- Mo hinh Random Forest thay doi nhieu nhat khi them class_weight='balanced'.")
print("- Khi can bang lop, ca hai mo hinh deu co xu huong tang recall (it bo sot lop bo hoc hon)")
print("  nhung phai danh doi bang precision giam (so bao dong nham tang).")
