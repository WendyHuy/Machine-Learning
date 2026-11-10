# -*- coding: utf-8 -*-
"""Bài tập 2: Đánh giá mô hình SVM nhận dạng chữ số viết tay.

In classification_report cho cả 10 chữ số.
Tìm chữ số có recall thấp nhất, precision thấp nhất.
Giải thích vì sao macro avg và weighted avg gần như bằng nhau.
"""

from sklearn.datasets import load_digits
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Nạp dữ liệu chữ số viết tay
X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

# Khớp mô hình SVM kernel RBF (chuẩn hóa trước)
svm = make_pipeline(StandardScaler(), SVC(random_state=0))
svm.fit(X_tr, y_tr)
du = svm.predict(X_te)

print("=== CLASSIFICATION REPORT CHO 10 CHU SO ===")
bc = classification_report(y_te, du, digits=4, output_dict=True)
print(classification_report(y_te, du, digits=4))

# Tìm chữ số có recall thấp nhất và precision thấp nhất trong các lớp 0..9
recalls = {str(i): bc[str(i)]["recall"] for i in range(10)}
precisions = {str(i): bc[str(i)]["precision"] for i in range(10)}

min_rec_digit = min(recalls, key=recalls.get)
min_prec_digit = min(precisions, key=precisions.get)

print("=== NHAN XET VA GIAI THICH ===")
print(f"- Chu so co recall thap nhat   : chu so '{min_rec_digit}' (recall = {recalls[min_rec_digit]:.4f})")
print(f"- Chu so co precision thap nhat: chu so '{min_prec_digit}' (precision = {precisions[min_prec_digit]:.4f})")
print("- Macro avg va weighted avg gan nhu bang nhau vi:")
print("  Bo du lieu load_digits duoc phan bo rat deu giua cac chu so (moi lop co khoang 178-182 mau),")
print("  va nho stratify=y, tap kiem tra co so luong mau (support) o moi lop gan nhu tuong duong nhau (~45 mau/lop).")
print("  Macro avg tinh trung binh cong khong trong so, con weighted avg tinh trung binh co trong so theo support.")
print("  Khi support cac lop deu nhau thi hai gia tri trung binh nay tro nen gan nhu trung hop.")
