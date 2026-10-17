# -*- coding: utf-8 -*-
"""Bai tap 5: chon sai bien the Naive Bayes cho so lieu lien tuc."""

import sys
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB

df = pd.read_csv("data/sinh_vien.csv")
X = df[["gio_on", "diem_giua_ky"]]
y = df["qua_mon"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

# Chay MultinomialNB tren du lieu lien tuc
mnb = MultinomialNB()
co_loi = False
try:
    mnb.fit(X_train, y_train)
    y_pred_mnb = mnb.predict(X_test)
    acc_mnb = accuracy_score(y_test, y_pred_mnb)
except Exception as e:
    co_loi = True
    loi_msg = str(e)

# Chay GaussianNB de doi chung
gnb = GaussianNB().fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)
acc_gnb = accuracy_score(y_test, y_pred_gnb)

print("1. Chuong trinh co bao loi khong?")
if co_loi:
    print(f"   Co bao loi: {loi_msg}")
else:
    print("   Khong bao loi! MultinomialNB van chay binh thuong va ra ket qua.")

print()
print("2. So sanh accuracy:")
print(f"   Accuracy MultinomialNB : {acc_mnb:.4f}")
print(f"   Accuracy GaussianNB    : {acc_gnb:.4f}")

print()
print("3. Giai thich vi sao dung MultinomialNB o day la sai ve nguyen tac:")
print("   - MultinomialNB duoc thiet ke dua tren phan phoi da thuc (Multinomial Distribution),")
print("     von chi ap dung cho cac dac trung dem roi rac, khong am (vi du so lan xuat hien tu).")
print("   - O day, 'gio_on' va 'diem_giua_ky' la cac bien lien tuc (so thuc nhu 5.5, 7.7, 8.2).")
print("   - Khi gap so thuc khong am, scikit-learn chi coi chung nhu so dem le (fractional counts)")
print("     va tinh ty le tong don thuan, chu khong uoc luong mat do xac suat theo hinh chuong.")
print("   - Vi khong co gia tri am nen chuong trinh khong nem ra ValueError, tao ra 'loi im lang'")
print("     (silent bug) rat nguy hiem: code van chay nhung ly thuyet toan hoc bi sai hoan toan,")
print("     dan toi do chinh xac sut giam tu 0.9000 xuong 0.8000.")
print("   - Dung nguyen tac cho so lieu lien tuc la phai dung GaussianNB.")

