# -*- coding: utf-8 -*-
"""Bai tap 2: tu nao to cao tin rac va tin thuong manh nhat."""

import sys
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Doc du lieu 300 tin nhan
df = pd.read_csv("data/tin_nhan.csv")

vec = CountVectorizer(token_pattern=r"(?u)\S+")
X = vec.fit_transform(df["noi_dung"])
y = df["la_rac"]

# Huan luyen tren toan bo 300 tin nhan
mo_hinh = MultinomialNB()
mo_hinh.fit(X, y)

# Tinh hieu loga xac suat giua lop 1 (rac) va lop 0 (thuong)
# diff > 0: nghieng ve tin rac; diff < 0: nghieng ve tin thuong
hieu_log_prob = mo_hinh.feature_log_prob_[1] - mo_hinh.feature_log_prob_[0]
tu_vung = vec.get_feature_names_out()
do_lech = pd.Series(hieu_log_prob, index=tu_vung)

print("5 tu to cao tin rac manh nhat (hieu log prob lon nhat):")
top_rac = do_lech.sort_values(ascending=False).head(5)
for tu, val in top_rac.items():
    print(f"  {tu:10s} : {val:+.4f}")

print()
print("5 tu cho thay tin thuong manh nhat (hieu log prob nho nhat):")
top_thuong = do_lech.sort_values(ascending=True).head(5)
for tu, val in top_thuong.items():
    print(f"  {tu:10s} : {val:+.4f}")

