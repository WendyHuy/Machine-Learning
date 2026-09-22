# Machine Learning: Hồi quy tuyến tính

Bài thực hành dùng dữ liệu giá của 60 căn hộ. Thư mục bài học nằm tại
`MayHoc/bai01_hoi_quy/`:

```text
bai01_hoi_quy/
├── data/gia_nha.csv
├── code/b1_doc_du_lieu.py ... b7_nhieu_bien.py
├── figures/    # hình minh họa
├── outputs/    # ảnh chụp kết quả
└── scripts/    # tiện ích chạy lại
```

Môi trường ảo `.venv` nằm ở gốc repository trên máy cá nhân và không được đưa
lên GitHub. Để tạo lại môi trường với Python 3.11 trở lên, chạy ở gốc repository:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Chạy các bài từ thư mục gốc của bài thực hành, vì các file Python đọc
`data/gia_nha.csv` theo thư mục hiện tại:

```powershell
cd MayHoc\bai01_hoi_quy
& '..\..\.venv\Scripts\python.exe' code\b1_doc_du_lieu.py
```

Thay `b1_doc_du_lieu.py` bằng tên file từ `b2` đến `b7` để chạy bước tương ứng.
