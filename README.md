# Học máy ứng dụng (Applied Machine Learning)

Kho lưu trữ toàn bộ mã nguồn thực hành, bài tập và kết quả của môn học **Học máy ứng dụng**. Toàn bộ các bài học được triển khai với Python, sử dụng các thư viện chuẩn như `NumPy`, `Pandas`, `Scikit-learn` và `Matplotlib`.

---

## 📂 Cấu trúc tổng thể kho lưu trữ

Toàn bộ các bài học và bài tập được đặt trong thư mục `MayHoc/`:

```text
Machine-Learning/
├── README.md
├── requirements.txt
└── MayHoc/
    ├── bai01_hoi_quy/            # Bài 1: Hồi quy tuyến tính (Linear Regression)
    ├── bai02_hoi_quy_logistic/   # Bài 2: Hồi quy Logistic (Logistic Regression)
    ├── bai03_Naive_Bayes/        # Bài 3: Bộ phân loại Naive Bayes
    ├── bai04_knn/                # Bài 4: K láng giềng gần nhất (K-Nearest Neighbors)
    ├── bai05_cay/                # Bài 5: Cây quyết định và Rừng ngẫu nhiên (Decision Tree & Random Forest)
    ├── bai06_svm/                # Bài 6: Máy véc-tơ hỗ trợ (Support Vector Machines - SVM)
    ├── bai07_danh_gia/           # Bài 7: Đánh giá mô hình học máy (Model Evaluation & Metrics)
    ├── bai08_kmeans/             # Bài 8: Phân cụm K-Means (K-Means Clustering)
    └── bai09_mlp/                # Bài 9: Mạng nơ-ron nhiều lớp (Multilayer Perceptron - MLP)
```

Mỗi thư mục bài học đều tuân thủ cấu trúc chuẩn:
- `data/`: Bộ dữ liệu CSV phục vụ học thực hành và làm bài tập.
- `code/`: Các tệp mã nguồn mẫu hướng dẫn từng bước theo tài liệu giảng dạy.
- `baitap0X/`: Các tệp mã nguồn bài tập về nhà (`bai1.py` ... `bai6.py`) có chú thích chi tiết và trả lời câu hỏi bằng tiếng Việt.
- `outputs/`: Ảnh chụp kết quả màn hình terminal thực thi của các tệp mã nguồn.
- `figures/`: Các biểu đồ trực quan hóa dữ liệu và kết quả huấn luyện mô hình (nếu có).

---

## 📑 Danh mục các bài học và bài tập

| Bài | Tên bài học | Nội dung lý thuyết & thực hành | Dữ liệu chính |
| :--- | :--- | :--- | :--- |
| **Lab 01** | **Hồi quy tuyến tính** | Khám phá dữ liệu, phương trình chính quy, hạ độ dốc (Gradient Descent), hồi quy đơn biến & đa biến, đánh giá MAE, MSE, RMSE, $R^2$. | `gia_nha.csv` |
| **Lab 02** | **Hồi quy Logistic** | Hàm Sigmoid, tối ưu hóa Binary Cross-Entropy, phân loại nhị phân, chuẩn hóa `StandardScaler`, ranh giới quyết định. | `sinh_vien.csv` |
| **Lab 03** | **Naive Bayes** | Phân loại văn bản spam/ham, định lý Bayes, GaussianNB, MultinomialNB, trích xuất đặc trưng `CountVectorizer`. | `tin_nhan.csv`, `sinh_vien.csv` |
| **Lab 04** | **K láng giềng gần nhất (KNN)** | Khoảng cách Euclidean, chuẩn hóa dữ liệu, chọn $K$ tối ưu, ảnh hưởng của nhiễu và chiều dữ liệu, KNN phân loại & hồi quy. | `hai_vang_trang.csv`, `sinh_vien_mo_rong.csv`, `gia_nha.csv` |
| **Lab 05** | **Cây quyết định & Rừng ngẫu nhiên** | Chỉ số Gini, độ sâu tối đa (`max_depth`), hiện tượng quá khớp (Overfitting), Bagging, Feature Importance, DecisionTreeRegressor & RandomForest. | `khach_vay.csv`, `gia_nha.csv` |
| **Lab 06** | **Máy véc-tơ hỗ trợ (SVM)** | Siêu phẳng phân chia, lề cực đại (Hard/Soft Margin), siêu tham số $C$, hạt nhân RBF và tham số $\gamma$, nhận dạng chữ số viết tay `load_digits`. | `hai_vang_trang.csv`, `sinh_vien.csv`, `load_digits` |
| **Lab 07** | **Đánh giá mô hình** | Hạn chế của Accuracy trên tập lệch lớp, ma trận nhầm lẫn (Confusion Matrix), Precision, Recall, $F_1$-score, điều chỉnh ngưỡng quyết định, đường cong ROC & AUC, kiểm định chéo phân tầng, cân bằng lớp `class_weight="balanced"`. | `sinh_vien_bo_hoc.csv`, `gia_nha.csv` |
| **Lab 08** | **Phân cụm K-Means** | Học không giám sát, tự cài đặt thuật toán K-Means, độ quán tính (Inertia), phương pháp khuỷu tay (Elbow), hệ số Silhouette, 3 cái bẫy K-Means (chuẩn hóa, khởi tạo, cụm phi cầu), nén màu ảnh 16 màu, so sánh DBSCAN. | `khach_hang.csv`, `hai_vang_trang.csv`, `china.jpg`, `flower.jpg` |
| **Lab 09** | **Mạng nơ-ron nhiều lớp (MLP)** | Tự tính lan truyền tiến (Forward Propagation), giải bài toán phi tuyến XOR, hàm kích hoạt (Sigmoid, Tanh, ReLU), điều chuẩn $L_2$ (`alpha`), đường cong mất mát (`loss_curve_`), tốc độ học (`learning_rate_init`), dò lưới `GridSearchCV`, MLPRegressor và hạn chế ngoại suy. | `xor.csv`, `hai_vang_trang.csv`, `sinh_vien_mo_rong.csv`, `gia_nha.csv`, `load_digits` |

---

## 🛠️ Cài đặt môi trường

Yêu cầu môi trường Python 3.10 trở lên. Khuyến nghị sử dụng môi trường ảo (`venv`):

```powershell
# 1. Khởi tạo môi trường ảo
python -m venv .venv

# 2. Kích hoạt môi trường ảo (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# 3. Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

Nội dung tệp `requirements.txt`:
```text
numpy
pandas
matplotlib
scikit-learn>=1.6.0
scipy
pillow
```

---

## 🚀 Hướng dẫn thực thi

Mỗi bài học được thiết kế để chạy từ **thư mục gốc của chính bài học đó** (để đường dẫn tương đối tới thư mục `data/` hoạt động chính xác):

### Ví dụ chạy Bài 07 (Đánh giá mô hình):
```powershell
# Chuyển vào thư mục bài học
cd MayHoc\bai07_danh_gia

# Chạy tệp lý thuyết/thực hành mẫu
..\..\.venv\Scripts\python.exe code\h1_do_bua.py

# Chạy bài tập về nhà
..\..\.venv\Scripts\python.exe baitap07\bai1.py
```

### Ví dụ chạy Bài 08 (Phân cụm K-Means):
```powershell
cd MayHoc\bai08_kmeans
..\..\.venv\Scripts\python.exe code\k2_khach_hang.py
..\..\.venv\Scripts\python.exe baitap08\bai4.py
```

### Ví dụ chạy Bài 09 (Mạng nơ-ron nhiều lớp MLP):
```powershell
cd MayHoc\bai09_mlp
..\..\.venv\Scripts\python.exe code\m2_xor.py
..\..\.venv\Scripts\python.exe baitap09\bai1.py
```

*(Thực hiện tương tự cho các bài học từ `bai01` đến `bai06`)*

---

## 👤 Thông tin tác giả
- **Tác giả:** [WendyHuy](https://github.com/WendyHuy)
- **Email:** `huy.2474802010135@vanlanguni.vn`
- **Kho lưu trữ:** [https://github.com/WendyHuy/Machine-Learning](https://github.com/WendyHuy/Machine-Learning)
