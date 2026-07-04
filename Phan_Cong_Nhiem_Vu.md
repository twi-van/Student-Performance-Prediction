# BẢNG PHÂN CÔNG NHIỆM VỤ DỰ ÁN GIỮA KỲ (MACHINE LEARNING)

**Tên dự án:** Student Performance Prediction (Dự báo Kết quả Học tập của Học sinh)
**Dataset:** Student Performance Dataset (UCI Machine Learning Repository)
**Số lượng thành viên:** 3 người

---

## 👨‍💻 THÀNH VIÊN 1: DATA & MODELING (Mô hình Cốt lõi)
**Phụ trách:** Bài 1 trong đề (5 điểm)

**Nhiệm vụ chi tiết:**
- [ ] Tải dataset "Student Performance" từ UCI.
- [ ] Viết code tiền xử lý dữ liệu (Xóa missing values, mã hóa Label Encoding/One-Hot cho các trường dữ liệu dạng chữ, chuẩn hóa dữ liệu Scaler).
- [ ] Lựa chọn bài toán Phân loại (Classification) (VD: xếp loại Giỏi/Khá/TB hoặc Đậu/Rớt).
- [ ] Xây dựng và huấn luyện ít nhất **3 mô hình học máy** khác nhau (Gợi ý: Logistic Regression, Random Forest, SVM hoặc Decision Tree).
- [ ] Tính toán và xuất kết quả các độ đo: `accuracy`, `precision`, `recall`, `f1-score` của từng class và `weighted average f1-score`.
- [ ] Ghi nhận thời gian huấn luyện (training time) và thời gian dự đoán (testing time) của từng mô hình.
- [ ] Lưu mô hình tốt nhất lại dưới dạng file `.pkl` để Thành viên 3 tích hợp vào Web.

---

## 📊 THÀNH VIÊN 2: FEATURE SELECTION & REPORTING (Phân tích & Báo cáo)
**Phụ trách:** Bài 2 trong đề (3 điểm) + Thiết kế Slide Thuyết trình

**Nhiệm vụ chi tiết:**
- [ ] Dùng phương pháp Correlation tính toán độ tương quan giữa các đặc trưng (features).
- [ ] Vẽ đồ thị minh họa Ma trận tương quan (Correlation Heatmap).
- [ ] Chọn ra các tập features khác nhau dựa trên mức độ tương quan cao/thấp.
- [ ] Xây dựng mô hình **Linear Regression** (Hồi quy tuyến tính) và chạy trên các tập features vừa chọn.
- [ ] So sánh hiệu năng các tập features thông qua độ đo **Mean Absolute Error (MAE)**.
- [ ] **Báo Cáo (Canva/PPT):** Thiết kế slide tổng hợp theo yêu cầu đề bài:
  - Tên dataset + danh sách nhóm.
  - Nguồn gốc bộ dữ liệu, ý nghĩa các trường dữ liệu.
  - Sơ lược các mô hình đã dùng (lấy thông tin từ Thành viên 1).
  - Kết quả so sánh Bài 1 (lấy thông tin từ Thành viên 1).
  - Trình bày phương pháp chọn đặc trưng và kết quả của Bài 2.

---

## 🖥️ THÀNH VIÊN 3: APPLICATION & INTEGRATION (Ứng dụng Web & Tích hợp)
**Phụ trách:** Bài 3 trong đề (2 điểm) + Tổng hợp Nộp Bài

**Nhiệm vụ chi tiết:**
- [ ] Khởi tạo môi trường ảo (Virtual Environment - `env`).
- [ ] Thiết lập file `requirements.txt` bao gồm các thư viện bắt buộc (Python > 3.10, `streamlit`, `scikit-learn`, `plotly`, `pandas`, `numpy`).
- [ ] Viết ứng dụng giao diện bằng **Streamlit**.
- [ ] Giao diện gồm các phần:
  - Cho phép người dùng chọn/nhập các thông tin của 1 học sinh (VD: tuổi, thời gian học, hỗ trợ giáo dục,...).
  - Hiển thị đồ thị tương quan đẹp mắt bằng thư viện `plotly` (lấy code vẽ từ Thành viên 2).
  - Nút "Dự đoán" gọi mô hình `.pkl` của Thành viên 1 để hiển thị kết quả đậu/rớt.
- [ ] Gom toàn bộ source code (`app.py`, file dataset, `requirements.txt`, slide báo cáo) thành 1 thư mục gọn gàng.
- [ ] Nén thành file `MSSV1_MSSV2_MSSV3.zip` và nộp bài.

---
> **Lưu ý chung:** Các thành viên có thể dùng Google Colab để nháp code và chia sẻ cho nhau, sau đó Thành viên 3 sẽ là người gom toàn bộ code đưa vào chạy trên máy tính cá nhân.
