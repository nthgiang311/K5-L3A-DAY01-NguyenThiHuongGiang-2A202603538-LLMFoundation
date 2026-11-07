# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603538
- **Họ và tên:** Nguyễn Thị Hương Giang
- **SEED** (in ở ô đầu notebook): 6256

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | có |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | có |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | có |

## 2. Kết quả chính

*Chép bảng so sánh cuối Phần 5 (F1 test, F1 SUB 40, p95 ms/tin, USD/1k tin) của chính bạn.*
                                 precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            0.988   0.847      0.912        0.919   
2. BoW + MLP (deep learning)         0.989   0.918      0.952        0.974   
3a. Encoder (MiniLM) + kNN           0.931   0.969      0.950        1.000   
3b. Encoder (MiniLM) + LogReg        0.979   0.959      0.969        0.974   
3c. Encoder (MiniLM) + MLP head      0.949   0.949      0.949        0.974   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            2.360      76858.226       0.026   
2. BoW + MLP (deep learning)         5.334      56293.906       0.153   
3a. Encoder (MiniLM) + kNN          60.384         37.046      40.441   
3b. Encoder (MiniLM) + LogReg       60.555         47.087      19.657   
3c. Encoder (MiniLM) + MLP head     61.062         36.985      22.052   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.046           0.0  
2. BoW + MLP (deep learning)          0.222           0.0  
3a. Encoder (MiniLM) + kNN           81.660           0.0  
3b. Encoder (MiniLM) + LogReg        27.165           0.0  
3c. Encoder (MiniLM) + MLP head      31.740           0.0  

## 3. Dự đoán và thực tế

*Chọn 3 dự đoán `predict(...)` lệch nhiều nhất so với kết quả. Mỗi cái: bạn đoán gì, ra gì, vì sao lệch.*

## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*

## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*

## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*

## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
ML: Giải thích Logistic Regression, Sigmoid và Metrics. Giải thích công thức và cơ chế học
DL: Giải thích MLP, ReLU và Overfitting. Phân tích cấu trúc mạng và quá trình học. Kiểm tra số tham số và loss thực nghiệm
Embedding: Giải thích Cosine Similarity và kNN. Giải thích cách biểu diễn ngữ nghĩa và tìm hàng xóm
LLM: Kiểm tra prompt, API và hàm parse label. Rà soát code, chưa xác nhận kết quả API nếu chưa chạy
Phân tích: Kiểm tra lỗi phương pháp và cách diễn giải kết quả. Chỉ ra các điểm cần chú ý trong đánh giá

