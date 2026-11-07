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
ml_lr (Learning rate): Dự đoán rằng với lr=50 thì loss sẽ dao động mạnh do bước cập nhật quá lớn, nhưng biểu đồ thực tế cho thấy hàm loss giảm thẳng xuống gần 0 cực nhanh và đi ngang bằng phẳng (do dữ liệu BoW nhị phân và nhãn đơn giản khiến mô hình nhanh chóng bão hòa/hội tụ về cực tiểu thay vì dao động mạnh).   
dl_relu (Bỏ tầng kích hoạt ReLU): Dự đoán rằng F1 sẽ giảm hoặc chỉ ngang mức Logistic Regression, nhưng trên thực tế F1 trên tập test của mạng tuyến tính 2 lớp không có ReLU vẫn đạt 0.952 (bằng đúng mạng MLP có ReLU nhờ tối ưu hóa bằng Adam và cấu trúc tham số).   
dl_curve (Hiện tượng Overfitting ở Deep Learning): Dự đoán trong code là "không" (cho rằng val loss sẽ không giảm mãi), nhưng trên thực tế mô hình bị overfit từ rất sớm (ngay sau epoch 9, val loss bắt đầu tăng ngược trở lại trong khi train loss tiếp tục giảm sâu).

## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*
vẫn bị dài chỉ cần trả lời đúng câu hỏi là được
1. Model tụt nhiều nhất: BoW + MLP (F1 tụt từ 0.974 xuống 0.324; Acc: 50%) và BoW + LogReg (F1 từ 0.919 xuống 0.333; Acc: 52%).
Nhóm case tụt nặng nhất:
Spam giọng tự nhiên (natural_spam): BoW + MLP đúng 0%, BoW + LogReg đúng 10%.   
Né từ khoá (obfuscation): BoW + MLP đúng 0%, BoW + LogReg đúng 17%.   
2. 2 case cụ thể và cơ chế sai:
Case #29 (Né từ khoá):
Nội dung: "C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd. Cl4im n0w at w1n-gift.c0m"   
Thực tế: Spam | Đoán: Ham.   
Cơ chế: Kẻ tấn công thay chữ cái bằng số. Tokenizer coi đây là từ lạ ngoài từ điển, tạo vector toàn 0, không kích hoạt được trọng số spam nên mô hình đoán nhầm thành ham.   
Case #14 (Spam giọng tự nhiên):
Nội dung: "Hi, I'm working from home and making about £300 a day, happy to show you how, no experience needed, just text me back"   
Thực tế: Spam | Đoán: Ham.   
Cơ chế: Không chứa từ khóa spam lộ liễu mà chứa nhiều đại từ thông thường ("I", "me", "hi") — những từ mang trọng số âm kéo mạnh về ham, khiến mô hình bị lừa.   
## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*

## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*
Phương án chọn: BoW + Logistic Regression (hoặc BoW + MLP nếu ưu tiên F1 hơn nữa).
Dẫn số liệu thực nghiệm:
  Độ trễ: p95 đạt 0.05 ms/tin.   
  Thông lượng: 76.858 tin/s xử lý 10 triệu SMS chỉ mất khoảng 2,2 phút (với 10 triệu tin/ngày, thông lượng trung bình cần chỉ ~116 tin/s).   
  Hiệu năng, chi phí: F1 = 0.912 (test thường), chi phí API = 0 USD.   
  (So sánh: Các phương án Embedding có p95 từ 27 – 82 ms, thông lượng chỉ 37 – 47 tin/s nên mất ~60 giờ cho 10 triệu tin, không thể đáp ứng thời gian thực).   
Chấp nhận đánh đổi:Dễ bị lừa bởi biến thể: F1 tụt mạnh xuống 0.333 trên các mẫu bẫy né từ khóa hoặc spam tự nhiên do không hiểu ngữ cảnh ngữ nghĩa.   

## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
ML: Giải thích Logistic Regression, Sigmoid và Metrics. Giải thích công thức và cơ chế học
DL: Giải thích MLP, ReLU và Overfitting. Phân tích cấu trúc mạng và quá trình học. Kiểm tra số tham số và loss thực nghiệm
Embedding: Giải thích Cosine Similarity và kNN. Giải thích cách biểu diễn ngữ nghĩa và tìm hàng xóm
LLM: Kiểm tra prompt, API và hàm parse label. Rà soát code, chưa xác nhận kết quả API nếu chưa chạy
Phân tích: Kiểm tra lỗi phương pháp và cách diễn giải kết quả. Chỉ ra các điểm cần chú ý trong đánh giá
Điều nhận ra mà trước đó chưa nhận ra: em cảm thấy trước kia mình sử dụng AI quá vội vàng mà chưa kịp viết bài toán, liệt kê kế hoạch và đưa ra outline ý tưởng và thực hiện ý tưởng. Em nghĩ AI chỉ nên là công cụ support ý tưởng đã hình thành trên 60% để đảm bảo suy nghĩ và ý tưởng không bị AI thao túng.

