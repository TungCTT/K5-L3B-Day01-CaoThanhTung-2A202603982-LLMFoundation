# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:*2A202603982*
- **Họ và tên:*CAO THANH TÙNG*
- **SEED** (in ở ô đầu notebook): 6762

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | v |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | v |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | v |

## 2. Kết quả chính

                                precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            1.000   0.740      0.850        0.889   
2. BoW + MLP (deep learning)         0.989   0.927      0.957        0.974   
3a. Encoder (MiniLM) + kNN           0.875   0.948      0.910        1.000   
3b. Encoder (MiniLM) + LogReg        0.926   0.917      0.921        1.000   
3c. Encoder (MiniLM) + MLP head      0.947   0.938      0.942        1.000   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            5.626       2569.888       0.062   
2. BoW + MLP (deep learning)         5.008      24717.290       0.281   
3a. Encoder (MiniLM) + kNN          64.515         18.544      63.139   
3b. Encoder (MiniLM) + LogReg       65.819         34.649      24.226   
3c. Encoder (MiniLM) + MLP head     65.830         34.595      26.785   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             6.729           0.0  
2. BoW + MLP (deep learning)          0.431           0.0  
3a. Encoder (MiniLM) + kNN          116.481           0.0  
3b. Encoder (MiniLM) + LogReg        48.760           0.0  
3c. Encoder (MiniLM) + MLP head      45.976           0.0  
• Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
• SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
• Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
• train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.

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
