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

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
|---|---|---|---|
| 4.1 | Dùng Gemini, nhờ bổ sung hàm `call_openai` | Viết thân hàm: import `OpenAI` trong hàm, tạo client, gọi `chat.completions.create` kèm `**provider_kwargs(max_tokens)`, đo latency bằng `time.perf_counter()`, trả `(text, latency)`; thêm `or ""` cho trường hợp `content` là `None`. Hướng dẫn đặt `LLM_PROVIDER = "gemini"`, thêm secret `GEMINI_API_KEY`, giữ `max_tokens=512` |chạy ô mock test `4.1 + 4.4`, chạy ô "Say hello", kiểm tra tên model bằng `list_models()` |
| 4.2 | Nhờ viết `build_classification_prompt` | Viết prompt gồm mô tả nhiệm vụ, yêu cầu trả lời đúng một từ, chèn ví dụ few-shot cùng định dạng, kết thúc bằng `Message: """<text>"""\nLabel:` | chạy test `4.2–4.3`, in thử prompt few-shot |
| 4.3 | Nhờ viết `parse_label` | Chuẩn hoá chuỗi, bỏ dấu câu, xử lý tiền tố `label`, kiểm tra `not spam` và `ham` **trước** `spam`, dùng `startswith`, trả `None` nếu không rõ | tự thử `"Not spam"`, `"I am not sure"` |
| 4.4 | Nhờ viết `llm_classify_batch` | Viết vòng lặp qua từng tin: tạo prompt bằng `prompt_fn`, gọi `call_llm` qua `retry_with_backoff`, parse nhãn (None thành 0 và đếm `n_invalid`), ước tính token bằng `số_từ / 0.75`, tính USD theo `PRICE_PER_1M` | chạy test mock, kiểm tra `preds`, `usd > 0`, `len(latencies)` |

>Trước đây tôi nghĩ gọi LLM chỉ là gửi prompt rồi đọc kết quả. Tôi hiểu ra rằng phần khó nằm ở các chi tiết xung quanh: tham số riêng của từng provider (model suy luận cần max_tokens lớn), việc parse câu trả lời phải xử lý các trường hợp như “not spam”, và chi phí được tính theo token nên prompt few-shot làm mỗi tin đắt hơn.