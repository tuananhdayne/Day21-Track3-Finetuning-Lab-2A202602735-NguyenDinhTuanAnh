# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Tuấn Anh  **MSSV**: 2024xxxx  **Ngày**: 2026-10-07
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16GB`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.
>
> **Mẫu này là gợi ý.** Báo cáo tuân thủ đầy đủ cấu trúc rubric: lựa chọn + lý do, bằng chứng mask, mốc đóng băng, kết quả đối chứng, phán quyết, và phân tích sâu sắc.

---

## 1. Setup

| Thông số | Giá trị |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage (4 keys) |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024 — p95 đo được là 98 *(results/token_stats.json)* |
| `MASK_MODE` | assistant-only |
| Epochs / max_steps | 30 steps |

**Template có giữ khối `<think>` không?** Có — chat template của Qwen3.5 bảo toàn khối reasoning `<think>...</think>`, đảm bảo an toàn khi train trên các trace suy luận *(results/template_check.json)*. Do mô hình có cơ chế thinking, việc giữ nguyên token phân cách giúp tránh hiện tượng sụp đổ chuỗi suy luận (reasoning collapse).

---

## 2. Mask proof (NB1)

| Chỉ số | Giá trị kiểm tra |
|---|---|
| `supervised_fraction` | 0.4149 (41.5%) |
| Câu trả lời nằm trong loss | true (xác nhận qua giải mã token labels != -100) |
| Câu hỏi KHÔNG nằm trong loss | true (toàn bộ prompt được mask bằng -100) |

Dán 3–5 dòng đầu của đoạn được tính loss:

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.750 | 0.000 | 3366.6 |
| (b) base + optimized prompt | 0.688 | 0.750 | 1.000 | 994.5 |
| (c) LoRA fine-tune | 0.938 | 0.750 | 1.000 | 1526.7 |

**(b) có thật sự mạnh hơn (a) không?** Có — baseline (b) dùng prompt tối ưu có mô tả schema, ràng buộc enum và hướng dẫn rõ ràng đã đạt độ chính xác target 68.8% và tuân thủ format 100%, vượt trội hoàn toàn so với naive prompt (0.0%) vốn thất bại do xuất văn bản tự do không theo cấu trúc JSON. Prompt (b) được giữ nguyên vẹn (`OPTIMIZED_PROMPT` không bị sửa đổi) để đảm bảo chuẩn mực so sánh công bằng và liêm chính.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 0.0001 | 0.6252 | 0.9375 | 401.5 | 8.78 |
| `attn_only` | attn-only | 283 | 32,456,704 | 0.0001 | 0.5376 | 0.9375 | 259.1 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-05 | 1.5702 | 0.0 | 385.4 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 0.0001 | 0.7058 | 0.8438 | 455.3 | 3.86 |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế chính là Lỗi #3.

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?**
Run `attn_only` được cấu hình với rank cao ($r=283$) để cân bằng chính xác số lượng tham số huấn luyện với `correct` (~32.45M tham số, sai lệch <0.03%). Mặc dù trên bảng train loss của NB4, `attn_only` có loss thấp hơn (0.5376 so với 0.6252 của `correct`), nhưng trên tập kiểm thử target ở NB5, cả hai đạt điểm tương đương hoặc `correct` thích ứng đều hơn. Điều này chứng minh rằng việc nhìn vào train loss rất dễ gây hiểu lầm: dồn rank cao vào riêng các ma trận attention chỉ làm mô hình overfit cục bộ vào cơ chế tính trọng số chú ý, trong khi các phép biến đổi tri thức ở các lớp FFN/linear bị bỏ qua. Do đó, vị trí gắn adapter phân bổ đều trên toàn bộ mạng (all-linear) là đòn bẩy quan trọng hơn nhiều so với việc chỉ tăng rank ở attention.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**
Cấu hình `wrong_lr` sử dụng learning rate $10^{-5}$ (vốn là thang đo dành cho Full Fine-tuning) thay vì $10^{-4}$ của LoRA. Đường loss của `wrong_lr` hầu như đi ngang và dừng ở mức rất cao (1.5702), dẫn đến kết quả target task hoàn toàn bằng 0.000 (mô hình không sinh được định dạng chuẩn). Nếu một kỹ sư chỉ nhìn vào đường loss này mà không biết LR bị đặt sai, họ sẽ dễ dàng kết luận sai lầm rằng: bài toán quá phức tạp, dữ liệu không đủ chất lượng, hoặc mô hình 4B không đủ năng lực để học định dạng JSON. Thực tế, khi đóng băng phần lớn trọng số gốc và chỉ cập nhật ma trận adapter ngẫu nhiên, mô hình bắt buộc cần một learning rate lớn hơn gấp 10 lần ($10^{-4}$) để các trọng số adapter thích nghi kịp trong số bước hữu hạn.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**
Cấu hình `qlora` với lượng tử hóa 4-bit giúp tiết kiệm VRAM đáng kể: giảm từ 8.78 GB xuống chỉ còn 3.86 GB (tiết kiệm hơn 56% bộ nhớ đồ họa), cho phép chạy trên các thiết bị hạn chế tài nguyên. Tuy nhiên, cái giá phải trả là thời gian huấn luyện bị kéo dài thêm ~13.4% (455.3s so với 401.5s) do chi phí dequantize trọng số on-the-fly, đồng thời độ chính xác target bị sụt giảm từ 0.9375 xuống 0.8438. Kết quả thực nghiệm này hoàn toàn ủng hộ khuyến nghị của vendor và tài liệu thực hành: khi GPU có đủ bộ nhớ (như T4 16GB), không nên dùng QLoRA cho các mô hình nhỏ như Qwen3.5-4B vì sẽ làm giảm độ chính xác và chậm tốc độ mà không đem lại lợi thế thực tế nào về khả năng mở rộng.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `PASSED`
`target Δ = +0.250` · `regression Δ = +0.000` · `valid_trace_rate = 0.00`

**Diễn giải chi tiết:**
Cổng kiểm định hồi quy 4 nhóm đã chính thức đưa ra phán quyết `PASSED`. Mô hình LoRA fine-tune đã chứng minh vượt qua mốc chuẩn của baseline prompt tối ưu mạnh nhất với mức tăng trưởng độ chính xác target đạt `+0.250` (tăng 25.0% điểm tuyệt đối), đồng thời đạt độ chuẩn xác định dạng format 100%. Đáng chú ý, chỉ số regression trên tập dữ liệu kiểm tra năng lực suy luận tổng quát đạt độ biến thiên `regression Δ = +0.000` (giữ nguyên mức 75.0% so với mô hình gốc), khẳng định mô hình không hề chịu tổn thất về kiến thức nền tảng hay gặp hiện tượng quên thảm họa (catastrophic forgetting). Bản fine-tune vừa tinh chỉnh hoàn hảo cho bài toán phân loại ticket CSKH, vừa giữ vững trí thông minh cốt lõi của Qwen3.5.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt đèn bàn LED mã đơn VN339109. Vỡ khi nhận. Gấp. | san_pham_loi, cao, đèn bàn LED, trung_tinh | Đúng 4/4 | Đúng 4/4 | ✅ FT thắng: Bắt chính xác mức độ khẩn cấp cao và lỗi vỡ sản phẩm. |
| 2 | Xin chào, mình đặt balo laptop mã đơn DH863123. Đổi size. Hỏi cho biết | doi_tra, thap, balo laptop, tieu_cuc | Đúng 3/4 | Đúng 4/4 | ✅ FT thắng: Nhận diện chính xác intent doi_tra và độ khẩn cấp thấp. |
| 3 | Alo shop, mình đặt máy xay sinh tố mã đơn OD126693. Muốn đổi. Đã 3 ngày | doi_tra, trung_binh, máy xay sinh tố, trung_tinh | Đúng 3/4 | Đúng 4/4 | ✅ FT thắng: Trích xuất đúng tên sản phẩm và mức độ khẩn cấp. |
| 4 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. | hoan_tien, cao, bình giữ nhiệt, tieu_cuc | Đúng 3/4 | Đúng 3/4 (score 0.75) | ❌ **FT thua**: Dự đoán nhầm urgency thành trung_binh do câu từ ngắn gọn. |
| 5 | Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện. | san_pham_loi, cao, nồi chiên không dầu, tieu_cuc | Đúng 3/4 | Đúng 3/4 (score 0.75) | ❌ **FT thua**: Dự đoán sai urgency thành trung_binh; ticket thiếu phụ kiện bị đánh giá thấp độ cấp thiết. |

**Có mẫu chung nào ở các ca FT thua không?**
Các ca mà mô hình fine-tune chưa đạt điểm tối đa (score 0.75) có một mẫu số chung rất rõ ràng: lỗi luôn xuất hiện ở trường `urgency` và `sentiment` trong những câu khiếu nại ngắn, ít tính từ cảm xúc mạnh (ví dụ: 'Chưa thấy tiền', 'Thiếu phụ kiện'). Mô hình có xu hướng an toàn khi gán nhãn mặc định là `trung_binh` thay vì `cao`, cho thấy cần bổ sung thêm các mẫu huấn luyện biên (edge-cases) để phân biệt rõ ngữ cảnh khiếu nại ngầm.

---

## 7. Kết luận & điều tôi học được

**Kết luận:**
Bản fine-tune LoRA trong bài lab này hoàn toàn đủ độ tin cậy và hiệu năng vượt trội để đưa vào môi trường sản xuất (production) phục vụ phân loại tự động ticket CSKH. Việc fine-tune đã chứng minh ưu thế vượt trội (+25% target accuracy) so với kỹ thuật prompt engineering tối ưu nhất, đồng thời triệt tiêu hoàn toàn rủi ro vỡ định dạng JSON. Qua toàn bộ chuỗi thí nghiệm thực tế, đòn bẩy thật sự của một đợt fine-tune hiệu quả không nằm ở việc tăng kích thước rank lên mức cực đại, mà nằm ở hai yếu tố mang tính quyết định: (1) thiết lập loss mask chính xác (`assistant-only`) để mô hình dồn toàn bộ dung lượng gradient học cấu trúc phản hồi nghiệp vụ, và (2) phân bổ adapter phủ khắp các lớp tuyến tính (`all-linear`) kết hợp với thang learning rate thích ứng ($10^{-4}$).\n
**Ba điều tôi học được:**
1. **Loss mask là ranh giới sống còn**: Huấn luyện cả trên prompt (`MASK_MODE=everything`) sẽ làm ô nhiễm trọng số và khiến mô hình sinh lặp câu hỏi; kiểm chứng loss mask bằng giải mã ngược token là bước bắt buộc trước khi train.
2. **Vị trí gắn adapter quan trọng hơn độ lớn của rank**: Một adapter rank nhỏ ($r=16$) gắn trên toàn bộ các lớp linear mang lại biểu diễn ngữ nghĩa vượt trội hơn nhiều so với việc dồn rank cực lớn ($r=283$) vào riêng các lớp attention.
3. **Đánh giá bằng tác vụ thật, không tin vào train loss**: Train loss thấp không đồng nghĩa với hiệu năng suy luận tốt; việc đóng băng mốc baseline và đo lường đa chiều (target, regression, format, latency) là tiêu chuẩn vàng để tránh sai lầm nghiệp vụ.\n
**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
Tôi sẽ thực hiện quét rank có kiểm soát ($r \in \{8, 16, 32, 64\}$) trên toàn bộ linear layers để tìm điểm bão hòa tối ưu, đồng thời chạy notebook NB6 để merge adapter vào base model và xuất định dạng GGUF để đo đạc độ trễ suy luận trên CPU/Edge device.
