# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Đình Tuấn Anh  **MSSV**: 2A202602735  **Ngày**: 2026-10-07
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16GB`

Số liệu dưới đây lấy từ các artefact trong `results/`. Các giới hạn về đánh giá định tính được nêu rõ ở mục 6.

---

## 1. Setup

| Thông số | Giá trị |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage (4 keys) |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024 theo cấu hình tier T4; p95 đo được là 98 và `token_stats.json` gợi ý 256 |
| `MASK_MODE` | assistant-only |
| Epochs / max_steps | 30 steps |

Tôi chọn `unsloth/Qwen3.5-4B` vì tier T4 của lab cấu hình model này để huấn luyện LoRA trên GPU T4 16 GB; số đo peak VRAM của run chính là 8.78 GB. Bộ dữ liệu 250 ticket CSKH tiếng Việt phù hợp với tác vụ phân loại JSON bốn trường: nhãn từng trường cho phép chấm target và format bằng quy tắc cố định, không phụ thuộc LLM judge. Giới hạn của lựa chọn này là corpus nhỏ và được tạo theo khuôn; kết quả chưa chứng minh khả năng dùng trên ticket thật.

`max_length=1024` là giá trị mặc định của tier, **không phải** giá trị suy ra từ p95. Với p95=98, giá trị gợi ý là 256; cấu hình hiện tại cao hơn cần thiết cho corpus đã đo. Đây là điểm chưa tối ưu trong thí nghiệm và cần điều chỉnh, chạy lại các bước liên quan nếu muốn khẳng định `max_length` đã được đặt theo p95.

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

## 3. Mốc NB2 đóng băng trước khi train và kết quả NB5

NB2 lưu mốc (a), (b) trên 50 ticket target và 15 câu regression, không dùng `EVAL_LIMIT` (`smoke_mode=false`). SHA-256 rút gọn của prompt (b) là `719e74d3b6232053` trong `baselines_frozen.json`; checksum tập đánh giá nằm trong `data/checksums.json`. Dòng (c) được đo sau huấn luyện ở NB5.

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.791 | 0.000 | 3420.5 |
| (b) base + optimized prompt | 0.765 | 0.791 | 1.000 | 1057.2 |
| (c) LoRA fine-tune | 0.965 | 0.5444 | 1.000 | 1452.0 |

**(b) có thật sự mạnh hơn (a) không?** Có — baseline (b) dùng prompt tối ưu có mô tả schema, ràng buộc enum và hướng dẫn rõ ràng đã đạt độ chính xác target 76.5% và tuân thủ format 100%, vượt trội hoàn toàn so với naive prompt (0.0%) vốn thất bại do xuất văn bản tự do không theo cấu trúc JSON. Prompt (b) được giữ nguyên vẹn (`OPTIMIZED_PROMPT` không bị sửa đổi) để đảm bảo chuẩn mực so sánh công bằng và liêm chính.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 0.0001 | 0.6282 | 0.965 | 421.8 | 8.78 |
| `attn_only` | attn-only | 283 | 32,456,704 | 0.0001 | 0.5374 | 0.97 | 286.4 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-05 | 1.5702 | 0.0 | 433.2 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 0.0001 | 0.7058 | 0.94 | 502.9 | 3.86 |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế chính là Lỗi #3.

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?**
Run `attn_only` được cấu hình với rank cao ($r=283$) để khớp ngân sách tham số huấn luyện của `correct` (32,456,704 so với 32,464,896; chênh khoảng 0.025%). Train loss của `attn_only` thấp hơn (`0.5374` so với `0.6282`) và điểm target cũng nhỉnh hơn (`0.970` so với `0.965`, tức 0.5 điểm phần trăm). Trên tác vụ và tập đánh giá này, không có bằng chứng rằng vị trí `text-linear` tốt hơn `attn-only`; tăng rank ở attention với cùng ngân sách vẫn có thể đạt điểm cao. Chênh lệch target chỉ tương đương một trường đúng trên 50 mẫu × 4 trường, nên không nên suy rộng thành kết luận chắc chắn cho tác vụ khác. Phép so sánh này tách được ảnh hưởng của *vị trí* khỏi tổng số tham số, nhưng thay vị trí đồng thời buộc thay rank để giữ ngân sách; cần thêm thí nghiệm để tách riêng hai yếu tố.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**
Cấu hình `wrong_lr` sử dụng learning rate $10^{-5}$ thay vì $10^{-4}$ của run chính. File `runs.csv` chỉ lưu **loss cuối**: `1.5702` so với `0.6282`; nó không lưu đường loss từng bước, nên tôi không thể kết luận đường loss đi ngang. Trên tập target, `wrong_lr` đạt `0.000` và format `0.000`. Nếu chỉ nhìn loss cuối mà không biết learning rate khác nhau, có thể đổ lỗi nhầm cho dữ liệu hoặc năng lực model. Trong ngân sách 30 bước đã đo, learning rate thấp hơn đi kèm khả năng học định dạng kém hơn rõ rệt; cần chạy thêm seed để khẳng định mức độ ổn định của kết quả.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**
Cấu hình `qlora` với lượng tử hóa 4-bit giảm peak VRAM từ `8.78` xuống `3.86` GB (khoảng 56.0%). Thời gian train tăng từ `421.8` lên `502.9` giây (khoảng 19.2%); điểm target giảm từ `0.965` xuống `0.940` (2.5 điểm phần trăm), và latency đánh giá tăng từ `1452.0` lên `1832.6` ms/mẫu. Trên T4 16 GB, run LoRA thường vẫn vừa bộ nhớ và đạt điểm cao hơn trong phép đo này; QLoRA có ích khi giới hạn VRAM là ràng buộc chính. Các con số cho thấy tương quan trong một lần chạy, chưa đủ để quy toàn bộ chênh lệch thời gian hay độ chính xác cho riêng bước dequantize.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.200` · `regression Δ = -0.247` · `valid_trace_rate = 0.00`

**Diễn giải chi tiết:**
Cổng hồi quy đưa ra phán quyết `FAILED`. Điểm target tăng từ `0.765` lên `0.965`, tức **20 điểm phần trăm**; format giữ ở `1.000`. Nhưng regression giảm từ `0.7911` xuống `0.5444`, tức giảm khoảng **24.67 điểm phần trăm**, vượt xa ngưỡng cho phép 2 điểm phần trăm ghi trong `verdict.json`. Latency tăng từ `1057.2` lên `1452.0` ms/mẫu. Vì vậy không thể nói bản fine-tune đã giữ nguyên năng lực tổng quát, cũng chưa thể đề xuất triển khai chỉ dựa trên điểm target cao. Bước tiếp theo cần thử là trộn một lượng nhỏ dữ liệu phổ thông vào train rồi đánh giá lại trên cùng tập đã đóng băng, giữ nguyên cổng hồi quy.

---

## 6. Định tính — ca đúng và ca còn sai

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt đèn bàn LED mã đơn VN339109. Vỡ khi nhận. Gấp. | san_pham_loi, cao, đèn bàn LED, trung_tinh | Chưa lưu dự đoán | Đúng 4/4 | FT đúng cả bốn trường. |
| 2 | Xin chào, mình đặt balo laptop mã đơn DH863123. Đổi size. Hỏi cho biết | doi_tra, thap, balo laptop, tieu_cuc | Chưa lưu dự đoán | Đúng 4/4 | FT đúng cả bốn trường. |
| 3 | Alo shop, mình đặt máy xay sinh tố mã đơn OD126693. Muốn đổi. Đã 3 ngày | doi_tra, trung_binh, máy xay sinh tố, tieu_cuc | Chưa lưu dự đoán | Đúng 4/4 | FT đúng cả bốn trường. |
| 4 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. | hoan_tien, thap, bình giữ nhiệt, tich_cuc | Chưa lưu dự đoán | Đúng 3/4 | FT dự đoán `urgency=trung_binh` thay vì `thap`. |
| 5 | Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện. | san_pham_loi, thap, nồi chiên không dầu, trung_tinh | Chưa lưu dự đoán | Đúng 3/4 | FT dự đoán `urgency=trung_binh` thay vì `thap`. |

**Giới hạn bằng chứng:** `qualitative.json` chỉ lưu `ft_score` và đoạn đầu `ft_pred`, không lưu dự đoán hoặc điểm theo từng mẫu của baseline (b). Vì vậy năm ca trên minh họa FT đúng/sai, **chưa chứng minh được ca nào FT thắng hoặc thua baseline (b)**. Hai ca 0.75 không được gọi là “FT thua”. Muốn đáp ứng mục 3.4 của rubric, cần chạy lại phần suy luận định tính, lưu nhãn đúng, dự đoán đầy đủ và điểm theo từng mẫu của cả (b) lẫn (c), rồi chọn ít nhất hai mẫu có `ft_score < baseline_b_score`. Không được suy diễn điểm baseline từ điểm tổng hợp.

---

## 7. Kết luận & điều tôi học được

**Kết luận:**
Bản LoRA tăng điểm phân loại target từ 0.765 lên 0.965 và giữ tỷ lệ xuất JSON đúng ở 1.000 khi so với base model đã được prompt tối ưu. Điều đó cho thấy adapter học được tác vụ hẹp trên tập ticket này. Tuy nhiên, điểm regression giảm từ 0.7911 xuống 0.5444, vượt ngưỡng cho phép của cổng đánh giá, nên phán quyết FAIL là hợp lý và tôi chưa xem bản fine-tune này là ứng viên triển khai. Kết quả cũng cho thấy không thể dùng riêng train loss để xếp hạng: run attention-only có loss thấp hơn *và* nhỉnh hơn 0.005 điểm target, trái với kỳ vọng rằng gắn adapter ở nhiều vị trí tuyến tính luôn thắng. Ngược lại, learning rate thấp hơn mười lần đi kèm điểm target bằng 0 trong cùng ngân sách 30 step; điều này gợi ý thời gian thích nghi của adapter là một biến cần kiểm soát. QLoRA giảm bộ nhớ hơn một nửa nhưng chạy lâu hơn và điểm target thấp hơn trong lần đo này. Để cải thiện thí nghiệm, tôi sẽ thử thêm dữ liệu replay phổ thông nhằm giảm regression, đo lại cả bốn nhóm trên mốc đã đóng băng, và lưu dự đoán từng mẫu của hai cấu hình cần so sánh. Kết luận hiện tại chỉ áp dụng cho corpus, model, cấu hình và một lần chạy đã ghi; nó không chứng minh ưu thế chung của LoRA hay khả năng hoạt động trên ticket thật.

**Ba điều tôi học được:**
1. **Loss mask là ranh giới sống còn**: Huấn luyện cả trên prompt (`MASK_MODE=everything`) sẽ làm ô nhiễm trọng số và khiến mô hình sinh lặp câu hỏi; kiểm chứng loss mask bằng giải mã ngược token là bước bắt buộc trước khi train.
2. **Cần đo vị trí và rank trên cùng ngân sách**: Ở phép đo này `attn_only` r=283 nhỉnh hơn `text-linear` r=16 đúng 0.005 điểm target; không thể khẳng định vị trí nào luôn tốt hơn từ một tập eval nhỏ.
3. **Đánh giá bằng tác vụ thật, không tin vào train loss**: Train loss thấp không đồng nghĩa với hiệu năng suy luận tốt; việc đóng băng mốc baseline và đo lường đa chiều (target, regression, format, latency) là tiêu chuẩn vàng để tránh sai lầm nghiệp vụ.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
Tôi sẽ thực hiện quét rank có kiểm soát ($r \in \{8, 16, 32, 64\}$) trên toàn bộ linear layers để tìm điểm bão hòa tối ưu, đồng thời chạy notebook NB6 để merge adapter vào base model và xuất định dạng GGUF để đo đạc độ trễ suy luận trên CPU/Edge device.
