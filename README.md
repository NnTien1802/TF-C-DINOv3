# TF-C: Thay thế hai Encoder bằng DINOv3

**Dự án phát triển từ:** [Self-Supervised Contrastive Pre-Training for Time Series via Time-Frequency Consistency (TF-C)](https://github.com/mims-harvard/TFC-pretraining)  
**Nhóm tác giả gốc:** Xiang Zhang, Ziyuan Zhao, Theodoros Tsiligkaridis và Marinka Zitnik  
**Bài báo:** [NeurIPS 2022](https://openreview.net/forum?id=OJ4mMfGKLN) · [arXiv](https://arxiv.org/abs/2206.08496)  
**Backbone thay thế:** [Meta DINOv3 ViT-S/16](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)

> **Phạm vi cập nhật:** Thay hai khối Encoder trong `code/TFC/model.py` ở mức mã nguồn. Chưa huấn luyện, đánh giá hoặc xác nhận toàn bộ pipeline TF-C sau khi thay backbone.

## 1. Tổng quan

**TF-C (Time-Frequency Consistency)** là phương pháp học biểu diễn chuỗi thời gian bằng *self-supervised contrastive learning*. Mô hình gồm hai nhánh trích xuất đặc trưng: **miền thời gian (Time Domain)** và **miền tần số (Frequency Domain)**. Hai *projection heads* đưa đặc trưng về không gian chung để thực hiện học tương phản.

Trong phiên bản này, **cả Time Encoder và Frequency Encoder đều được thay bằng DINOv3 ViT-S/16 đã pre-train**. Các thành phần contrastive loss và downstream classifier của dự án gốc không được thiết kế lại.

## 2. Encoder trong dự án TF-C gốc

**Vị trí:** [`code/TFC/model.py`](code/TFC/model.py), class `TFC`.

Phiên bản mã nguồn TF-C được nhóm tác giả cập nhật tháng 01/2023 sử dụng **hai `TransformerEncoder` của PyTorch**, mỗi encoder gồm **2 Transformer layers**, **2 attention heads**. Trước bản cập nhật này, nhóm tác giả sử dụng CNN 3 tầng; việc thay backbone ở đây được thực hiện trên **phiên bản Transformer hiện có của repository**, không phải CNN trong thiết kế trước đó.

| Thành phần | Tên trong code gốc | Nhiệm vụ |
| --- | --- | --- |
| Time Encoder | `self.transformer_encoder_t` | Trích xuất đặc trưng từ chuỗi miền thời gian |
| Frequency Encoder | `self.transformer_encoder_f` | Trích xuất đặc trưng từ biểu diễn độ lớn FFT |
| Time Projector | `self.projector_t` | Chiếu đặc trưng thời gian sang embedding 128 chiều |
| Frequency Projector | `self.projector_f` | Chiếu đặc trưng tần số sang embedding 128 chiều |

**Code trước khi thay thế:**

```python
# Time Encoder — Transformer 2 layers
encoder_layers_t = TransformerEncoderLayer(
    configs.TSlength_aligned,
    dim_feedforward=2 * configs.TSlength_aligned,
    nhead=2,
)
self.transformer_encoder_t = TransformerEncoder(encoder_layers_t, 2)

# Frequency Encoder — Transformer 2 layers
encoder_layers_f = TransformerEncoderLayer(
    configs.TSlength_aligned,
    dim_feedforward=2 * configs.TSlength_aligned,
    nhead=2,
)
self.transformer_encoder_f = TransformerEncoder(encoder_layers_f, 2)
```

Các lớp Transformer được gọi từ thư viện `torch.nn`; phần kiến trúc kết hợp hai nhánh, projector và loss do dự án TF-C triển khai.

## 3. DINOv3 và lý do lựa chọn ViT-S/16

**DINOv3** là họ mô hình thị giác học biểu diễn theo phương pháp tự giám sát do **Meta AI** phát triển. Trong bản cập nhật này, hai encoder sử dụng checkpoint:

```text
facebook/dinov3-vits16-pretrain-lvd1689m
```

| Thông số | DINOv3 ViT-S/16 |
| --- | --- |
| Kiến trúc | Vision Transformer — Small (ViT-S) |
| Kích thước patch | 16 × 16 |
| Kích thước vector đặc trưng | 384 chiều |
| Trọng số | Pre-trained bởi Meta; nạp bằng `transformers.AutoModel` |
| Vai trò | Backbone cho cả hai nhánh thời gian và tần số |

**Lý do lựa chọn:**

1. **Có trọng số pre-trained:** dùng mô hình nền tảng làm bộ trích xuất đặc trưng thay vì khởi tạo Transformer mới từ đầu.
2. **Phiên bản Small phù hợp để thử nghiệm ban đầu:** nhẹ hơn các phiên bản ViT lớn trong họ DINOv3, phù hợp định hướng triển khai trên GPU RTX 4060 Laptop 8 GB.
3. **Dễ tích hợp vào PyTorch:** tải thông qua `AutoModel.from_pretrained(...)`, lấy vector đặc trưng từ CLS token và kết nối lại hai projector của TF-C.

> Việc lựa chọn ViT-S/16 nhằm giảm độ phức tạp khi tích hợp; **chưa có kết quả thực nghiệm chứng minh DINOv3 cải thiện hiệu năng TF-C** trên dữ liệu chuỗi thời gian.

## 4. Những vị trí code đã thay đổi

**File thay đổi duy nhất:** [`code/TFC/model.py`](code/TFC/model.py). Các liên kết bên dưới trỏ trực tiếp đến vị trí trong phiên bản `model.py` được cung cấp cùng README này.

| Đánh dấu | Vị trí | Nội dung cập nhật |
| --- | --- | --- |
| `[UPDATED 1]` | [Dòng 15–16](code/TFC/model.py#L15-L16) | Khai báo mã checkpoint DINOv3 ViT-S/16 |
| `[NEW]` | [Dòng 19–56](code/TFC/model.py#L19-L56) | Thêm `SignalToDINOImage` để chuyển tensor 1D sang đầu vào RGB của DINOv3 |
| **`[UPDATED 2]`** | **[Dòng 64–65](code/TFC/model.py#L64-L65)** | **Thay `transformer_encoder_t` bằng `dinov3_time`** |
| **`[UPDATED 3]`** | **[Dòng 67–69](code/TFC/model.py#L67-L69)** | **Thay `transformer_encoder_f` bằng `dinov3_freq`** |
| `[UPDATED 4]` | [Dòng 75–93](code/TFC/model.py#L75-L93) | Thay kích thước đầu vào hai projector theo `hidden_size` của DINOv3 (384) |
| `[UPDATED 5]` | [Dòng 95–100](code/TFC/model.py#L95-L100) | Cập nhật `forward()` của nhánh thời gian, lấy CLS token |
| `[UPDATED 6]` | [Dòng 102–106](code/TFC/model.py#L102-L106) | Cập nhật `forward()` của nhánh tần số, lấy CLS token |

### 4.1. Thay hai khối Encoder

**Code mới:**

```python
from transformers import AutoModel

DINOV3_MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"

# [UPDATED 2] Time Encoder
self.dinov3_time = AutoModel.from_pretrained(DINOV3_MODEL_ID)

# [UPDATED 3] Frequency Encoder
self.dinov3_freq = AutoModel.from_pretrained(DINOV3_MODEL_ID)
```

Hai nhánh khởi tạo từ **cùng một checkpoint pre-trained**, nhưng là **hai instance có bộ tham số độc lập**. File trọng số được cache để không phải tải hai lần từ Internet.

### 4.2. Điều chỉnh đầu vào và Projection Heads

TF-C gốc nhận chuỗi 1D dạng `[B, 1, L]`, còn DINOv3 ViT-S/16 cần ảnh RGB. Vì vậy, class [`SignalToDINOImage`](code/TFC/model.py#L19-L56) thực hiện chuẩn hóa theo từng mẫu, resize, nhân thành ba kênh và chuẩn hóa ảnh, để tạo tensor `[B, 3, 224, 224]`.

Sau backbone, đặc trưng được lấy từ **CLS token**, có kích thước **384 chiều**; hai projector được điều chỉnh từ `Linear(configs.TSlength_aligned, 256)` thành `Linear(384, 256)`, nhưng vẫn xuất ra **128 chiều**.

```python
# [UPDATED 5] Time branch
h_time = self.dinov3_time(pixel_values=time_image).last_hidden_state[:, 0, :]
z_time = self.projector_t(h_time)

# [UPDATED 6] Frequency branch
h_freq = self.dinov3_freq(pixel_values=freq_image).last_hidden_state[:, 0, :]
z_freq = self.projector_f(h_freq)

# Giữ nguyên giao diện đầu ra cho trainer.py
return h_time, z_time, h_freq, z_freq
```

**Các thành phần giữ nguyên:** cấu trúc hai projector ngoài kích thước đầu vào, `target_classifier`, `loss.py` và giao diện bốn tensor trả về của `TFC.forward()`.

> **Giới hạn kỹ thuật:** `SignalToDINOImage` hiện chỉ là bộ chuyển đổi *hình dạng tensor* để minh họa việc thay encoder. Đây **không phải GAF, spectrogram hay phép biến đổi thời gian–tần số được kiểm chứng**. Bản sửa code chưa được huấn luyện hay xác nhận hoạt động đầu cuối; đặc biệt, checkpoint Transformer cũ trong `main.py` không thể nạp trực tiếp cho kiến trúc DINOv3 mới.

## 5. Nguồn tham khảo

- **TF-C gốc:** [mims-harvard/TFC-pretraining](https://github.com/mims-harvard/TFC-pretraining), [bài báo NeurIPS 2022](https://openreview.net/forum?id=OJ4mMfGKLN).
- **DINOv3:** [Repository chính thức của Meta](https://github.com/facebookresearch/dinov3), [checkpoint ViT-S/16 trên Hugging Face](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m).

*Bản chỉnh sửa này phục vụ yêu cầu thay encoder/backbone trong mã nguồn TF-C; không phải phiên bản phát hành chính thức của nhóm tác giả TF-C hoặc Meta AI.*
