# 📖 Giải thích chi tiết dự án AUTO CROP theo ID

## 🎯 Mục đích dự án

Dự án này là một ứng dụng **tự động cắt ảnh chân dung từng người** từ các frame video đã được trích xuất sẵn (khoảng 60 frame/clip). Ứng dụng sử dụng **YOLO Pose** để phát hiện người và 17 điểm khớp cơ thể COCO, sau đó **tracking (bám theo)** từng người qua tất cả frame để tạo ra dataset ảnh riêng cho mỗi cá nhân.

Điểm đặc biệt: ứng dụng có **bộ nhớ ID theo ngoại hình** — dùng ResNet50 (ImageNet) + histogram màu áo để tự động gợi ý ID khi mở folder mới, giảm thao tác gán tay.

### Ứng dụng thực tế
- Tạo **dataset khuôn mặt/chân dung** cho từng người từ video giám sát
- Phục vụ cho **nhận dạng khuôn mặt**, huấn luyện model AI
- Xử lý hàng loạt nhiều video clip cùng lúc

---

## 🏗️ Kiến trúc dự án

```
test1/
├── main.py              # Điểm khởi chạy — tạo cửa sổ PyQt5 Fusion
├── config.py            # Cấu hình: model, ngưỡng phát hiện, tham số cắt
├── detection.py         # Xử lý ảnh: detect, khử trùng, crop, tracking, đánh số
├── gui.py               # Giao diện PyQt5 (1073 dòng)
├── gallery.py           # Bộ nhớ ID: ResNet50 + màu áo → gợi ý ID
├── gallery.json         # Vector đặc trưng đã lưu (tự động)
├── app_state.json       # Lưu phiên làm việc (tự động, bị gitignore)
├── requirements.txt     # Danh sách dependency
├── start.bat            # Chạy app không hiện terminal
├── restart.bat          # Khởi động lại app
├── yolo26m-pose.pt      # Model YOLO Pose mới nhất (~53MB)
├── yolov8m-pose.pt      # YOLOv8 Medium Pose dự phòng (~53MB)
├── yolov8m.pt / s.pt / n.pt  # Các model YOLO khác
├── rule.png             # Ảnh quy tắc gán ID (cũ)
├── rulev2.jpg           # Ảnh quy tắc gán ID (mới)
└── PROJECT_EXPLANATION.md / README.md
```

---

## 📂 Giải thích từng file

### 1. `main.py` — Điểm khởi chạy

```python
# Tạo app PyQt5 Fusion style → MainWindow → exec
app = QApplication(sys.argv)
app.setStyle("Fusion")
win = MainWindow()
win.show()
sys.exit(app.exec_())
```

### 2. `config.py` — Cấu hình & tải model

| Nhóm | Thông số | Giá trị | Ý nghĩa |
|------|----------|---------|----------|
| **Model** | `_MODEL_NAME` | `yolo26m-pose.pt` | Model YOLO Pose chính |
| **Phát hiện** | `_CONF` | 0.25 | Ngưỡng tin cậy "có người" |
| | `_NMS_IOU` | 0.5 | Non-Maximum Suppression |
| **Khử trùng** | `_HEAD_DUP` | 0.5 | Khoảng cách đầu tối đa (× vai) để coi là cùng người |
| | `_CONTAIN_THR` | 0.85 | Dự phòng: khung con trong khung cha |
| | `_DUP_AREA_RATIO` | 0.75 | Dự phòng: tỉ lệ diện tích trùng |
| **Cắt rìa** | `_DROP_EDGE_PX` | 5 | Bỏ người bị cắt ở rìa TRÁI (x1 ≤ 5px). 0 = tắt |
| **Tracking** | `_TRACK_IOU` | 0.3 | Ngưỡng IoU để khớp cùng người giữa 2 frame |
| **Đánh số** | `_NUM_ROWS` | 3 | Số hàng ghế (đánh số dưới→trên, trái→phải) |
| **Cắt Pose** | `_KP_CONF` | 0.30 | Ngưỡng tin cậy điểm khớp |
| | `_CROWN_FACTOR` | 0.65 | Ước lượng đỉnh đầu |
| | `_FINGER_FACTOR` | 0.60 | Nới tay giơ |
| | `_TORSO_FACTOR` | 1.25 | Đáy cắt (tới ngực) |
| | `_SIDE_MARGIN` | 0.20 | Nới hai bên |
| | `_MARGIN_PX` | 7 | Padding cố định (px) |
| | `_PAD` | 0.12 | Padding fallback khi thiếu keypoints |

**`get_model()`**: Singleton (có khoá double-check, an toàn đa luồng) — tải model YOLO một lần, tái sử dụng.

**`predict(image, **kw)`**: Lời gọi YOLO **có khoá** (`_infer_lock`). Ultralytics dùng chung một predictor nội bộ nên không được chạy song song (luồng nền làm nóng vs luồng GUI phát hiện). Mọi nơi gọi YOLO đều đi qua hàm này.

**`warmup()`**: Nạp model + chạy 1 inference giả để "làm nóng" (import nặng + cấp phát GPU). Được gọi ở **luồng nền** lúc mở app để giao diện không bị đơ. `gallery.warmup()` làm tương tự cho ResNet50.

### 3. `detection.py` — Lõi xử lý ảnh

#### a) Phát hiện — `detect(image)`
```
Input:  Ảnh BGR
Output: (boxes, kpts) — đã khử trùng
```

Quy trình:
1. Chạy YOLO Pose (qua `config.predict`, có khoá) → boxes + 17 keypoints
2. **Khử trùng theo Pose**: Sắp xếp khung lớn→nhỏ, so sánh **khoảng cách đầu** (mũi/mắt/tai) chuẩn hóa theo bề rộng vai. Nếu khoảng cách < `_HEAD_DUP × scale` → cùng người → bỏ khung nhỏ. Dự phòng: IoU > 0.7
3. **Bỏ rìa trái**: Nếu `_DROP_EDGE_PX > 0`, loại người có `x1 ≤ _DROP_EDGE_PX` (thường là người qua đường, bị cắt nửa người)

#### b) Cắt thông minh — `crop_region(box, kpts, w, h)`

Dùng **bề rộng vai (S)** làm thước đo:

1. **Đỉnh đầu** = `head_top - 0.65×S` (ước lượng đỉnh sọ, tránh lẹm tóc)
2. **Tay giơ**: Nếu cổ tay (keypoint 9,10) cao hơn đỉnh → nới lên `cổ_tay - 0.60×S` (lấy đủ ngón). Nếu khuỷu (7,8) cao → nới đến khuỷu
3. **Đáy**: `vai + 1.25×S` (dừng quanh ngực, không xuống bàn)
4. **Hai bên**: Nới 20% bề rộng + 7px. Chỉ tính tay đang giơ nếu ở trên ngực
5. **Fallback**: Thiếu keypoints → `bbox + padding 12%`

#### c) Đánh số — `order_boxes(boxes, kpts)`

Phân cụm 1D theo tọa độ Y của đầu → `_NUM_ROWS` hàng (tối ưu tổng phương sai). Hàng dưới cùng trước, trong hàng từ trái→phải. Dùng **đầu** (không dùng đỉnh khung) → giơ tay không làm số nhảy.

#### d) Tracking — `IoUTracker`

Greedy IoU matching: frame đầu → seed IDs, mỗi frame sau tính IoU giữa detection mới và track cũ, ghép cặp cao nhất (ngưỡng ≥ 0.3). Detection không khớp → tạo track mới.

### 4. `gui.py` — Giao diện PyQt5 (1073 dòng)

#### a) `FrameEditor` — Widget chỉnh khung

- Hiển thị ảnh canh giữa, giữ tỉ lệ
- Vẽ khung: vàng (đang chọn) / xanh mờ (khác), 8 tay cầm
- Kéo cạnh/góc/di chuyển, giữ kích thước tối thiểu, clamp trong ảnh
- Ưu tiên khung đang chọn khi rê chuột

#### b) `MainWindow` — 3 pane QSplitter

| Vùng | Nội dung |
|------|----------|
| **Trái** | Thumbnail các folder batch, ✓/▶ đánh dấu, chuột phải xóa, phím Delete |
| **Giữa** | FrameEditor + nút Đầu/Giữa/Cuối + Khôi phục khung |
| **Phải** | Form gán ID (QComboBox 1–21), nút Khung#, ô gợi ý tô nền xanh/vàng |

**Các nút chức năng:**
- `🧠 Bộ nhớ id` — Mở ảnh quy tắc → detect → gán ID → lưu gallery
- `💾 Lưu bộ nhớ id` — Tính đặc trưng ResNet50 + màu → lưu `gallery.json`
- `🔍 Tải & phát hiện` — Detect YOLO + tự gợi ý ID nếu có gallery
- `🚀 Xử lý & xuất tất cả` — Track + cắt tất cả frame
- `🌐 Gán 1 lần → xuất TẤT CẢ folder` — Batch với IoU matching + chọn phạm vi
- `↺ Khôi phục khung tự động` — Bỏ chỉnh tay

**Gợi ý ID tự động** (`_suggest_ids`):
- Trích đặc trưng từng khung cắt → cosine similarity với gallery
- Ghép 1-1 (mỗi ID chỉ gán 1 người)
- Tô **nền** ô id: xanh (`s ≥ 0.62`) / vàng (`0.45 ≤ s < 0.62`) / không tô (`s < 0.45`).
  Tô nền cả ô (qua `_suggest_style`) thay vì chỉ viền, vì QComboBox có stylesheet toàn cục
  sẽ che viền mảnh. Sửa tay ô nào → `editTextChanged` xoá màu ô đó.
- **Màu được nhớ theo folder**: lưu trong `self._cache[folder]["colors"]` + khôi phục khi
  quay lại folder (không chạy lại gợi ý), nên chuyển qua lại folder không mất màu.

**Làm nóng model lúc mở app** (`_warmup_models`):
- `__init__` khởi một luồng nền (daemon) gọi `config.warmup()` + `gallery.warmup()`.
- Giao diện hiện ra ngay; lần phát hiện đầu không phải chờ import/nạp model (~6–10s).

### 5. `gallery.py` — Bộ nhớ ID theo ngoại hình

#### Kiến trúc đặc trưng kết hợp

```
Đặc trưng = [ √0.45 × CNN(ResNet50) ; √0.55 × Color(HSV) ]   (2048 + 324 = 2372 chiều)
```

- **CNN** (`ResNet50`, ImageNet, bỏ lớp fc): Trích dáng người, kết cấu quần áo
- **Màu** (histogram H+S vùng thân 32%–98% cao, 18%–82% rộng): Bất biến tỉ lệ, lọc nền, Hellinger normalization
- Cosine similarity trên vector kết hợp

#### Quy trình

1. `build_gallery()`: Mở ảnh quy tắc → detect → gán ID → `save_gallery_now()` → `embed()` từng crop → lưu vector
2. `load_and_detect()`: Nếu có gallery và ID chưa gán → `_suggest_ids()` → cosine match + greedy 1-1
3. `gallery.json`: Dictionary `{id: [vector, ...]}`

---

## 🔄 Luồng hoạt động chính

```
1. Người dùng chọn folder chứa ~60 frame
                    ↓
2. App chạy YOLO Pose trên FRAME ĐẦU TIÊN
   → Phát hiện người + 17 keypoints
   → Khử trùng (khoảng cách đầu)
   → Bỏ rìa trái (DROP_EDGE_PX)
   → Tính vùng cắt (đầu + nửa ngực)
   → Gợi ý ID (nếu có gallery)
                    ↓
3. Hiển thị frame với khung cắt, đánh số #1, #2...
   → Ô id tô nền xanh/vàng nếu ID được gợi ý tự động
   → Người dùng kiểm tra, chỉnh khung/sửa ID nếu cần
                    ↓
4. Nhấn "Xử lý & xuất tất cả"
   → IoU Tracker bám theo từng người qua TẤT CẢ frame
   → Cắt ảnh chân dung cho mỗi frame
   → Áp delta chỉnh tay
   → Frame mất dấu: giữ khung lần gần nhất
   → Lưu: <folder>/id_<N>/<tên_frame>.jpg
                    ↓
5. Kết quả: Mỗi người có folder riêng chứa ~60 ảnh chân dung
```

---

## 🧠 Các thuật toán quan trọng

### Khử trùng theo Pose
So sánh khoảng cách đầu (mũi/mắt/tai) chuẩn hóa bằng bề rộng vai → bỏ khung trùng, giữ người bị che.

### Crop thông minh theo Pose
Thước đo = bề rộng vai (S). Xử lý: cúi, tay giơ, khuỷu giơ, che một phần.

### Phân cụm hàng 1D
Tối ưu tổng phương sai (`_segment_1d`) → chia `_NUM_ROWS` hàng. Đánh số dưới→trên, trái→phải.

### Gợi ý ID (Appearance Matching)
Đặc trưng kết hợp ResNet50 (45%) + màu HSV (55%). Cosine similarity → greedy 1-1 matching. Ngưỡng: xanh ≥ 0.62, vàng ≥ 0.45.

### Greedy IoU Tracking
Tính IoU mọi cặp → sắp xếp giảm dần → ghép cao nhất trước → dư tạo track mới.

---

## 📊 Model YOLO trong dự án

| File | Loại | Kích thước | Dùng cho |
|------|------|-----------|----------|
| `yolo26m-pose.pt` | **YOLO Pose (mới)** | ~53MB | **Chính** |
| `yolov8m-pose.pt` | YOLOv8 Medium Pose | ~53MB | Dự phòng |
| `yolov8m.pt` | YOLOv8 Medium | ~52MB | Detection thuần |
| `yolov8s.pt` | YOLOv8 Small | ~22MB | Nhẹ hơn |
| `yolov8n.pt` | YOLOv8 Nano | ~6.5MB | Nhanh nhất |

---

## 🔧 Các thư viện sử dụng

| Thư viện | Phiên bản | Vai trò |
|----------|-----------|---------|
| **PyQt5** | ≥ 5.15 | GUI |
| **OpenCV** | ≥ 4.5 | Đọc/ghi/xử lý ảnh |
| **NumPy** | ≥ 1.20 | Mảng số |
| **Ultralytics** | ≥ 8.0 | Framework YOLO |
| **PyTorch** | ≥ 2.0 | Backend YOLO + ResNet50 |
| **TorchVision** | ≥ 0.15 | ResNet50 pre-trained |
