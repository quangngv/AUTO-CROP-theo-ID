# 📖 Giải thích chi tiết dự án AUTO CROP theo ID

## 🎯 Mục đích dự án

Dự án này là một ứng dụng **tự động cắt ảnh chân dung từng người** từ các frame video đã được trích xuất sẵn (khoảng 60 frame/clip). Ứng dụng sử dụng **YOLOv8 Pose** để phát hiện người và điểm khớp cơ thể, sau đó **tracking (bám theo)** từng người qua tất cả frame để tạo ra dataset ảnh riêng cho mỗi cá nhân.

### Ứng dụng thực tế
- Tạo **dataset khuôn mặt/chân dung** cho từng người từ video giám sát
- Phục vụ cho **nhận dạng khuôn mặt**, huấn luyện model AI
- Xử lý hàng loạt nhiều video clip cùng lúc

---

## 🏗️ Kiến trúc dự án

```
test1/
├── main.py              # Điểm khởi chạy - chỉ tạo cửa sổ PyQt5
├── config.py            # Cấu hình: thông số model, ngưỡng phát hiện, tham số cắt
├── detection.py         # Xử lý ảnh: phát hiện người, khử trùng, cắt khung, tracking
├── gui.py               # Giao diện đồ họa PyQt5 (cửa sổ chính + editor khung cắt)
├── yolov8m-pose.pt      # Model YOLOv8 Medium Pose (~53MB) - model chính
├── yolov8m.pt           # Model YOLOv8 Medium (~52MB) - dự phòng
├── yolov8n.pt           # Model YOLOv8 Nano (~6.5MB) - nhẹ, nhanh
├── yolov8s.pt           # Model YOLOv8 Small (~22MB)
└── rulepic.png          # Ảnh bộ quy tắc gán ID (hướng dẫn cho người dùng)
```

---

## 📂 Giải thích từng file

### 1. `main.py` — Điểm khởi chạy

File rất ngắn gọn, chỉ làm 2 việc:
- Tạo ứng dụng PyQt5 với style **Fusion** (giao diện hiện đại, đồng nhất trên mọi OS)
- Khởi tạo và hiển thị cửa sổ chính `MainWindow`

```
Chạy: python main.py
```

### 2. `config.py` — Cấu hình & tải model

Chứa **tất cả thông số tinh chỉnh** của ứng dụng, chia thành các nhóm:

| Nhóm | Thông số | Giá trị | Ý nghĩa |
|------|----------|---------|----------|
| **Model** | `_MODEL_NAME` | `yolov8m-pose.pt` | Model YOLO Pose dùng để phát hiện người + điểm khớp |
| **Phát hiện** | `_CONF` | 0.25 | Ngưỡng tin cậy để coi là "có người" |
| | `_NMS_IOU` | 0.5 | Non-Maximum Suppression - loại bỏ khung chồng nhau |
| **Khử trùng** | `_CONTAIN_THR` | 0.85 | Ngưỡng "khung con nằm trong khung cha" |
| | `_DUP_AREA_RATIO` | 0.75 | Tỉ lệ diện tích để xác định khung trùng |
| **Tracking** | `_TRACK_IOU` | 0.3 | Ngưỡng IoU để khớp cùng 1 người giữa 2 frame |
| **Cắt Pose** | `_CROWN_FACTOR` | 0.75 | Ước lượng đỉnh đầu (tránh cắt lẹm tóc) |
| | `_FINGER_FACTOR` | 0.55 | Nới vùng cắt khi tay giơ (lấy đủ ngón) |
| | `_TORSO_FACTOR` | 1.6 | Đáy cắt: dưới vai 1.6× bề rộng vai (tới ngực) |
| | `_SIDE_MARGIN` | 0.22 | Nới hai bên khung cắt |
| | `_PAD` | 0.12 | Padding fallback khi thiếu điểm khớp |

**Hàm `get_model()`**: Singleton pattern — tải model YOLO một lần duy nhất và tái sử dụng.

### 3. `detection.py` — Xử lý ảnh (không dính giao diện)

Đây là **lõi xử lý** của ứng dụng, gồm các phần:

#### a) Tiện ích hình học
- **`_iou(a, b)`**: Tính Intersection over Union giữa 2 bounding box — thước đo mức độ trùng lặp
- **`_contain_ratio(a, b)`**: Tỉ lệ phần giao so với khung nhỏ hơn — phát hiện khung con nằm trong khung cha

#### b) Phát hiện người — `detect(image)`
```
Input:  Ảnh BGR (OpenCV)
Output: (boxes, kpts)
  - boxes: danh sách [x1, y1, x2, y2] - khung bao người
  - kpts:  danh sách mảng (17, 3) - 17 điểm khớp COCO [x, y, confidence]
```

Quy trình:
1. Chạy model YOLOv8 Pose → lấy bounding box + keypoints
2. **Khử trùng**: Sắp xếp theo diện tích giảm dần → bỏ khung nhỏ nằm gần trọn trong khung lớn VÀ có kích thước tương đương (= cùng 1 người bị phát hiện 2 lần)
3. Người bị che một phần có khung nhỏ hơn nhiều → vẫn được giữ lại

#### c) Cắt khung thông minh — `crop_region(box, kpts, w, h)`

Đây là thuật toán cốt lõi, cắt vùng **đầu + nửa ngực** sử dụng điểm khớp COCO:

```
Điểm khớp COCO sử dụng:
  0: Mũi          5-6: Vai
  1-2: Mắt        7-8: Khuỷu tay
  3-4: Tai        9-10: Cổ tay
                   11-12: Hông (không dùng — bỏ chân)
```

**Thuật toán**:
1. Lấy **bề rộng vai (S)** làm thước đo (ổn định khi người cúi/chồm)
2. **Đỉnh đầu** = điểm cao nhất mắt/tai − 0.75×S (ước lượng đỉnh sọ)
3. Nếu **tay giơ**: nới trên cổ tay 0.55×S → lấy đủ ngón tay
4. **Đáy**: vai + 1.6×S → dừng quanh ngực, không kéo xuống bàn
5. **Hai bên**: nới 22% bề rộng + 8px cố định
6. **Fallback**: Khi thiếu khớp đầu/vai → dùng bbox + padding 12%

#### d) Tracking — `IoUTracker`

Tracker nhẹ dùng **greedy IoU matching**:
1. Khởi tạo với các khung từ frame đầu (ID 1, 2, 3...)
2. Mỗi frame mới: tính IoU giữa detection mới và track cũ
3. Ghép cặp theo IoU cao nhất (greedy), ngưỡng ≥ 0.3
4. Detection không khớp → tạo track ID mới

#### e) Tiện ích khác
- **`order_boxes(boxes)`**: Sắp xếp khung trái→phải, trên→xuống để đánh số trực quan
- **`annotate(image, boxes, labels)`**: Vẽ khung + nhãn lên ảnh (hiển thị)
- **`list_frames(folder)`**: Liệt kê file ảnh, sắp theo số trong tên file
- **`find_clip_folders(parent)`**: Tìm các thư mục con có chứa frame ảnh

### 4. `gui.py` — Giao diện đồ họa PyQt5

Gồm 2 class chính:

#### a) `FrameEditor` — Widget hiển thị ảnh + chỉnh khung cắt

- Hiển thị frame ảnh với tỉ lệ, canh giữa trong widget
- Vẽ các khung cắt (bounding box) lên ảnh
- Cho phép **kéo cạnh/góc/di chuyển** từng khung cắt bằng chuột
- Hỗ trợ 8 tay cầm (handle): 4 góc + 4 cạnh
- Ánh xạ tọa độ chuột ↔ tọa độ ảnh gốc (scale + offset)
- Khung đang chọn: viền vàng, nổi bật; khung khác: viền xanh mờ

#### b) `MainWindow` — Cửa sổ chính

Giao diện chia 3 phần (dùng QSplitter — kéo được):

| Vùng | Nội dung |
|------|----------|
| **Trái** (ẩn/hiện) | Dải thumbnail các folder đã chọn (batch mode) |
| **Giữa** (chính) | Frame ảnh + editor khung cắt |
| **Phải** | Form gán ID cho từng khung |

**Các chức năng chính**:

1. **Chọn folder đơn**: `📂 Chọn...` → chọn 1 folder chứa frame
2. **Chọn nhiều folder**: `📁 Chọn nhiều folder...` → batch processing
3. **Tải & phát hiện**: `🔍 Tải & phát hiện` → chạy YOLO trên frame đầu
4. **Gán ID**: Nhập ID cho từng người được phát hiện
5. **Chỉnh khung**: Kéo cạnh/góc trên ảnh để tinh chỉnh vùng cắt
6. **Xuất**: `🚀 Xử lý & xuất tất cả` → track + cắt qua tất cả frame
7. **Batch**: `🌐 Gán 1 lần → xuất TẤT CẢ folder` → áp ID mẫu cho nhiều folder

**Caching**: Khi chuyển giữa các folder, trạng thái (ID đã gán + khung đã chỉnh) được lưu cache để không mất khi quay lại.

---

## 🔄 Luồng hoạt động chính

```
1. Người dùng chọn folder chứa ~60 frame (trích từ video)
                    ↓
2. App chạy YOLOv8 Pose trên FRAME ĐẦU TIÊN
   → Phát hiện tất cả người + điểm khớp cơ thể
   → Khử trùng (bỏ bounding box trùng lặp)
   → Tính vùng cắt thông minh (đầu + nửa ngực)
                    ↓
3. Hiển thị frame với các khung cắt, đánh số #1, #2, #3...
   → Người dùng chỉnh khung cắt bằng chuột (nếu cần)
   → Gõ ID thật cho từng người (VD: "1", "2", để trống = bỏ)
                    ↓
4. Nhấn "Xử lý & xuất tất cả"
   → IoU Tracker bám theo từng người qua TẤT CẢ frame
   → Cắt ảnh chân dung cho mỗi frame
   → Áp delta chỉnh tay (nếu có) lên khung cắt tự động
   → Lưu vào: <folder>/id_<N>/<tên_frame>.jpg
                    ↓
5. Kết quả: Mỗi người có folder riêng chứa ~60 ảnh chân dung
```

---

## 🧠 Các thuật toán quan trọng

### IoU (Intersection over Union)
```
         Diện tích phần giao
IoU = ─────────────────────────────
       Diện tích phần hợp (A ∪ B)
```
- IoU = 1.0: hai khung trùng hoàn toàn
- IoU = 0.0: hai khung không giao nhau
- Dùng trong: NMS, khử trùng, tracking

### Greedy IoU Tracking
1. Tính IoU giữa mọi cặp (detection mới, track cũ)
2. Sắp xếp giảm dần theo IoU
3. Ghép cặp (detection, track) có IoU cao nhất trước
4. Mỗi detection/track chỉ được ghép 1 lần
5. Detection dư → tạo track mới (người mới xuất hiện)

### Crop thông minh theo Pose
- Thước đo = **bề rộng vai (S)** — ổn định hơn khoảng cách đầu-vai (bị co khi cúi)
- Tự động xử lý: người cúi, tay giơ, bị che một phần
- Fallback: khi YOLO không đủ keypoints → dùng bbox + padding

---

## 📊 Model YOLOv8 trong dự án

| File | Loại | Kích thước | Dùng cho |
|------|------|-----------|----------|
| `yolov8m-pose.pt` | **Medium Pose** | ~53MB | **Chính** — phát hiện người + 17 điểm khớp |
| `yolov8m.pt` | Medium | ~52MB | Dự phòng (chỉ detection, không có keypoints) |
| `yolov8s.pt` | Small | ~22MB | Phương án nhẹ hơn |
| `yolov8n.pt` | Nano | ~6.5MB | Phương án nhanh nhất |

> **Lưu ý**: Dự án mặc định dùng `yolov8m-pose.pt` vì cần cả keypoints để cắt chân dung chính xác. Các model khác có thể dùng khi đổi `_MODEL_NAME` trong `config.py`.

---

## 🔧 Các thư viện sử dụng

| Thư viện | Phiên bản | Vai trò |
|----------|-----------|---------|
| **PyQt5** | ≥ 5.15 | Giao diện đồ họa (GUI) |
| **OpenCV** (`opencv-python`) | ≥ 4.5 | Đọc/ghi ảnh, xử lý hình ảnh |
| **NumPy** | ≥ 1.20 | Xử lý mảng số |
| **Ultralytics** | ≥ 8.0 | Framework YOLOv8 (phát hiện + pose estimation) |
| **PyTorch** | (tự cài kèm Ultralytics) | Backend deep learning cho YOLO |
