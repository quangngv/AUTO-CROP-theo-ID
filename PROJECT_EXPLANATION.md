# 📖 Giải thích chi tiết dự án AUTO CROP theo ID

## 🎯 Mục đích dự án

Dự án này là một ứng dụng **tự động cắt ảnh chân dung từng người** từ các frame video đã được trích xuất sẵn (khoảng 60 frame/clip). Ứng dụng sử dụng **YOLO Pose** để phát hiện người và 17 điểm khớp cơ thể COCO, sau đó **tracking (bám theo)** từng người qua tất cả frame để tạo ra dataset ảnh riêng cho mỗi cá nhân.

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
├── app_state.json       # Lưu phiên làm việc (tự động, bị gitignore)
├── start.bat            # Script chạy app không hiện terminal (dùng pythonw)
├── restart.bat          # Script khởi động lại app
├── yolo26m-pose.pt      # Model YOLO Pose mới nhất (~53MB) - model chính
├── yolov8m-pose.pt      # Model YOLOv8 Medium Pose (~53MB) - dự phòng
├── yolov8m.pt           # Model YOLOv8 Medium (~52MB) - detection thuần
├── yolov8s.pt           # Model YOLOv8 Small (~22MB)
├── yolov8n.pt           # Model YOLOv8 Nano (~6.5MB)
└── rule.png             # Ảnh bộ quy tắc gán ID (hướng dẫn cho người dùng)
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

### 2. `start.bat` / `restart.bat` — Script tiện ích

- `start.bat`: Tìm `pythonw.exe` để chạy app không hiện cửa sổ terminal
- `restart.bat`: Dùng `taskkill` đóng app cũ (theo tiêu đề cửa sổ), sau đó khởi động lại

### 3. `config.py` — Cấu hình & tải model

Chứa **tất cả thông số tinh chỉnh** của ứng dụng, chia thành các nhóm:

| Nhóm | Thông số | Giá trị | Ý nghĩa |
|------|----------|---------|----------|
| **Model** | `_MODEL_NAME` | `yolo26m-pose.pt` | Model YOLO Pose dùng để phát hiện người + 17 điểm khớp |
| **Phát hiện** | `_CONF` | 0.25 | Ngưỡng tin cậy để coi là "có người" |
| | `_NMS_IOU` | 0.5 | Non-Maximum Suppression - loại bỏ khung chồng nhau |
| **Khử trùng** | `_HEAD_DUP` | 0.5 | 2 khung có đầu cách nhau < 0.5 × bề rộng vai → cùng 1 người |
| | `_CONTAIN_THR` | 0.85 | Khung con nằm trong khung cha (dự phòng) |
| | `_DUP_AREA_RATIO` | 0.75 | Tỉ lệ diện tích để xác định khung trùng (dự phòng) |
| **Tracking** | `_TRACK_IOU` | 0.3 | Ngưỡng IoU để khớp cùng 1 người giữa 2 frame |
| **Đánh số** | `_NUM_ROWS` | 3 | Số hàng ghế trong phòng (đánh số dưới→trên, trái→phải) |
| **Cắt Pose** | `_KP_CONF` | 0.30 | Ngưỡng tin cậy của từng điểm khớp |
| | `_CROWN_FACTOR` | 0.65 | Ước lượng đỉnh đầu (tránh cắt lẹm tóc) |
| | `_FINGER_FACTOR` | 0.60 | Nới vùng cắt khi tay giơ (lấy đủ ngón) |
| | `_TORSO_FACTOR` | 1.25 | Đáy cắt: dưới vai × hệ số (tới ngực) |
| | `_SIDE_MARGIN` | 0.20 | Nới hai bên khung cắt theo bề rộng |
| | `_MARGIN_PX` | 7 | Nới thêm cố định (pixel) |
| | `_PAD` | 0.12 | Padding fallback khi thiếu điểm khớp |

**Hàm `get_model()`**: Singleton pattern — tải model YOLO một lần duy nhất và tái sử dụng.

### 4. `detection.py` — Xử lý ảnh (không dính giao diện)

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
1. Chạy model YOLO Pose → lấy bounding box + 17 keypoints
2. **Khử trùng theo Pose**: So sánh khoảng cách giữa các **điểm đầu** (mũi/mắt/tai), chuẩn hóa theo bề rộng vai:
   - 2 khung có đầu cách nhau < `_HEAD_DUP × bề rộng vai` → cùng 1 người → bỏ khung nhỏ hơn
   - Người bị che một phần có vị trí đầu khác → vẫn được giữ lại
   - Dự phòng: IoU > 0.7 cũng coi là trùng (khi thiếu keypoints)

#### c) Cắt khung thông minh — `crop_region(box, kpts, w, h)`

Đây là thuật toán cốt lõi, cắt vùng **đầu + nửa ngực** sử dụng 17 điểm khớp COCO:

```
Điểm khớp COCO sử dụng:
  0: Mũi          5-6: Vai
  1-2: Mắt        7-8: Khuỷu tay
  3-4: Tai        9-10: Cổ tay
                   11-12: Hông (không dùng — bỏ chân)
```

**Thuật toán**:
1. Lấy **bề rộng vai (S)** làm thước đo (ổn định khi người cúi/chồm)
2. **Đỉnh đầu** = điểm cao nhất mắt/tai − 0.65×S (ước lượng đỉnh sọ)
3. Nếu **tay giơ**: nới trên cổ tay 0.60×S → lấy đủ ngón tay; khuỷu giơ: nới đến khuỷu
4. **Đáy**: vai + 1.25×S → dừng quanh ngực, không kéo xuống bàn
5. **Hai bên**: nới 20% bề rộng + 7px cố định, tính cả tay đang giơ (nếu ở trên ngực)
6. **Fallback**: Khi thiếu khớp đầu/vai → dùng bbox + padding 12%

#### d) Đánh số thứ tự — `order_boxes(boxes, kpts)`

Sắp xếp người theo **hàng** (dùng phân cụm 1D theo tọa độ Y của đầu), chia thành `_NUM_ROWS` hàng:
- Hàng dưới cùng trước, đi từ **trái → phải**
- Dùng vị trí **đầu** (không dùng đỉnh khung) → người giơ tay không làm số nhảy loạn
- Phân cụm bằng tối ưu hóa tổng phương sai (segmentation 1D)

#### e) Tracking — `IoUTracker`

Tracker nhẹ dùng **greedy IoU matching**:
1. Khởi tạo với các khung từ frame đầu (ID 1, 2, 3...)
2. Mỗi frame mới: tính IoU giữa detection mới và track cũ
3. Ghép cặp theo IoU cao nhất (greedy), ngưỡng ≥ 0.3
4. Detection không khớp → tạo track ID mới

#### f) Tiện ích khác
- **`annotate(image, boxes, labels)`**: Vẽ khung + nhãn lên ảnh (hiển thị)
- **`list_frames(folder)`**: Liệt kê file ảnh, sắp theo số trong tên file
- **`find_clip_folders(parent)`**: Tìm các thư mục con có chứa frame ảnh

### 5. `gui.py` — Giao diện đồ họa PyQt5

Gồm 2 class chính:

#### a) `FrameEditor` — Widget hiển thị ảnh + chỉnh khung cắt

- Hiển thị frame ảnh với tỉ lệ, canh giữa trong widget
- Vẽ các khung cắt (bounding box) lên ảnh
- Cho phép **kéo cạnh/góc/di chuyển** từng khung cắt bằng chuột
- Hỗ trợ 8 tay cầm (handle): 4 góc + 4 cạnh
- Ánh xạ tọa độ chuột ↔ tọa độ ảnh gốc (scale + offset)
- Khung đang chọn: viền vàng, nổi bật; khung khác: viền xanh mờ
- Ưu tiên khung đang chọn khi rê chuột (kể cả khi bị khung khác đè)

#### b) `MainWindow` — Cửa sổ chính

Giao diện chia 3 phần (dùng QSplitter — kéo được):

| Vùng | Nội dung |
|------|----------|
| **Trái** (ẩn/hiện) | Dải thumbnail các folder đã chọn (batch mode) + nút chọn nhanh |
| **Giữa** (chính) | Frame ảnh + editor khung cắt + nút xem Đầu/Giữa/Cuối |
| **Phải** | Form gán ID dùng QComboBox (dropdown + gõ tay) cho từng khung |

**Các chức năng chính**:

1. **Chọn folder đơn**: `📂 Chọn...` → chọn 1 folder chứa frame
2. **Chọn nhiều folder**: `📁 Chọn nhiều folder...` → batch processing (dùng `DontUseNativeDialog` + `ExtendedSelection`)
3. **Tải & phát hiện**: `🔍 Tải & phát hiện` → chạy YOLO trên frame đầu
4. **Gán ID**: Chọn ID từ dropdown (1-21) hoặc gõ tay
5. **Chỉnh khung**: Kéo cạnh/góc trên ảnh để tinh chỉnh vùng cắt
6. **Khôi phục khung**: `↺ Khôi phục khung tự động` → bỏ chỉnh tay
7. **Xuất**: `🚀 Xử lý & xuất tất cả` → track + cắt qua tất cả frame
8. **Batch**: `🌐 Gán 1 lần → xuất TẤT CẢ folder` → áp ID mẫu cho nhiều folder (hỗ trợ chọn phạm vi như "1-5, 7, 9-12")
9. **Xem frame Đầu/Giữa/Cuối**: Kiểm tra tracking mà không mất khung đã chỉnh
10. **Lịch sử**: Tab "Lịch sử" hiển thị các folder đã làm, bấm để mở lại

**Caching**:
- Trong phiên: lưu cache `_cache` (giữ cả detection YOLO để không chạy lại)
- Ra đĩa: lưu `app_state.json` (queue, done, edits) — khôi phục khi mở lại app
- **Lưu cửa sổ**: QSettings lưu kích thước, vị trí, trạng thái phóng to, vạch chia splitter

---

## 🔄 Luồng hoạt động chính

```
1. Người dùng chọn folder chứa ~60 frame (trích từ video)
                    ↓
2. App chạy YOLO Pose trên FRAME ĐẦU TIÊN
   → Phát hiện tất cả người + 17 điểm khớp cơ thể
   → Khử trùng (so sánh khoảng cách ĐẦU)
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
   → Nếu frame mất dấu: giữ khung lần gần nhất để cắt
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

### Khử trùng theo Pose
Thay vì chỉ dùng IoU, thuật toán khử trùng dùng **vị trí đầu** (trung bình mũi/mắt/tai):
1. Sắp xếp khung theo diện tích giảm dần → xét khung to trước
2. Tính khoảng cách đầu giữa 2 khung, chuẩn hóa bằng bề rộng vai
3. Nếu khoảng cách < `_HEAD_DUP × scale` → cùng người → bỏ khung nhỏ
4. Dự phòng: IoU > 0.7 cũng coi là trùng

→ **Ưu điểm**: Người bị che khuất một phần có đầu khác vị trí → được giữ lại

### Greedy IoU Tracking
1. Tính IoU giữa mọi cặp (detection mới, track cũ)
2. Sắp xếp giảm dần theo IoU
3. Ghép cặp (detection, track) có IoU cao nhất trước
4. Mỗi detection/track chỉ được ghép 1 lần
5. Detection dư → tạo track mới (người mới xuất hiện)

### Crop thông minh theo Pose
- Thước đo = **bề rộng vai (S)** — ổn định hơn khoảng cách đầu-vai (bị co khi cúi)
- Tự động xử lý: người cúi, tay giơ, bị che một phần
- Tính cả khuỷu tay giơ cao khi xác định đỉnh khung
- Fallback: khi YOLO không đủ keypoints → dùng bbox + padding

### Phân cụm hàng 1D
- Gom người thành `_NUM_ROWS` hàng dựa trên tọa độ Y của đầu
- Dùng tối ưu tổng phương sai (segment 1D) để tìm ranh giới hàng
- Đánh số: hàng dưới cùng trước (dễ nhìn), trong hàng từ trái → phải

---

## 📊 Model YOLO trong dự án

| File | Loại | Kích thước | Dùng cho |
|------|------|-----------|----------|
| `yolo26m-pose.pt` | **YOLO Pose (mới)** | ~53MB | **Chính** — phát hiện người + 17 điểm khớp |
| `yolov8m-pose.pt` | YOLOv8 Medium Pose | ~53MB | Dự phòng (cấu trúc cũ) |
| `yolov8m.pt` | YOLOv8 Medium | ~52MB | Detection thuần (không keypoints) |
| `yolov8s.pt` | YOLOv8 Small | ~22MB | Phương án nhẹ hơn |
| `yolov8n.pt` | YOLOv8 Nano | ~6.5MB | Phương án nhanh nhất |

> **Lưu ý**: Dự án mặc định dùng `yolo26m-pose.pt` vì cần cả keypoints để cắt chân dung chính xác. Các model khác có thể dùng khi đổi `_MODEL_NAME` trong `config.py`.

---

## 🔧 Các thư viện sử dụng

| Thư viện | Phiên bản | Vai trò |
|----------|-----------|---------|
| **PyQt5** | ≥ 5.15 | Giao diện đồ họa (GUI) |
| **OpenCV** (`opencv-python`) | ≥ 4.5 | Đọc/ghi ảnh, xử lý hình ảnh |
| **NumPy** | ≥ 1.20 | Xử lý mảng số |
| **Ultralytics** | ≥ 8.0 | Framework YOLO (phát hiện + pose estimation) |
| **PyTorch** | (tự cài kèm Ultralytics) | Backend deep learning cho YOLO |
