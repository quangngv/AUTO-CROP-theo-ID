# ✂️ AUTO CROP theo ID — Tạo dataset từng người từ video

Ứng dụng desktop sử dụng **YOLO Pose** và **PyQt5** để tự động phát hiện, bám theo (tracking), và cắt ảnh chân dung từng người qua tất cả frame trích từ video. Hỗ trợ **gợi ý ID tự động** dựa trên ngoại hình (ResNet50 + màu áo).

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green)
![YOLO](https://img.shields.io/badge/AI-YOLO%20Pose-red)

---

## 📋 Mục lục

- [Tính năng](#-tính-năng)
- [Yêu cầu hệ thống](#-yêu-cầu-hệ-thống)
- [Cài đặt](#-cài-đặt)
  - [Bước 1: Cài Python](#bước-1-cài-python)
  - [Bước 2: Tạo môi trường ảo (khuyên dùng)](#bước-2-tạo-môi-trường-ảo-khuyên-dùng)
  - [Bước 3: Cài thư viện](#bước-3-cài-thư-viện)
  - [Bước 4: Tải model YOLO](#bước-4-tải-model-yolo)
- [Chạy ứng dụng](#-chạy-ứng-dụng)
- [Hướng dẫn sử dụng](#-hướng-dẫn-sử-dụng)
- [Bộ nhớ ID — Gợi ý tự động](#-bộ-nhớ-id--gợi-ý-tự-động)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Cấu hình nâng cao](#-cấu-hình-nâng-cao)
- [Xử lý lỗi thường gặp](#-xử-lý-lỗi-thường-gặp)

---

## ✨ Tính năng

- 🔍 **Phát hiện tự động**: Dùng YOLO Pose phát hiện người + 17 điểm khớp cơ thể (COCO keypoints)
- ✂️ **Cắt thông minh**: Cắt chân dung đầu + nửa ngực dựa trên bề rộng vai, bao gồm cả tay giơ
- 🧹 **Khử trùng theo Pose**: Loại bỏ bounding box trùng bằng khoảng cách đầu — người bị che vẫn được giữ
- 🏃 **Tracking**: Bám theo từng người qua tất cả frame bằng greedy IoU matching
- 🖱️ **Chỉnh tay**: Kéo cạnh/góc/di chuyển khung cắt trực tiếp trên ảnh (8 handle)
- 🧠 **Gợi ý ID tự động**: So khớp ngoại hình (ResNet50 + màu áo) với bộ nhớ ID đã lưu
- 📁 **Batch processing**: Xử lý nhiều folder video cùng lúc, áp ID mẫu cho tất cả
- 💾 **Cache thông minh**: Nhớ trạng thái khi chuyển giữa các folder; lưu phiên làm việc ra đĩa

---

## 💻 Yêu cầu hệ thống

| Thành phần | Yêu cầu tối thiểu | Khuyên dùng |
|------------|-------------------|-------------|
| **OS** | Windows 10 / Linux / macOS | Windows 10/11 |
| **Python** | 3.8+ | 3.10 – 3.11 |
| **RAM** | 4GB | 8GB+ |
| **GPU** | Không bắt buộc | NVIDIA GPU có CUDA (nhanh hơn 5-10x) |
| **Ổ cứng** | ~1GB (model + thư viện) | SSD |

---

## 🚀 Cài đặt

### Bước 1: Cài Python

1. Tải Python từ [python.org](https://www.python.org/downloads/)
   - Khuyên dùng **Python 3.10** hoặc **3.11**
2. Khi cài, **TICK vào ô "Add Python to PATH"** (rất quan trọng!)
3. Kiểm tra sau khi cài:

```bash
python --version
# Kết quả mong đợi: Python 3.10.x hoặc tương tự
```

> ⚠️ **Lưu ý Windows**: Nếu lệnh `python` không nhận, thử `python3` hoặc `py`

### Bước 2: Tạo môi trường ảo (khuyên dùng)

Mở **Command Prompt** hoặc **PowerShell**, di chuyển đến thư mục dự án:

```bash
# Di chuyển đến thư mục dự án
cd "C:\Users\Admin\Desktop\new project\test1"

# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường ảo
# Windows (Command Prompt):
venv\Scripts\activate

# Windows (PowerShell):
venv\Scripts\Activate.ps1

# Linux / macOS:
source venv/bin/activate
```

> Khi kích hoạt thành công, dòng lệnh sẽ hiện `(venv)` ở đầu.

### Bước 3: Cài thư viện

```bash
pip install -r requirements.txt
```

Nội dung `requirements.txt`:
```
opencv-python>=4.5.0
numpy>=1.20.0
PyQt5>=5.15.0
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0
```

#### 🎮 Cài PyTorch với CUDA (nếu có GPU NVIDIA)

Mặc định, `ultralytics` sẽ cài PyTorch bản CPU. Để dùng GPU cho tốc độ nhanh hơn:

```bash
# Kiểm tra phiên bản CUDA đã cài
nvidia-smi

# Cài PyTorch với CUDA 11.8 (phổ biến nhất)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

> Xem thêm tại [pytorch.org](https://pytorch.org/get-started/locally/) để chọn đúng phiên bản CUDA.

#### Kiểm tra cài đặt thành công

```bash
python -c "import cv2; import numpy; import PyQt5; from ultralytics import YOLO; import torch; import torchvision; print('✅ Tất cả thư viện OK!')"
```

### Bước 4: Tải model YOLO

Dự án cần file model pose. Có 2 cách:

**Cách 1**: Model đã có sẵn trong thư mục dự án (`yolo26m-pose.pt`) → không cần làm gì thêm.

**Cách 2**: Tự động tải lần đầu chạy — Ultralytics sẽ tự tải nếu không tìm thấy file.

---

## ▶️ Chạy ứng dụng

```bash
# Đảm bảo đang ở thư mục dự án và đã kích hoạt venv
cd "C:\Users\Admin\Desktop\new project\test1"

# Chạy
python main.py
```

**Hoặc dùng batch script:**
- `start.bat` — Chạy ứng dụng không hiện terminal (dùng `pythonw`)
- `restart.bat` — Tắt ứng dụng cũ và khởi động lại

---

## 📖 Hướng dẫn sử dụng

### Chế độ đơn (1 folder)

1. **Chọn folder**: Nhấn `📂 Chọn...` → chọn thư mục chứa các frame ảnh (trích từ video)
2. **Phát hiện**: Nhấn `🔍 Tải & phát hiện` → app chạy YOLO, hiển thị khung cắt trên frame đầu
3. **Gợi ý ID**: Nếu đã có bộ nhớ ID, app tự gợi ý — viền **xanh** = chắc chắn, viền **vàng** = nên kiểm tra lại
4. **Chỉnh khung** (nếu cần): Kéo cạnh/góc/di chuyển khung trực tiếp trên ảnh
5. **Gán ID**: Chọn ID từ dropdown hoặc gõ số cho từng người ở cột bên phải (để trống = bỏ qua)
6. **Xuất**: Nhấn `🚀 Xử lý & xuất tất cả` → app xử lý toàn bộ frame

### Chế độ batch (nhiều folder)

1. **Chọn nhiều folder**: Nhấn `📁 Chọn nhiều folder...` → giữ Ctrl/Shift để chọn nhiều
2. **Gán ID mẫu**: Gán ID cho folder đầu tiên
3. Có 2 lựa chọn:
   - `🚀 Xử lý & folder tiếp →`: Xử lý từng folder, chỉnh ID riêng nếu cần
   - `🌐 Gán 1 lần → xuất TẤT CẢ folder`: Áp ID mẫu cho tất cả folder (nhanh, phù hợp khi camera cố định)

### Chế độ xem frame khác

Dùng các nút `📍 Đầu / Giữa / Cuối` để xem frame ở các vị trí khác nhau trong clip (giữ nguyên khung cắt để so sánh).

### Kết quả đầu ra

```
<folder_gốc>/
├── id_1/           # Ảnh chân dung người ID 1
│   ├── frame_001.jpg
│   ├── frame_002.jpg
│   └── ...
├── id_2/           # Ảnh chân dung người ID 2
│   ├── frame_001.jpg
│   └── ...
└── (các frame gốc)
```

---

## 🧠 Bộ nhớ ID — Gợi ý tự động

### Cách tạo bộ nhớ ID lần đầu

1. Nhấn `🧠 Bộ nhớ id` → chọn ảnh quy tắc (ảnh chụp tất cả sinh viên đã đánh số)
2. App tự phát hiện người trong ảnh, hiện khung cắt
3. Gán số ID (01–21) cho từng người theo đúng ảnh quy tắc
4. Nhấn `💾 Lưu bộ nhớ id` → app trích đặc trưng ResNet50 + màu áo → lưu vào `gallery.json`

### Cách hoạt động

- Khi mở folder mới, app tự động so khớp ngoại hình từng người với bộ nhớ đã lưu
- **Viền xanh** (`s ≥ 0.62`): ID chắc chắn — có thể tin tưởng
- **Viền vàng** (`0.45 ≤ s < 0.62`): ID nghi ngờ — nên kiểm tra lại
- **Không viền** (`s < 0.45`): ID không tự động — cần gán tay

---

## 📁 Cấu trúc thư mục

```
test1/
├── main.py              # Điểm khởi chạy
├── config.py            # Cấu hình model + thông số cắt
├── detection.py         # Logic phát hiện, khử trùng, cắt, tracking
├── gui.py               # Giao diện PyQt5 (1073 dòng)
├── gallery.py           # Bộ nhớ ID theo ngoại hình (ResNet50 + màu)
├── gallery.json         # Vector đặc trưng đã lưu
├── app_state.json       # Lưu phiên làm việc (tự động)
├── requirements.txt     # Danh sách thư viện cần cài
├── start.bat            # Chạy app không hiện terminal
├── restart.bat          # Khởi động lại app
├── yolo26m-pose.pt      # Model chính (YOLO Pose, mới nhất)
├── yolov8m-pose.pt      # Model dự phòng (YOLOv8 Medium Pose)
├── yolov8m.pt           # YOLOv8 Medium (detection thuần)
├── yolov8s.pt           # YOLOv8 Small (nhẹ hơn)
├── yolov8n.pt           # YOLOv8 Nano (nhẹ nhất)
├── rule.png             # Ảnh hướng dẫn quy tắc gán ID (cũ)
├── rulev2.jpg           # Ảnh quy tắc gán ID (mới)
├── PROJECT_EXPLANATION.md   # Giải thích chi tiết dự án
└── README.md            # File này
```

---

## ⚙️ Cấu hình nâng cao

Mở file `config.py` để điều chỉnh:

### Đổi model
```python
_MODEL_NAME = "yolo26m-pose.pt"   # Mặc định: YOLO Pose mới nhất
```

### Điều chỉnh ngưỡng phát hiện
```python
_CONF = 0.25        # Tăng lên (0.4-0.5) nếu bị phát hiện nhầm
_NMS_IOU = 0.5      # Giảm nếu có quá nhiều khung trùng
```

### Khử trùng theo Pose
```python
_HEAD_DUP = 0.5     # 2 khung có đầu cách nhau < 0.5 × bề rộng vai → cùng người
```

### Bỏ người rìa trái
```python
_DROP_EDGE_PX = 5   # Bỏ người có khung x1 <= 5px (bị cắt ở rìa trái). 0 = TẮT
```

### Điều chỉnh vùng cắt
```python
_CROWN_FACTOR = 0.65   # Ước lượng đỉnh đầu (tránh cắt lẹm tóc)
_FINGER_FACTOR = 0.60  # Nới vùng cắt khi tay giơ (lấy đủ ngón)
_TORSO_FACTOR = 1.25   # Đáy cắt: dưới vai × hệ số (tới ngực)
_SIDE_MARGIN = 0.20    # Nới hai bên khung cắt
_MARGIN_PX = 7         # Padding pixel cố định
```

---

## ❗ Xử lý lỗi thường gặp

### 1. `ModuleNotFoundError`
```bash
pip install -r requirements.txt
```

### 2. YOLO chạy chậm (dùng CPU)
- Cài PyTorch với CUDA
- Hoặc đổi sang model nhẹ hơn trong `config.py`:
```python
_MODEL_NAME = "yolov8s-pose.pt"
```

### 3. `qt.qpa.plugin: Could not find the Qt platform plugin "windows"`
```bash
pip uninstall PyQt5 PyQt5-sip PyQt5-Qt5 && pip install PyQt5
```

### 4. PowerShell không cho chạy script (venv)
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 📝 Ghi chú thêm

- **Dữ liệu đầu vào**: Các frame ảnh trích từ video (khoảng 60 frame/clip), đặt trong 1 folder
- **Chỉ xử lý frame đầu bằng YOLO**: Các frame sau dùng tracking (IoU) → nhanh hơn nhiều
- **Gợi ý ID**: Cần tạo bộ nhớ ID trước bằng nút `🧠 Bộ nhớ id`
- **Delta chỉnh tay**: Nếu bạn chỉnh khung ở frame đầu, phần chênh lệch sẽ được áp dụng cho tất cả frame sau
- **Cache**: Khi chuyển folder trong batch mode, trạng thái ID + khung được lưu lại, quay lại không mất

---

## 📄 License

Dự án sử dụng cho mục đích nghiên cứu và thực tập.

Model YOLO thuộc bản quyền [Ultralytics](https://github.com/ultralytics/ultralytics) — sử dụng theo giấy phép AGPL-3.0.
