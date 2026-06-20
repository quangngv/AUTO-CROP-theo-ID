# ✂️ AUTO CROP theo ID — Tạo dataset từng người từ video

Ứng dụng desktop sử dụng **YOLOv8 Pose** và **PyQt5** để tự động phát hiện, bám theo (tracking), và cắt ảnh chân dung từng người qua tất cả frame trích từ video.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green)
![YOLOv8](https://img.shields.io/badge/AI-YOLOv8%20Pose-red)

---

## 📋 Mục lục

- [Tính năng](#-tính-năng)
- [Yêu cầu hệ thống](#-yêu-cầu-hệ-thống)
- [Cài đặt](#-cài-đặt)
  - [Bước 1: Cài Python](#bước-1-cài-python)
  - [Bước 2: Tạo môi trường ảo (khuyên dùng)](#bước-2-tạo-môi-trường-ảo-khuyên-dùng)
  - [Bước 3: Cài thư viện](#bước-3-cài-thư-viện)
  - [Bước 4: Tải model YOLOv8](#bước-4-tải-model-yolov8)
- [Chạy ứng dụng](#-chạy-ứng-dụng)
- [Hướng dẫn sử dụng](#-hướng-dẫn-sử-dụng)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Cấu hình nâng cao](#-cấu-hình-nâng-cao)
- [Xử lý lỗi thường gặp](#-xử-lý-lỗi-thường-gặp)

---

## ✨ Tính năng

- 🔍 **Phát hiện tự động**: Dùng YOLOv8 Pose phát hiện người + điểm khớp cơ thể
- ✂️ **Cắt thông minh**: Cắt chân dung đầu + nửa ngực, bao gồm cả tay giơ
- 🏃 **Tracking**: Bám theo từng người qua tất cả frame bằng IoU matching
- 🖱️ **Chỉnh tay**: Kéo cạnh/góc/di chuyển khung cắt trực tiếp trên ảnh
- 📁 **Batch processing**: Xử lý nhiều folder video cùng lúc
- 💾 **Cache thông minh**: Nhớ trạng thái khi chuyển giữa các folder

---

## 💻 Yêu cầu hệ thống

| Thành phần | Yêu cầu tối thiểu | Khuyên dùng |
|------------|-------------------|-------------|
| **OS** | Windows 10 / Linux / macOS | Windows 10/11 |
| **Python** | 3.8+ | 3.10 – 3.11 |
| **RAM** | 4GB | 8GB+ |
| **GPU** | Không bắt buộc | NVIDIA GPU có CUDA (nhanh hơn 5-10x) |
| **Ổ cứng** | ~500MB (model + thư viện) | SSD |

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

#### Cách 1: Cài nhanh (1 lệnh)

```bash
pip install opencv-python numpy PyQt5 ultralytics
```

#### Cách 2: Cài từ file requirements (nếu có)

```bash
pip install -r requirements.txt
```

#### Cách 3: Cài từng thư viện riêng

```bash
# 1. OpenCV - Xử lý ảnh
pip install opencv-python

# 2. NumPy - Xử lý mảng số
pip install numpy

# 3. PyQt5 - Giao diện đồ họa
pip install PyQt5

# 4. Ultralytics - Framework YOLOv8 (tự cài kèm PyTorch)
pip install ultralytics
```

#### 🎮 Cài PyTorch với CUDA (nếu có GPU NVIDIA)

Mặc định, `ultralytics` sẽ cài PyTorch bản CPU. Để dùng GPU cho tốc độ nhanh hơn:

```bash
# Kiểm tra phiên bản CUDA đã cài
nvidia-smi

# Cài PyTorch với CUDA 11.8 (phổ biến nhất)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Hoặc CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

> Xem thêm tại [pytorch.org](https://pytorch.org/get-started/locally/) để chọn đúng phiên bản CUDA.

#### Kiểm tra cài đặt thành công

```bash
python -c "import cv2; import numpy; import PyQt5; from ultralytics import YOLO; print('✅ Tất cả thư viện OK!')"
```

### Bước 4: Tải model YOLOv8

Dự án cần file model `yolov8m-pose.pt` (~53MB). Có 2 cách:

**Cách 1**: Model đã có sẵn trong thư mục dự án (nếu bạn clone/copy đầy đủ) → không cần làm gì thêm.

**Cách 2**: Tự động tải lần đầu chạy — Ultralytics sẽ tự tải nếu không tìm thấy file.

**Cách 3**: Tải thủ công từ [Ultralytics releases](https://github.com/ultralytics/assets/releases) và đặt vào thư mục dự án.

---

## ▶️ Chạy ứng dụng

```bash
# Đảm bảo đang ở thư mục dự án và đã kích hoạt venv
cd "C:\Users\Admin\Desktop\new project\test1"

# Chạy
python main.py
```

---

## 📖 Hướng dẫn sử dụng

### Chế độ đơn (1 folder)

1. **Chọn folder**: Nhấn `📂 Chọn...` → chọn thư mục chứa các frame ảnh (trích từ video)
2. **Phát hiện**: Nhấn `🔍 Tải & phát hiện` → app chạy YOLO, hiển thị khung cắt trên frame đầu
3. **Chỉnh khung** (nếu cần): Kéo cạnh/góc/di chuyển khung trực tiếp trên ảnh
4. **Gán ID**: Gõ số ID cho từng người ở cột bên phải (để trống = bỏ qua)
5. **Xuất**: Nhấn `🚀 Xử lý & xuất tất cả` → app xử lý toàn bộ frame

### Chế độ batch (nhiều folder)

1. **Chọn nhiều folder**: Nhấn `📁 Chọn nhiều folder...` → giữ Ctrl/Shift để chọn nhiều
2. **Gán ID mẫu**: Gán ID cho folder đầu tiên
3. Có 2 lựa chọn:
   - `🚀 Xử lý & folder tiếp →`: Xử lý từng folder, chỉnh ID riêng nếu cần
   - `🌐 Gán 1 lần → xuất TẤT CẢ folder`: Áp ID mẫu cho tất cả folder (nhanh, phù hợp khi camera cố định)

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

## 📁 Cấu trúc thư mục

```
test1/
├── main.py              # Điểm khởi chạy
├── config.py            # Cấu hình model + thông số cắt
├── detection.py         # Logic phát hiện, cắt, tracking
├── gui.py               # Giao diện PyQt5
├── yolov8m-pose.pt      # Model chính (YOLOv8 Medium Pose)
├── yolov8m.pt           # Model dự phòng
├── yolov8s.pt           # Model nhẹ hơn
├── yolov8n.pt           # Model nhẹ nhất
├── rulepic.png          # Ảnh hướng dẫn quy tắc gán ID
├── PROJECT_EXPLANATION.md   # Giải thích chi tiết dự án
└── README.md            # File này
```

---

## ⚙️ Cấu hình nâng cao

Mở file `config.py` để điều chỉnh:

### Đổi model
```python
_MODEL_NAME = "yolov8m-pose.pt"   # Mặc định: Medium Pose
# Có thể đổi sang:
# "yolov8s-pose.pt"  — nhẹ hơn, nhanh hơn, ít chính xác hơn
# "yolov8l-pose.pt"  — nặng hơn, chính xác hơn
```

### Điều chỉnh ngưỡng phát hiện
```python
_CONF = 0.25      # Tăng lên (0.4-0.5) nếu bị phát hiện nhầm
                   # Giảm xuống (0.15-0.2) nếu bỏ sót người

_NMS_IOU = 0.5    # Giảm nếu có quá nhiều khung trùng
```

### Điều chỉnh vùng cắt
```python
_TORSO_FACTOR = 1.6   # Tăng lên → cắt xuống thấp hơn (lấy thêm ngực/bụng)
_CROWN_FACTOR = 0.75  # Tăng lên → chừa nhiều không gian phía trên đầu
_SIDE_MARGIN = 0.22   # Tăng lên → nới rộng hai bên
```

### Đổi thư mục mặc định
```python
_DEFAULT_DIR = r"C:\Users\Admin\Desktop\2025.2\thuctap\video\savevideo"
# Đổi thành đường dẫn bạn hay dùng
```

---

## ❗ Xử lý lỗi thường gặp

### 1. `ModuleNotFoundError: No module named 'cv2'`
```bash
pip install opencv-python
```

### 2. `ModuleNotFoundError: No module named 'PyQt5'`
```bash
pip install PyQt5
```

### 3. `ModuleNotFoundError: No module named 'ultralytics'`
```bash
pip install ultralytics
```

### 4. `Could not load library libGL` (Linux)
```bash
sudo apt-get install libgl1-mesa-glx
# hoặc
pip install opencv-python-headless
```

### 5. YOLO chạy chậm (dùng CPU)
- Cài PyTorch với CUDA (xem [Bước 3](#-cài-pytorch-với-cuda-nếu-có-gpu-nvidia))
- Hoặc đổi sang model nhẹ hơn trong `config.py`:
```python
_MODEL_NAME = "yolov8s-pose.pt"  # Small: nhanh hơn ~2x
# hoặc
_MODEL_NAME = "yolov8n-pose.pt"  # Nano: nhanh nhất
```

### 6. `qt.qpa.plugin: Could not find the Qt platform plugin "windows"`
```bash
# Thử cài lại PyQt5
pip uninstall PyQt5 PyQt5-sip PyQt5-Qt5
pip install PyQt5
```

### 7. PowerShell không cho chạy script (venv)
```powershell
# Chạy PowerShell với quyền Admin, rồi:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 8. Frame không tải được
- Kiểm tra folder chứa file ảnh có đuôi: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`
- Tên file nên chứa số để sắp xếp đúng thứ tự (VD: `frame_001.jpg`, `frame_002.jpg`)

---

## 📝 Ghi chú thêm

- **Dữ liệu đầu vào**: Các frame ảnh trích từ video (khoảng 60 frame/clip), đặt trong 1 folder
- **Chỉ xử lý frame đầu bằng YOLO**: Các frame sau dùng tracking (IoU) → nhanh hơn nhiều
- **Delta chỉnh tay**: Nếu bạn chỉnh khung ở frame đầu, phần chênh lệch sẽ được áp dụng cho tất cả frame sau
- **Cache**: Khi chuyển folder trong batch mode, trạng thái ID + khung được lưu lại, quay lại không mất

---

## 📄 License

Dự án sử dụng cho mục đích nghiên cứu và thực tập.

Model YOLOv8 thuộc bản quyền [Ultralytics](https://github.com/ultralytics/ultralytics) — sử dụng theo giấy phép AGPL-3.0.
# AUTO-CROP-theo-ID
