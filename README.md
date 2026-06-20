# ✂️ AUTO CROP theo ID — Tạo dataset từng người từ video

Ứng dụng desktop sử dụng **YOLOv8/YOLOv26 Pose** và **PyQt5** để tự động phát hiện, bám theo (tracking), và cắt ảnh chân dung từng người qua tất cả frame trích từ video.

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

# 4. Ultralytics - Framework YOLO (tự cài kèm PyTorch)
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

### Bước 4: Tải model YOLO

Dự án cần file model pose. Có 2 cách:

**Cách 1**: Model đã có sẵn trong thư mục dự án (`yolo26m-pose.pt` hoặc `yolov8m-pose.pt`) → không cần làm gì thêm.

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

**Hoặc dùng batch script:**
- `start.bat` — Chạy ứng dụng không hiện terminal (dùng `pythonw`)
- `restart.bat` — Tắt ứng dụng cũ và khởi động lại

---

## 📖 Hướng dẫn sử dụng

### Chế độ đơn (1 folder)

1. **Chọn folder**: Nhấn `📂 Chọn...` → chọn thư mục chứa các frame ảnh (trích từ video)
2. **Phát hiện**: Nhấn `🔍 Tải & phát hiện` → app chạy YOLO, hiển thị khung cắt trên frame đầu
3. **Chỉnh khung** (nếu cần): Kéo cạnh/góc/di chuyển khung trực tiếp trên ảnh
4. **Gán ID**: Chọn ID từ dropdown hoặc gõ số cho từng người ở cột bên phải (để trống = bỏ qua)
5. **Xuất**: Nhấn `🚀 Xử lý & xuất tất cả` → app xử lý toàn bộ frame

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

## 📁 Cấu trúc thư mục

```
test1/
├── main.py              # Điểm khởi chạy
├── config.py            # Cấu hình model + thông số cắt
├── detection.py         # Logic phát hiện, khử trùng, cắt, tracking
├── gui.py               # Giao diện PyQt5
├── app_state.json       # Lưu phiên làm việc (tự động)
├── start.bat            # Chạy app không hiện terminal
├── restart.bat          # Khởi động lại app
├── yolo26m-pose.pt      # Model chính (YOLO Pose, mới nhất)
├── yolov8m-pose.pt      # Model dự phòng (YOLOv8 Medium Pose)
├── yolov8m.pt           # YOLOv8 Medium (detection thuần)
├── yolov8s.pt           # YOLOv8 Small (nhẹ hơn)
├── yolov8n.pt           # YOLOv8 Nano (nhẹ nhất)
├── rule.png             # Ảnh hướng dẫn quy tắc gán ID
├── PROJECT_EXPLANATION.md   # Giải thích chi tiết dự án
└── README.md            # File này
```

---

## ⚙️ Cấu hình nâng cao

Mở file `config.py` để điều chỉnh:

### Đổi model
```python
_MODEL_NAME = "yolo26m-pose.pt"   # Mặc định: YOLO Pose mới nhất
# Có thể đổi sang:
# "yolov8m-pose.pt" — dự phòng
# "yolov8s-pose.pt" — nhẹ hơn, nhanh hơn
```

### Điều chỉnh ngưỡng phát hiện
```python
_CONF = 0.25        # Tăng lên (0.4-0.5) nếu bị phát hiện nhầm
                    # Giảm xuống (0.15-0.2) nếu bỏ sót người

_NMS_IOU = 0.5      # Giảm nếu có quá nhiều khung trùng
```

### Khử trùng theo Pose
```python
_HEAD_DUP = 0.5     # 2 khung có đầu cách nhau < 0.5 × bề rộng vai → cùng người
```

### Điều chỉnh vùng cắt
```python
_CROWN_FACTOR = 0.65   # Ước lượng đỉnh đầu (tránh cắt lẹm tóc)
_FINGER_FACTOR = 0.60  # Nới vùng cắt khi tay giơ (lấy đủ ngón)
_TORSO_FACTOR = 1.25   # Đáy cắt: dưới vai × hệ số (tới ngực)
_SIDE_MARGIN = 0.20    # Nới hai bên khung cắt
_MARGIN_PX = 7         # Padding pixel cố định
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
- **Khử trùng thông minh**: So sánh khoảng cách đầu — người bị che một phần vẫn được giữ
- **Delta chỉnh tay**: Nếu bạn chỉnh khung ở frame đầu, phần chênh lệch sẽ được áp dụng cho tất cả frame sau
- **Xem frame Đầu/Giữa/Cuối**: Giữ nguyên khung cắt để kiểm tra chất lượng tracking
- **Cache**: Khi chuyển folder trong batch mode, trạng thái ID + khung được lưu lại, quay lại không mất

---

## 📄 License

Dự án sử dụng cho mục đích nghiên cứu và thực tập.

Model YOLO thuộc bản quyền [Ultralytics](https://github.com/ultralytics/ultralytics) — sử dụng theo giấy phép AGPL-3.0.
