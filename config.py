# -*- coding: utf-8 -*-
"""
Cấu hình + nạp model YOLO.
==> Muốn tinh chỉnh nhận diện/cắt thì sửa các thông số TRONG FILE NÀY.
"""

import os

# ----- Model -----
# Model POSE: cho ra điểm khớp cơ thể (đầu/vai/khuỷu/cổ tay/hông) -> cắt nửa thân
# trên chính xác: gồm cả TAY GIƠ, bỏ CHÂN, không cắt vào mặt. (~50MB tải lần đầu)
_MODEL_NAME = "yolov8m-pose.pt"
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), _MODEL_NAME)

# ----- Ngưỡng phát hiện / khử trùng -----
_CONF = 0.25                          # tin cậy người (pose cần đủ cao để điểm khớp đáng tin)
_NMS_IOU = 0.5                        # NMS thấp -> bớt khung chồng (trùng người)
_CONTAIN_THR = 0.85                   # khung nằm trong khung khác > tỉ lệ này...
_DUP_AREA_RATIO = 0.75                # ...VÀ kích thước tương đương -> mới coi là trùng (bỏ).
                                      # Người bị che có khung nhỏ hơn nhiều -> được GIỮ lại.

# ----- Tracking -----
_TRACK_IOU = 0.3                      # ngưỡng IoU để coi là cùng một người giữa 2 frame

# ----- Cắt theo điểm khớp (pose): chân dung ĐẦU + NỬA NGỰC -----
# Mọi khoảng cách đo theo BỀ RỘNG VAI (S) - ổn định kể cả khi người CÚI/CHỒM xuống
# (đo theo "đầu->vai" sẽ co lại khi cúi, gây lẹm đầu / khung quá nhỏ).
_KP_CONF = 0.30                       # ngưỡng tin cậy của từng điểm khớp
# COCO không có điểm "đỉnh đầu" (chỉ mũi/mắt/tai) -> ước lượng đỉnh sọ phía trên
# mắt/tai một khoảng _CROWN_FACTOR x S để KHÔNG lẹm vào tóc/đỉnh đầu.
_CROWN_FACTOR = 0.75                  # đỉnh sọ ~ head_top - 0.75 x bề_rộng_vai
_FINGER_FACTOR = 0.55                 # khi GIƠ TAY: nới trên cổ tay 0.55 x S -> lấy đủ NGÓN TAY
_TORSO_FACTOR = 1.6                   # đáy crop = vai + 1.6 x S (tới ~ngực, không xuống mặt bàn)
_SIDE_MARGIN = 0.22                   # nới hai bên (theo bề rộng vùng khớp)
_MARGIN_PX = 8                        # nới thêm cố định (pixel)
# Khi thiếu điểm khớp đầu/vai (bị che nặng) -> fallback: dùng khung bbox + nới _PAD
_PAD = 0.12

# ----- Khác -----
_IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
# Thư mục mặc định khi mở hộp thoại chọn folder (cho dễ chọn nhanh)
_DEFAULT_DIR = r"C:\Users\Admin\Desktop\2025.2\thuctap\video\savevideo"

_yolo_model = None

def get_model():
    """Tải/khởi tạo model YOLO một lần và tái sử dụng."""
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO(_MODEL_PATH if os.path.exists(_MODEL_PATH) else _MODEL_NAME)
    return _yolo_model
