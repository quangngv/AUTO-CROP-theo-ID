# -*- coding: utf-8 -*-
"""
Cấu hình + nạp model YOLO.
==> Muốn tinh chỉnh nhận diện/cắt thì sửa các thông số TRONG FILE NÀY.
"""

import os
import threading

# ----- Model -----
# Model POSE: cho ra điểm khớp cơ thể (đầu/vai/khuỷu/cổ tay/hông) -> cắt nửa thân
# trên chính xác: gồm cả TAY GIƠ, bỏ CHÂN, không cắt vào mặt. (~50MB tải lần đầu)
_MODEL_NAME = "yolo26m-pose.pt"
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), _MODEL_NAME)

# ----- Ngưỡng phát hiện / khử trùng -----
_CONF = 0.25                          # tin cậy người (pose cần đủ cao để điểm khớp đáng tin)
_NMS_IOU = 0.5                        # NMS thấp -> bớt khung chồng (trùng người)
_CONTAIN_THR = 0.85                   # khung nằm trong khung khác > tỉ lệ này...
_DUP_AREA_RATIO = 0.75                # ...VÀ kích thước tương đương -> mới coi là trùng (bỏ).
                                      # Người bị che có khung nhỏ hơn nhiều -> được GIỮ lại.
# Khử trùng theo POSE: 2 khung có ĐẦU cách nhau < _HEAD_DUP x bề rộng vai => cùng 1 người.
# (đo thực tế: trùng ~0.01, hai người khác nhau ~0.9 -> 0.5 tách sạch)
_HEAD_DUP = 0.5

# ----- Tracking -----
_TRACK_IOU = 0.3                      # ngưỡng IoU để coi là cùng một người giữa 2 frame

# ----- Đánh số -----
_NUM_ROWS = 3                         # số HÀNG ghế trong phòng (đánh số dưới->trên, trái->phải)
# Bỏ người bị CẮT ở rìa TRÁI khung hình (giám sát/qua đường, nửa người ngoài khung).
# Khung x1 <= _DROP_EDGE_PX px -> bỏ. 0 = TẮT. (cũng là biên "chạm mép" cho rìa phải bên dưới)
_DROP_EDGE_PX = 5
# Bỏ người bị CẮT ở rìa PHẢI: khung CHẠM mép phải (x2 >= W - _DROP_EDGE_PX) VÀ chỉ ló một
# DẢI HẸP (bề rộng < _DROP_RIGHT_MAXW px) -> người ngồi ngoài rìa, hầu hết thân ngoài khung
# (vd "người số 12"). Người sát mép phải nhưng ĐỦ THÂN (bề rộng >= ngưỡng) vẫn GIỮ. 0 = TẮT.
_DROP_RIGHT_MAXW = 170

# ----- Cắt theo điểm khớp (pose): chân dung ĐẦU + NỬA NGỰC -----
# Mọi khoảng cách đo theo BỀ RỘNG VAI (S) - ổn định kể cả khi người CÚI/CHỒM xuống
# (đo theo "đầu->vai" sẽ co lại khi cúi, gây lẹm đầu / khung quá nhỏ).
_KP_CONF = 0.30                       # ngưỡng tin cậy của từng điểm khớp
# COCO không có điểm "đỉnh đầu" (chỉ mũi/mắt/tai) -> ước lượng đỉnh sọ phía trên
# mắt/tai một khoảng _CROWN_FACTOR x S để KHÔNG lẹm vào tóc/đỉnh đầu.
_CROWN_FACTOR = 0.60                  # đỉnh sọ ~ head_top - 0.75 x bề_rộng_vai
_FINGER_FACTOR = 0.65                 # khi GIƠ TAY: nới trên cổ tay 0.55 x S -> lấy đủ NGÓN TAY
_TORSO_FACTOR = 1.25                   # đáy crop = vai + 1.6 x S (tới ~ngực, không xuống mặt bàn)
_SIDE_MARGIN = 0.20                  # nới hai bên (theo bề rộng vùng khớp)
_MARGIN_PX = 5                        # nới thêm cố định (pixel)
# Khi thiếu điểm khớp đầu/vai (bị che nặng) -> fallback: dùng khung bbox + nới _PAD
_PAD = 0.12

# ----- Khác -----
_IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
# Thư mục mặc định khi mở hộp thoại chọn folder (cho dễ chọn nhanh)
_DEFAULT_DIR = r"C:\Users\Admin\Desktop\2025.2\thuctap\video\savevideo"

_yolo_model = None
_model_lock = threading.Lock()
_infer_lock = threading.Lock()

def get_model():
    """Tải/khởi tạo model YOLO một lần và tái sử dụng (an toàn đa luồng)."""
    global _yolo_model
    if _yolo_model is None:
        with _model_lock:                      # khoá: tránh 2 luồng cùng nạp model 2 lần
            if _yolo_model is None:
                from ultralytics import YOLO
                _yolo_model = YOLO(_MODEL_PATH if os.path.exists(_MODEL_PATH) else _MODEL_NAME)
    return _yolo_model

def predict(image, **kwargs):
    """Chạy YOLO có KHOÁ tuần tự: ultralytics dùng chung 1 predictor nội bộ nên 2 lời
    gọi song song (warmup nền + detect GUI) có thể hỏng. Khoá -> an toàn, gần như 0 chi phí."""
    model = get_model()
    with _infer_lock:
        return model(image, **kwargs)

def warmup():
    """Nạp model + chạy 1 lần inference giả để 'làm nóng' (import nặng + cấp phát GPU).
    Gọi ở luồng nền lúc mở app -> khi nạp ảnh thật chỉ còn ~0.02s thay vì ~6s."""
    import numpy as np
    predict(np.zeros((64, 64, 3), dtype=np.uint8), conf=_CONF, iou=_NMS_IOU, verbose=False)
