# -*- coding: utf-8 -*-
"""
Phát hiện người, khử trùng, cắt khung, tracking và liệt kê file.
(Phần xử lý ảnh - không dính giao diện.)
"""

import os
import re
import glob
import cv2

from config import (
    get_model, _CONF, _NMS_IOU, _CONTAIN_THR, _DUP_AREA_RATIO, _TRACK_IOU,
    _KP_CONF, _CROWN_FACTOR, _FINGER_FACTOR, _TORSO_FACTOR,
    _SIDE_MARGIN, _MARGIN_PX, _PAD, _IMG_EXTS,
)

# Điểm khớp COCO: 0 mũi, 1-2 mắt, 3-4 tai, 5-6 vai, 7-8 khuỷu, 9-10 cổ tay,
#                 11-12 hông, 13-16 gối/cổ chân (KHÔNG dùng -> để bỏ chân).

# ===== TIỆN ÍCH HÌNH HỌC =====
def _iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)

def _contain_ratio(a, b):
    """Tỉ lệ phần giao so với khung NHỎ hơn (1.0 = khung nhỏ nằm trọn trong khung kia)."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / min(area_a, area_b)

# ===== PHÁT HIỆN (POSE) / CẮT =====
def detect(image):
    """
    Phát hiện người bằng model POSE.
    Trả về (boxes, kpts) là 2 danh sách SONG SONG (đã khử trùng):
      - boxes[i] = [x1,y1,x2,y2] khung người
      - kpts[i]  = mảng (17,3) [x,y,conf] điểm khớp (hoặc None nếu không có)
    """
    model = get_model()
    results = model(image, conf=_CONF, iou=_NMS_IOU, verbose=False)[0]
    if results.boxes is None or len(results.boxes) == 0:
        return [], []

    boxes = [list(b) for b in results.boxes.xyxy.cpu().numpy().tolist()]
    if results.keypoints is not None and results.keypoints.data is not None:
        kdata = results.keypoints.data.cpu().numpy()
        kpts = [kdata[i] for i in range(len(boxes))]
    else:
        kpts = [None] * len(boxes)

    def area(x):
        return (x[2] - x[0]) * (x[3] - x[1])

    # Khung lớn xét trước; bỏ khung nằm gần trọn trong khung đã giữ VÀ cỡ tương đương (trùng người)
    order = sorted(range(len(boxes)), key=lambda i: area(boxes[i]), reverse=True)
    keep_b, keep_k = [], []
    for i in order:
        b = boxes[i]
        dup = any(_contain_ratio(b, k) > _CONTAIN_THR and area(b) / area(k) > _DUP_AREA_RATIO
                  for k in keep_b)
        if not dup:
            keep_b.append(b); keep_k.append(kpts[i])
    return keep_b, keep_k

def order_boxes(boxes):
    """Sắp xếp trái->phải, trên->xuống để đánh số tạm cho trực quan."""
    return sorted(range(len(boxes)),
                  key=lambda i: (int(boxes[i][1]) // 100, boxes[i][0]))

def _bbox_fallback(box, w, h):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    return (max(0, int(x1 - bw * _PAD)), max(0, int(y1 - bh * _PAD)),
            min(w, int(x2 + bw * _PAD)), min(h, int(y2 + bh * _PAD)))

def crop_region(box, kpts, w, h):
    """
    Vùng cắt CHÂN DUNG: đầu + nửa ngực, gồm cả tay GIƠ (đủ ngón), KHÔNG kéo xuống mặt bàn.
    Mọi khoảng đo theo BỀ RỘNG VAI S (ổn định khi cúi). Thiếu khớp đầu/vai -> fallback bbox.
    """
    if kpts is None:
        return _bbox_fallback(box, w, h)

    P = {j: (kpts[j][0], kpts[j][1]) for j in range(len(kpts)) if kpts[j][2] > _KP_CONF}
    head = [P[j] for j in (0, 1, 2, 3, 4) if j in P]      # mũi/mắt/tai
    shoulders = [P[j] for j in (5, 6) if j in P]
    if not head or not shoulders:
        return _bbox_fallback(box, w, h)

    shoulder_y = sum(p[1] for p in shoulders) / len(shoulders)
    head_top = min(p[1] for p in head)
    # Thước đo = bề rộng vai (fallback theo bề rộng bbox); chặn dưới để không quá nhỏ
    if len(shoulders) == 2:
        S = abs(shoulders[0][0] - shoulders[1][0])
    else:
        S = (box[2] - box[0]) * 0.6
    S = max(S, (box[2] - box[0]) * 0.35, 30.0)

    top_y = head_top - _CROWN_FACTOR * S                  # đỉnh sọ ước lượng (tránh lẹm tóc)
    for j in (9, 10):                                     # cổ tay giơ -> chừa chỗ cho NGÓN TAY
        if j in P:
            top_y = min(top_y, P[j][1] - _FINGER_FACTOR * S)
    for j in (7, 8):                                      # khuỷu giơ cao
        if j in P:
            top_y = min(top_y, P[j][1])
    bottom_y = shoulder_y + _TORSO_FACTOR * S             # dừng quanh ngực

    xs = [p[0] for p in shoulders] + [p[0] for p in head]
    for j in (7, 8, 9, 10):                               # chỉ tính tay PHÍA TRÊN ngực
        if j in P and P[j][1] <= bottom_y:
            xs.append(P[j][0])
    x1, x2 = min(xs), max(xs)
    mx = (x2 - x1) * _SIDE_MARGIN + _MARGIN_PX
    return (max(0, int(x1 - mx)),
            max(0, int(top_y - _MARGIN_PX)),
            min(w, int(x2 + mx)),
            min(h, int(bottom_y + _MARGIN_PX)))

def annotate(image, boxes, labels):
    """Vẽ khung + nhãn lên ảnh."""
    out = image.copy()
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 2)
        (tw, th), _ = cv2.getTextSize(str(label), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(out, (x1, y1 - th - 10), (x1 + tw + 10, y1), (0, 220, 0), -1)
        cv2.putText(out, str(label), (x1 + 5, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return out

# ===== LIỆT KÊ FILE =====
def list_frames(folder):
    """Trả về danh sách đường dẫn frame, sắp xếp theo SỐ trong tên file."""
    files = [f for f in glob.glob(os.path.join(folder, '*'))
             if f.lower().endswith(_IMG_EXTS)]

    def key(p):
        nums = re.findall(r'\d+', os.path.basename(p))
        return (int(nums[0]) if nums else 0, os.path.basename(p))

    return sorted(files, key=key)

def find_clip_folders(parent):
    """Tìm các thư mục con (trực tiếp) có chứa frame ảnh. Bỏ qua thư mục id_*."""
    out = []
    for name in sorted(os.listdir(parent)):
        sub = os.path.join(parent, name)
        if not os.path.isdir(sub) or name.lower().startswith('id_'):
            continue
        if list_frames(sub):
            out.append(sub)
    # nếu chính thư mục cha cũng chứa frame thì thêm vào đầu
    if list_frames(parent):
        out.insert(0, parent)
    return out

# ===== TRACKER NHẸ (greedy IoU) =====
class IoUTracker:
    """Bám người qua các frame bằng IoU. Khởi tạo id từ frame đầu."""
    def __init__(self, seed_boxes):
        self.active = {i + 1: list(b) for i, b in enumerate(seed_boxes)}  # id 1..K
        self.next_id = len(seed_boxes) + 1

    def update(self, dets):
        """dets: list khung frame mới. Trả về dict {chỉ_số_det: track_id}."""
        pairs = []
        for di, d in enumerate(dets):
            for tid, tb in self.active.items():
                io = _iou(d, tb)
                if io >= _TRACK_IOU:
                    pairs.append((io, di, tid))
        pairs.sort(reverse=True)
        used_d, used_t, matches = set(), set(), {}
        for io, di, tid in pairs:
            if di in used_d or tid in used_t:
                continue
            used_d.add(di); used_t.add(tid)
            matches[di] = tid
            self.active[tid] = list(dets[di])
        # det chưa khớp -> track mới
        for di, d in enumerate(dets):
            if di not in used_d:
                tid = self.next_id; self.next_id += 1
                matches[di] = tid
                self.active[tid] = list(d)
        return matches
