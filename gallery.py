# -*- coding: utf-8 -*-
"""
Bộ nhớ id theo NGOẠI HÌNH (appearance): trích đặc trưng ảnh cắt mỗi người bằng
ResNet50 (ImageNet) rồi so khớp cosine. Dùng để GỢI Ý id bán tự động.
"""

import os
import json
import numpy as np
import cv2

_model = None
_dev = None
_GALLERY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gallery.json")
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _get_model():
    global _model, _dev
    if _model is None:
        import torch
        from torchvision.models import resnet50, ResNet50_Weights
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        m.fc = torch.nn.Identity()      # lấy vector đặc trưng 2048-d
        m.eval()
        _dev = "cuda" if torch.cuda.is_available() else "cpu"
        m.to(_dev)
        _model = m
    return _model, _dev


# Trọng số kết hợp: đặc trưng CNN (dáng/kết cấu) + màu ÁO (bất biến tỉ lệ, mạnh cùng buổi)
_W_CNN = 0.45
_W_COLOR = 0.55


def _cnn_feat(crop_bgr):
    import torch
    m, dev = _get_model()
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (128, 256)).astype(np.float32) / 255.0   # (W=128,H=256)
    rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
    x = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(dev)
    with torch.no_grad():
        f = m(x)[0].detach().cpu().numpy().astype(np.float32)
    n = np.linalg.norm(f)
    return f / n if n > 0 else f


def _color_feat(crop_bgr):
    """Histogram màu (H,S) vùng THÂN/áo (giữa-dưới khung) -> bất biến tỉ lệ, lọc bớt nền."""
    h, w = crop_bgr.shape[:2]
    torso = crop_bgr[int(h * 0.32):int(h * 0.98), int(w * 0.18):int(w * 0.82)]
    if torso.size == 0:
        torso = crop_bgr
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [18, 18], [0, 180, 0, 256]).flatten()
    hist = np.sqrt(hist / (hist.sum() + 1e-6)).astype(np.float32)   # Hellinger
    n = np.linalg.norm(hist)
    return hist / n if n > 0 else hist


def embed(crop_bgr):
    """Ảnh cắt (BGR) -> vector đặc trưng kết hợp CNN + màu áo (đã chuẩn hoá). None nếu rỗng."""
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    f_cnn = _cnn_feat(crop_bgr) * np.float32(np.sqrt(_W_CNN))
    f_col = _color_feat(crop_bgr) * np.float32(np.sqrt(_W_COLOR))
    return np.concatenate([f_cnn, f_col]).astype(np.float32)        # dot = w_cnn*cos + w_col*cos


def match(emb, gallery):
    """emb -> (id, điểm cosine) khớp nhất trong gallery {id: [vector,...]}."""
    best_id, best = None, -1.0
    if emb is None:
        return None, -1.0
    for idv, vecs in gallery.items():
        for v in vecs:
            s = float(np.dot(emb, v))
            if s > best:
                best, best_id = s, idv
    return best_id, best


def save_gallery(gallery, path=None):
    path = path or _GALLERY_PATH
    data = {idv: [v.tolist() for v in vecs] for idv, vecs in gallery.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def load_gallery(path=None):
    path = path or _GALLERY_PATH
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    return {idv: [np.array(v, dtype=np.float32) for v in vecs] for idv, vecs in data.items()}
