# -*- coding: utf-8 -*-
"""Giao diện PyQt5 cho AUTO CROP theo ID."""

import os
import re
import json
import math
import threading
import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from detection import (
    detect, order_boxes, crop_region,
    list_frames, IoUTracker,
)
from config import _DEFAULT_DIR
import gallery

def _start_dir(current):
    """Thư mục mở đầu cho hộp thoại: đường dẫn đang nhập -> mặc định -> rỗng."""
    if current:
        return current
    return _DEFAULT_DIR if os.path.isdir(_DEFAULT_DIR) else ""

# ===== WIDGET HIỂN THỊ ẢNH + RÊ CHỈNH KHUNG CẮT =====
class FrameEditor(QWidget):
    """Hiển thị frame và cho RÊ cạnh/góc/di chuyển từng khung cắt theo x, y tùy ý."""
    HANDLE = 9        # bán kính vùng bắt cạnh/góc (pixel màn hình)
    MIN_SIZE = 10     # kích thước tối thiểu của khung (pixel ảnh gốc)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(500, 360)
        self.setMouseTracking(True)
        self._img = None          # QImage gốc
        self.boxes = []           # [[x1,y1,x2,y2], ...] theo TOẠ ĐỘ ẢNH GỐC
        self.labels = []
        self.selected = None      # chỉ số khung ĐANG CHỌN (chỉ khung này mới sửa được)
        self.on_select = None     # callback(idx) khi đổi khung chọn (để đồng bộ danh sách)
        self._drag = None         # (index, mode)  mode: move/l/r/t/b/tl/tr/bl/br
        self._last = None         # vị trí chuột trước (toạ độ ảnh gốc)
        self._hint = "Chọn folder rồi nhấn 'Tải & phát hiện'"
        self._zoom = 1.0          # 1.0 = vừa khung; >1 = phóng to
        self._panx = 0.0          # dịch ảnh (pixel màn hình)
        self._pany = 0.0
        self._panning = False
        self._pan_last = None
        self._nav = False         # chế độ di chuyển/zoom (tắt = ảnh cố định, rê khung)
        self._draw_mode = False   # chế độ VẼ khung mới (model bỏ sót -> tự khoanh tay)
        self._draw_start = None   # điểm bắt đầu vẽ (toạ độ ảnh gốc)
        self._draw_cur = None     # điểm hiện tại khi đang kéo
        self.on_new_box = None    # callback(box) khi vẽ xong 1 khung mới

    def has_image(self):
        return self._img is not None

    def set_draw_mode(self, on):
        """Bật chế độ VẼ: kéo chuột trên ảnh để tạo 1 khung thủ công (1 lần, xong tự tắt)."""
        self._draw_mode = bool(on)
        self._draw_start = self._draw_cur = None
        if on:
            self.set_nav(False)               # tắt di chuyển để vẽ được
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
        self.update()

    def add_box(self, box, label):
        """Thêm 1 khung mới (vẽ tay) vào danh sách hiển thị, chọn luôn khung đó."""
        self.boxes.append([int(v) for v in box])
        self.labels.append(str(label))
        self.selected = len(self.boxes) - 1
        self.update()
        return self.selected

    def set_nav(self, on):
        """Bật/tắt chế độ di chuyển: bật mới được zoom+kéo ảnh; tắt thì cố định để chỉnh khung."""
        self._nav = bool(on)
        self._panning = False
        self.setCursor(Qt.OpenHandCursor if self._nav else Qt.ArrowCursor)

    # --- dữ liệu ---
    def set_frame(self, bgr):
        if bgr is None:
            self._img = None; self.boxes = []; self.labels = []; self.update(); return
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        h, w, _ = rgb.shape
        self._img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self.update()

    def set_boxes(self, boxes, labels):
        self.boxes = [[int(v) for v in b] for b in boxes]
        self.labels = list(labels)
        self.selected = 0 if self.boxes else None
        self.reset_view()                 # folder mới -> về vừa khung

    def reset_view(self):
        self._zoom = 1.0; self._panx = 0.0; self._pany = 0.0
        self.update()

    def get_boxes(self):
        return [list(b) for b in self.boxes]

    def select(self, idx):
        """Chọn khung theo chỉ số (gọi từ danh sách id bên phải)."""
        if idx is not None and 0 <= idx < len(self.boxes):
            self.selected = idx
            self.update()

    # --- ánh xạ toạ độ màn hình <-> ảnh gốc (vừa khung + zoom + dịch) ---
    def _geom(self):
        if self._img is None:
            return 1.0, 0.0, 0.0
        W, H = self.width(), self.height()
        w, h = self._img.width(), self._img.height()
        s = min(W / w, H / h) * self._zoom
        ox = (W - w * s) / 2 + self._panx
        oy = (H - h * s) / 2 + self._pany
        return s, ox, oy

    def zoom_at(self, factor, mx, my):
        """Phóng to/thu nhỏ quanh điểm (mx,my) trên màn hình."""
        if self._img is None:
            return
        ix, iy = self._to_img(QPoint(int(mx), int(my)))
        self._zoom = max(1.0, min(self._zoom * factor, 12.0))
        if self._zoom <= 1.0:                         # về vừa khung -> bỏ dịch
            self._panx = self._pany = 0.0
            self.update(); return
        W, H = self.width(), self.height()
        w, h = self._img.width(), self._img.height()
        s = min(W / w, H / h) * self._zoom
        self._panx = mx - (ix * s + (W - w * s) / 2)
        self._pany = my - (iy * s + (H - h * s) / 2)
        self.update()

    def wheelEvent(self, e):
        if self._img is None or not self._nav:    # chỉ zoom khi bật chế độ di chuyển
            super().wheelEvent(e); return
        p = e.pos()
        self.zoom_at(1.25 if e.angleDelta().y() > 0 else 0.8, p.x(), p.y())
        e.accept()

    def _to_img(self, p):
        s, ox, oy = self._geom()
        return (p.x() - ox) / s, (p.y() - oy) / s

    def _to_wid(self, x, y):
        s, ox, oy = self._geom()
        return QPoint(int(x * s + ox), int(y * s + oy))

    # --- vẽ ---
    def paintEvent(self, e):
        qp = QPainter(self)
        qp.fillRect(self.rect(), QColor("#2b2b2b"))
        if self._img is None:
            qp.setPen(QColor("#aaa"))
            qp.drawText(self.rect(), Qt.AlignCenter, self._hint)
            qp.end(); return
        s, ox, oy = self._geom()
        qp.drawImage(QRectF(ox, oy, self._img.width() * s, self._img.height() * s), self._img)
        f = qp.font(); f.setBold(True); f.setPointSize(11); qp.setFont(f)
        # vẽ khung KHÔNG chọn trước (mờ), khung đang chọn vẽ sau cùng (nổi lên trên)
        order = [i for i in range(len(self.boxes)) if i != self.selected]
        if self.selected is not None and 0 <= self.selected < len(self.boxes):
            order.append(self.selected)
        for i in order:
            b = self.boxes[i]
            r = QRect(self._to_wid(b[0], b[1]), self._to_wid(b[2], b[3])).normalized()
            sel = (i == self.selected)
            color = QColor(255, 210, 0) if sel else QColor(0, 200, 0, 130)
            qp.setPen(QPen(color, 3 if sel else 1)); qp.setBrush(Qt.NoBrush)
            qp.drawRect(r)
            if sel:  # chỉ khung đang chọn mới hiện tay cầm
                qp.setBrush(color)
                cx, cy = (r.left() + r.right()) // 2, (r.top() + r.bottom()) // 2
                for hx, hy in [(r.left(), r.top()), (r.right(), r.top()), (r.left(), r.bottom()),
                               (r.right(), r.bottom()), (cx, r.top()), (cx, r.bottom()),
                               (r.left(), cy), (r.right(), cy)]:
                    qp.drawRect(hx - 4, hy - 4, 8, 8)
                qp.setBrush(Qt.NoBrush)
            lbl = str(self.labels[i]) if i < len(self.labels) else str(i + 1)
            qp.setPen(QColor(255, 210, 0) if sel else QColor(255, 90, 90))
            qp.drawText(r.left() + 4, r.top() + 17, lbl)
        # khung đang VẼ (kéo chuột) -> viền xanh nét đứt
        if self._draw_mode and self._draw_start is not None and self._draw_cur is not None:
            r = QRect(self._to_wid(*self._draw_start), self._to_wid(*self._draw_cur)).normalized()
            qp.setPen(QPen(QColor(0, 200, 255), 2, Qt.DashLine)); qp.setBrush(Qt.NoBrush)
            qp.drawRect(r)
        qp.end()

    # --- xác định cạnh/góc dưới con trỏ ---
    def _hit_box(self, i, p):
        """Mode (cạnh/góc/move) nếu con trỏ p trúng khung i, ngược lại None."""
        b = self.boxes[i]
        p1 = self._to_wid(b[0], b[1]); p2 = self._to_wid(b[2], b[3])
        L, R = min(p1.x(), p2.x()), max(p1.x(), p2.x())
        T, B = min(p1.y(), p2.y()), max(p1.y(), p2.y())
        if not (L - self.HANDLE <= p.x() <= R + self.HANDLE and
                T - self.HANDLE <= p.y() <= B + self.HANDLE):
            return None
        nl = abs(p.x() - L) <= self.HANDLE; nr = abs(p.x() - R) <= self.HANDLE
        nt = abs(p.y() - T) <= self.HANDLE; nb = abs(p.y() - B) <= self.HANDLE
        if nt and nl: return "tl"
        if nt and nr: return "tr"
        if nb and nl: return "bl"
        if nb and nr: return "br"
        if nl: return "l"
        if nr: return "r"
        if nt: return "t"
        if nb: return "b"
        if L < p.x() < R and T < p.y() < B: return "move"
        return None

    def _topmost(self, p):
        """Chỉ số khung trên cùng nằm dưới con trỏ (để CHỌN)."""
        for i in range(len(self.boxes) - 1, -1, -1):
            if self._hit_box(i, p) is not None:
                return i
        return None

    def _cursor_for(self, hit):
        if hit is None:
            return Qt.ArrowCursor
        m = hit[1]
        return {"l": Qt.SizeHorCursor, "r": Qt.SizeHorCursor,
                "t": Qt.SizeVerCursor, "b": Qt.SizeVerCursor,
                "tl": Qt.SizeFDiagCursor, "br": Qt.SizeFDiagCursor,
                "tr": Qt.SizeBDiagCursor, "bl": Qt.SizeBDiagCursor,
                "move": Qt.SizeAllCursor}.get(m, Qt.ArrowCursor)

    def mousePressEvent(self, e):
        if self._img is None:
            return
        # CHẾ ĐỘ VẼ: bắt đầu khoanh 1 khung mới (chỉ chuột trái)
        if self._draw_mode:
            if e.button() == Qt.LeftButton:
                self._draw_start = self._draw_cur = self._to_img(e.pos())
                self.update()
            return
        # CHẾ ĐỘ DI CHUYỂN: mọi kéo = di chuyển ảnh (không đụng khung)
        if self._nav or e.button() == Qt.MiddleButton:
            if e.button() in (Qt.LeftButton, Qt.MiddleButton):
                self._panning = True; self._pan_last = e.pos()
                self.setCursor(Qt.ClosedHandCursor)
            return
        if e.button() != Qt.LeftButton:
            return
        # CHẾ ĐỘ CHỈNH KHUNG (ảnh cố định)
        # 1) Ưu tiên khung ĐANG CHỌN (kể cả khi bị khung khác đè) -> sửa luôn
        if self.selected is not None:
            mode = self._hit_box(self.selected, e.pos())
            if mode:
                self._drag = (self.selected, mode)
                self._last = self._to_img(e.pos())
                return
        # 2) Bấm vào khung khác -> chọn khung đó rồi bắt đầu sửa
        idx = self._topmost(e.pos())
        if idx is not None:
            self.selected = idx
            if self.on_select:
                self.on_select(idx)
            self._drag = (idx, self._hit_box(idx, e.pos()) or "move")
            self._last = self._to_img(e.pos())
            self.update()

    def mouseMoveEvent(self, e):
        if self._img is None:
            return
        if self._draw_mode:
            if self._draw_start is not None:
                self._draw_cur = self._to_img(e.pos())
                self.update()
            return
        if self._panning:
            d = e.pos() - self._pan_last
            self._panx += d.x(); self._pany += d.y()
            self._pan_last = e.pos(); self.update(); return
        if self._nav:
            return
        if self._drag is None:
            mode = None
            if self.selected is not None:
                mode = self._hit_box(self.selected, e.pos())
            self.setCursor(self._cursor_for((0, mode) if mode else None))
            return
        i, mode = self._drag
        ix, iy = self._to_img(e.pos()); lx, ly = self._last
        dx, dy = ix - lx, iy - ly
        b = self.boxes[i]
        W, H = self._img.width(), self._img.height()
        if mode == "move":
            b[0] += dx; b[2] += dx; b[1] += dy; b[3] += dy
        else:
            if "l" in mode: b[0] += dx
            if "r" in mode: b[2] += dx
            if "t" in mode: b[1] += dy
            if "b" in mode: b[3] += dy
        # giữ thứ tự + kích thước tối thiểu + trong ảnh
        b[0], b[2] = min(b[0], b[2] - self.MIN_SIZE), max(b[2], b[0] + self.MIN_SIZE)
        b[1], b[3] = min(b[1], b[3] - self.MIN_SIZE), max(b[3], b[1] + self.MIN_SIZE)
        b[0] = max(0, min(b[0], W)); b[2] = max(0, min(b[2], W))
        b[1] = max(0, min(b[1], H)); b[3] = max(0, min(b[3], H))
        self.boxes[i] = [int(b[0]), int(b[1]), int(b[2]), int(b[3])]
        self._last = (ix, iy)
        self.update()

    def mouseReleaseEvent(self, e):
        # CHẾ ĐỘ VẼ: kết thúc kéo -> tạo khung (nếu đủ to) rồi tự tắt chế độ vẽ
        if self._draw_mode:
            if self._img is not None and self._draw_start is not None and self._draw_cur is not None:
                (x1, y1), (x2, y2) = self._draw_start, self._draw_cur
                W, H = self._img.width(), self._img.height()
                box = [max(0, min(int(min(x1, x2)), W)), max(0, min(int(min(y1, y2)), H)),
                       max(0, min(int(max(x1, x2)), W)), max(0, min(int(max(y1, y2)), H))]
                self._draw_start = self._draw_cur = None
                if (box[2] - box[0] >= self.MIN_SIZE and box[3] - box[1] >= self.MIN_SIZE
                        and self.on_new_box):
                    self.on_new_box(box)        # báo MainWindow thêm khung + ô id
            self.set_draw_mode(False)
            return
        self._drag = None
        if self._panning:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)

# ===== HỘP THOẠI XEM ẢNH PHÓNG TO (xem nhanh ảnh đã cắt, lật trước/sau, xoá ảnh hỏng) =====
class ImageViewerDialog(QDialog):
    """Xem 1 ảnh phóng to trong danh sách; ← → lật ảnh; 🗑 xoá ảnh hỏng ngay tại chỗ.
    Ảnh đã xoá ghi vào self.deleted để cửa sổ cha làm mới lưới + số đếm sau khi đóng."""
    def __init__(self, paths, index, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Xem ảnh đã cắt")
        self.resize(680, 780)
        self.setStyleSheet("QDialog{background:#1e1e1e;} QLabel{color:#ddd;}"
                           "QPushButton{background:#3a6ea5;color:#fff;border:none;padding:8px 14px;"
                           "border-radius:6px;font-weight:bold;} QPushButton:hover{background:#4a7eb5;}")
        self.paths = list(paths)
        self.index = index
        self.deleted = []                 # ảnh đã xoá trong phiên xem
        lay = QVBoxLayout(self)
        self.pic = QLabel(); self.pic.setAlignment(Qt.AlignCenter); self.pic.setMinimumSize(400, 520)
        lay.addWidget(self.pic, 1)
        self.cap = QLabel(); self.cap.setAlignment(Qt.AlignCenter); self.cap.setWordWrap(True)
        lay.addWidget(self.cap)
        row = QHBoxLayout()
        b_prev = QPushButton("← Trước"); b_prev.clicked.connect(lambda: self._step(-1))
        b_next = QPushButton("Sau →"); b_next.clicked.connect(lambda: self._step(1))
        b_del = QPushButton("🗑 Xoá ảnh này"); b_del.setStyleSheet("background:#a33;")
        b_del.clicked.connect(self._delete)
        row.addWidget(b_prev); row.addWidget(b_next); row.addStretch(); row.addWidget(b_del)
        lay.addLayout(row)
        self._show_cur()

    def _show_cur(self):
        if not self.paths:
            self.pic.setText("(Không còn ảnh)"); self.pic.setPixmap(QPixmap()); self.cap.setText(""); return
        self.index = max(0, min(self.index, len(self.paths) - 1))
        p = self.paths[self.index]
        pm = QPixmap(p)
        if not pm.isNull():
            self.pic.setPixmap(pm.scaled(self.pic.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.pic.setText("(Không đọc được ảnh)")
        self.cap.setText(f"{self.index + 1}/{len(self.paths)} — {os.path.basename(p)}")

    def _step(self, d):
        if self.paths:
            self.index = (self.index + d) % len(self.paths)
            self._show_cur()

    def _delete(self):
        if not self.paths:
            return
        p = self.paths[self.index]
        if QMessageBox.question(self, "Xoá ảnh", f"Xoá ảnh này?\n{os.path.basename(p)}",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        try:
            os.remove(p)
        except OSError as e:
            QMessageBox.warning(self, "Lỗi", f"Không xoá được: {e}"); return
        self.deleted.append(p)
        del self.paths[self.index]
        self._show_cur()

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Left,): self._step(-1)
        elif e.key() in (Qt.Key_Right,): self._step(1)
        elif e.key() in (Qt.Key_Delete,): self._delete()
        else: super().keyPressEvent(e)

    def resizeEvent(self, e):
        super().resizeEvent(e); self._show_cur()

# ===== CỬA SỔ CHÍNH =====
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AUTO CROP theo ID - Tạo dataset từng người")
        self.setGeometry(80, 60, 1150, 760)
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #1e1e1e; color: #ddd; }
            QPushButton { background:#3a6ea5; color:white; border:none; padding:8px 14px;
                          border-radius:6px; font-size:13px; font-weight:bold; }
            QPushButton:hover { background:#4a7eb5; }
            QPushButton:disabled { background:#555; color:#999; }
            QPushButton#go { background:#28a745; }
            QPushButton#go:hover { background:#38b755; }
            QLineEdit { background:#2b2b2b; border:1px solid #555; border-radius:5px; padding:6px; color:#eee; }
            QComboBox { background:#2b2b2b; border:1px solid #555; border-radius:5px; padding:3px 4px; color:#eee; }
            QComboBox QAbstractItemView { background:#2b2b2b; color:#eee; selection-background-color:#3a6ea5; }
            QLabel#hdr { color:#fff; font-size:18px; font-weight:bold; }
            QGroupBox { border:1px solid #555; border-radius:8px; margin-top:8px; font-weight:bold; }
            QGroupBox::title { subcontrol-origin: margin; left:10px; padding:0 5px; }
            QScrollArea { border:none; }
            QProgressBar { border:1px solid #555; border-radius:5px; text-align:center; background:#2b2b2b; }
            QProgressBar::chunk { background:#28a745; border-radius:5px; }
        """)

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(
            "QTabBar::tab{background:#2b2b2b;color:#ccc;padding:8px 18px;margin-right:2px;}"
            "QTabBar::tab:selected{background:#3a6ea5;color:#fff;}")
        self.setCentralWidget(self.tabs)

        work = QWidget()
        root = QVBoxLayout(work); root.setContentsMargins(16, 16, 16, 16); root.setSpacing(10)

        hdr = QLabel("✂️ AUTO CROP theo ID — cắt từng người qua mọi frame")
        hdr.setObjectName("hdr"); root.addWidget(hdr)

        # Hàng chọn folder
        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("Thư mục frame:"))
        self.path_edit = QLineEdit(); self.path_edit.setPlaceholderText(r"VD: C:\...\savevideo\1781787519_1781787522_1")
        path_row.addWidget(self.path_edit, 1)
        btn_browse = QPushButton("📂 Chọn..."); btn_browse.clicked.connect(self.browse_folder)
        path_row.addWidget(btn_browse)
        btn_batch = QPushButton("📁 Chọn nhiều folder..."); btn_batch.clicked.connect(self.browse_batch)
        path_row.addWidget(btn_batch)
        btn_gal = QPushButton("🧠 Bộ nhớ id"); btn_gal.setStyleSheet("background:#6a3aa5;")
        btn_gal.setToolTip("Tạo bộ nhớ id từ ảnh quy tắc để TỰ GỢI Ý id"); btn_gal.clicked.connect(self.build_gallery)
        path_row.addWidget(btn_gal)
        self.btn_load = QPushButton("🔍 Tải & phát hiện"); self.btn_load.clicked.connect(self.load_and_detect)
        path_row.addWidget(self.btn_load)
        root.addLayout(path_row)

        # Cột thumbnail các folder đã chọn (bên phải; bấm để mở; ✓ = đã xong)
        self.queue_list = QListWidget()
        self.queue_list.setViewMode(QListView.IconMode)
        self.queue_list.setIconSize(QSize(150, 90))
        self.queue_list.setFlow(QListView.TopToBottom)
        self.queue_list.setWrapping(False)
        self.queue_list.setResizeMode(QListView.Adjust)
        self.queue_list.setMovement(QListView.Static)
        self.queue_list.setSpacing(6)
        self.queue_list.setStyleSheet(
            "QListWidget{background:#262626;border:1px solid #555;border-radius:6px;}"
            "QListWidget::item{color:#ccc;}"
            "QListWidget::item:selected{background:#3a6ea5;color:#fff;border-radius:4px;}")
        self.queue_list.itemClicked.connect(self._on_queue_clicked)
        self.queue_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.queue_list.customContextMenuRequested.connect(self._queue_menu)
        self.queue_list.installEventFilter(self)   # phím Delete để bỏ folder

        # Khu chính: 3 phần ngăn bởi VẠCH KÉO ĐƯỢC (QSplitter) -> tự chỉnh bề rộng
        main = QSplitter(Qt.Horizontal)
        main.setChildrenCollapsible(False)
        main.setHandleWidth(8)
        main.setStyleSheet("QSplitter::handle{background:#444;border-radius:3px;margin:2px;}"
                           "QSplitter::handle:hover{background:#3a6ea5;}")

        # Cột folder — bên TRÁI, chỉ hiện khi chọn nhiều folder
        self.queue_group = QGroupBox("📁 Folder đã chọn")
        ql = QVBoxLayout(); ql.addWidget(self.queue_list)
        self.queue_group.setLayout(ql)
        self.queue_group.setMinimumWidth(120)
        self.queue_group.hide()
        main.addWidget(self.queue_group)

        # Ảnh + chỉnh khung — GIỮA, chiếm phần lớn không gian
        left = QGroupBox("Frame — kéo cạnh/góc để chỉnh khung, gõ id bên phải")
        ll = QVBoxLayout()
        self.view = FrameEditor(); ll.addWidget(self.view, 1)

        # Chọn xem khung ở ĐẦU / GIỮA / CUỐI clip để so sánh (chỉ đổi ảnh nền, giữ khung)
        pos_row = QHBoxLayout()
        pos_row.addWidget(QLabel("📍 Xem khung:"))
        self.pos_buttons = {}
        for key, label in [("first", "Đầu"), ("mid", "Giữa"), ("last", "Cuối")]:
            btn = QPushButton(label); btn.setCheckable(True); btn.setFixedWidth(70)
            btn.setStyleSheet("QPushButton{background:#3a3a3a;padding:5px;}"
                              "QPushButton:checked{background:#3a6ea5;color:#fff;}")
            btn.clicked.connect(lambda _=False, k=key: self._show_frame_pos(k))
            self.pos_buttons[key] = btn; pos_row.addWidget(btn)
        pos_row.addSpacing(16)
        self.btn_nav = QPushButton("✋ Di chuyển ảnh"); self.btn_nav.setCheckable(True)
        self.btn_nav.setToolTip("Bật để ZOOM (lăn chuột) + KÉO di chuyển ảnh. Tắt để chỉnh khung.")
        self.btn_nav.setStyleSheet("QPushButton{background:#3a3a3a;padding:5px;}"
                                   "QPushButton:checked{background:#28a745;color:#fff;}")
        self.btn_nav.toggled.connect(self._toggle_nav)
        pos_row.addWidget(self.btn_nav)
        for label, fn in [("🔍－", lambda: self._zoom_btn(0.8)),
                          ("🔍＋", lambda: self._zoom_btn(1.25)),
                          ("Vừa khung", self.view.reset_view)]:
            zb = QPushButton(label); zb.setFixedWidth(54 if "🔍" in label else 90)
            zb.setStyleSheet("QPushButton{background:#3a3a3a;padding:5px;}")
            zb.clicked.connect(fn); pos_row.addWidget(zb)
        pos_row.addStretch()
        ll.addLayout(pos_row)

        tip = QLabel("💡 Mặc định: ảnh CỐ ĐỊNH, rê cạnh/góc để chỉnh khung. Bấm '✋ Di chuyển ảnh' "
                     "để lăn chuột ZOOM + kéo DI CHUYỂN; bấm lại để khoá, chỉnh khung tiếp.")
        tip.setStyleSheet("color:#888; font-weight:normal;"); tip.setWordWrap(True)
        ll.addWidget(tip)
        box_row = QHBoxLayout()
        self.btn_add_box = QPushButton("➕ Thêm đối tượng (vẽ tay)")
        self.btn_add_box.setStyleSheet("background:#8a5a00;")
        self.btn_add_box.setToolTip("Khi model bỏ sót người: bấm nút này rồi KÉO chuột "
                                    "trên ảnh để khoanh 1 khung quanh đối tượng.")
        self.btn_add_box.clicked.connect(self._start_add_box)
        box_row.addWidget(self.btn_add_box)
        self.btn_reset_box = QPushButton("↺ Khôi phục khung tự động")
        self.btn_reset_box.clicked.connect(self.reset_boxes)
        box_row.addWidget(self.btn_reset_box)
        ll.addLayout(box_row)
        left.setLayout(ll); left.setMinimumWidth(360)
        main.addWidget(left)

        # Gán ID — bên PHẢI
        right = QGroupBox("Gán ID (theo bộ quy tắc)")
        right.setMinimumWidth(200)
        rl = QVBoxLayout()
        hint = QLabel("Gõ id cho từng khung. Để TRỐNG = bỏ qua\n(người mới vào / không thuộc quy tắc).")
        hint.setStyleSheet("color:#aaa; font-weight:normal;"); hint.setWordWrap(True)
        rl.addWidget(hint)
        clr_row = QHBoxLayout()
        self.btn_clear_suggest = QPushButton("🧹 Xoá hết id")
        self.btn_clear_suggest.setStyleSheet("background:#555;")
        self.btn_clear_suggest.setToolTip("Xoá HẾT id của mọi khung (cả id gợi ý lẫn id đã gõ tay) "
                                          "để gán lại từ đầu cho dễ.")
        self.btn_clear_suggest.clicked.connect(self._clear_suggestions)
        clr_row.addWidget(self.btn_clear_suggest)
        self.btn_restore_suggest = QPushButton("↩ Khôi phục id gợi ý")
        self.btn_restore_suggest.setStyleSheet("background:#555;")
        self.btn_restore_suggest.setToolTip("Điền lại id theo GỢI Ý (so khớp ngoại hình với bộ nhớ id). "
                                            "Ghi đè cả ô đã xoá / đã sửa tay.")
        self.btn_restore_suggest.clicked.connect(self._restore_suggestions)
        clr_row.addWidget(self.btn_restore_suggest)
        rl.addLayout(clr_row)
        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(True)
        self.scroll_inner = QWidget(); self.form = QVBoxLayout(self.scroll_inner)
        self.form.setAlignment(Qt.AlignTop)
        self.scroll.setWidget(self.scroll_inner)
        rl.addWidget(self.scroll, 1)
        self.btn_export = QPushButton("🚀 Xử lý & xuất tất cả"); self.btn_export.setObjectName("go")
        self.btn_export.clicked.connect(self.export_all); self.btn_export.setEnabled(False)
        rl.addWidget(self.btn_export)
        self.btn_batch = QPushButton("🌐 Gán 1 lần → xuất TẤT CẢ folder")
        self.btn_batch.setStyleSheet("background:#8a5a00;")
        self.btn_batch.clicked.connect(self.batch_all); self.btn_batch.setEnabled(False)
        rl.addWidget(self.btn_batch)
        self.btn_savegal = QPushButton("💾 Lưu bộ nhớ id"); self.btn_savegal.setStyleSheet("background:#6a3aa5;")
        self.btn_savegal.clicked.connect(self.save_gallery_now); self.btn_savegal.setEnabled(False)
        rl.addWidget(self.btn_savegal)
        right.setLayout(rl)
        main.addWidget(right)

        # Bề rộng ban đầu + ưu tiên giãn cho phần ảnh ở giữa
        main.setStretchFactor(0, 0)   # folder
        main.setStretchFactor(1, 1)   # ảnh (giãn)
        main.setStretchFactor(2, 0)   # gán id
        main.setSizes([210, 1100, 270])
        self.splitter = main
        root.addWidget(main, 1)

        self.progress = QProgressBar(); self.progress.setValue(0); root.addWidget(self.progress)

        self.tabs.addTab(work, "✏️ Làm việc")
        self._build_history_tab()
        self._build_results_tab()
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.statusBar().showMessage("Sẵn sàng — chọn thư mục chứa frame")

        # Trạng thái
        self.frames = []
        self.first_boxes = []     # khung người (bbox) frame đầu, theo thứ tự đánh số
        self.first_kpts = []      # điểm khớp tương ứng frame đầu
        self.auto_crops = []      # khung CẮT tự động frame đầu (để tính chỉnh tay)
        self.id_inputs = []       # các QLineEdit tương ứng từng khung
        self.id_buttons = []      # nút chọn khung tương ứng từng khung
        self.queue = []           # danh sách folder khi xử lý nhiều folder
        self.qi = 0               # vị trí folder hiện tại trong hàng đợi
        self.done = set()         # chỉ số folder đã xuất xong (đánh dấu ✓)
        self._cache = {}          # lưu id + khung đã chỉnh theo từng folder (trong phiên)
        self._current_folder = None
        self._frame_pos = "first"   # đang xem frame đầu/giữa/cuối
        self._saved_edits = {}      # {folder: {ids, boxes, done}} -> lưu ra đĩa
        self._gallery = gallery.load_gallery()   # bộ nhớ id theo ngoại hình
        self._gallery_img = None
        self._res_extra = []        # folder xem thêm (ngoài các folder đã chọn) ở tab "Ảnh đã cắt"
        self._res_selected = None   # đường dẫn folder id_ đang xem ảnh
        self._suggest_colors = []   # màu gợi ý từng ô id (None nếu không gợi ý/đã sửa tay)

        # Khôi phục cài đặt cửa sổ (kích thước / phóng to / vạch chia) từ lần trước
        self._settings = QSettings("AutoCropID", "AutoCropID")
        geo = self._settings.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        st = self._settings.value("splitter")
        if st is not None:
            self.splitter.restoreState(st)
        self._want_max = self._settings.value("maximized", False, type=bool)
        self._shown_once = False

        # Làm NÓNG model ở luồng nền NGAY khi mở app (import nặng + cấp phát GPU ~6-10s).
        # Nhờ vậy lần nạp ảnh đầu chỉ còn ~0.02s thay vì đơ máy ~6s. Daemon -> tự thoát theo app.
        self._warm_thread = threading.Thread(target=self._warmup_models, daemon=True)
        self._warm_thread.start()

        # Khôi phục PHIÊN LÀM VIỆC (folder đang làm + chỉnh sửa) sau khi cửa sổ hiện
        QTimer.singleShot(0, self._restore_session)

    @staticmethod
    def _warmup_models():
        """Chạy ở luồng nền: nạp+làm nóng YOLO và ResNet. Lỗi (thiếu GPU/model) -> bỏ qua êm."""
        import config
        try:
            config.warmup()
        except Exception:
            pass
        try:
            gallery.warmup()
        except Exception:
            pass

    def showEvent(self, e):
        super().showEvent(e)
        if self._want_max and not self._shown_once:   # mở lại ở dạng phóng to như lần trước
            self._shown_once = True
            QTimer.singleShot(0, self.showMaximized)

    def closeEvent(self, e):
        # Lưu cài đặt cửa sổ để lần sau mở lại y như cũ
        self._save_current_state()
        self._save_state_to_disk()
        self._settings.setValue("maximized", self.isMaximized())
        if not self.isMaximized():
            self._settings.setValue("geometry", self.saveGeometry())
        self._settings.setValue("splitter", self.splitter.saveState())
        super().closeEvent(e)

    # ---- lưu/khôi phục phiên ra đĩa ----
    def _state_path(self):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_state.json")

    def _save_state_to_disk(self):
        data = {
            "queue": self.queue,
            "qi": self.qi,
            "done": sorted(self.done),
            "current_folder": self._current_folder,
            "edits": self._saved_edits,
            "frame_pos": self._frame_pos,         # vị trí khung đang xem (đầu/giữa/cuối) - nhớ mặc định
        }
        try:
            with open(self._state_path(), "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception:
            pass

    def _restore_session(self):
        """Mở lại đúng folder đang làm trước khi tắt, kèm các chỉnh sửa đã lưu."""
        try:
            with open(self._state_path(), "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return
        self._saved_edits = {k: v for k, v in data.get("edits", {}).items() if os.path.isdir(k)}
        fp = data.get("frame_pos")                 # nhớ vị trí khung (đầu/giữa/cuối) trước khi nạp folder
        if fp in ("first", "mid", "last"):
            self._frame_pos = fp
        self._refresh_history()
        queue = [d for d in data.get("queue", []) if os.path.isdir(d)]
        if queue:
            self.queue = queue
            self.done = set(i for i in data.get("done", []) if i < len(queue))
            self.qi = min(data.get("qi", 0), len(queue) - 1)
            self._build_queue_list()
            self._load_queue_current()
        else:
            cur = data.get("current_folder")
            if cur and os.path.isdir(cur):
                self.path_edit.setText(cur)
                self.load_and_detect()

    # ---- thao tác ----
    def browse_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Chọn thư mục chứa frame", _start_dir(self.path_edit.text()))
        if d:
            self.queue = []
            self.queue_list.clear(); self.queue_group.hide()
            self._res_extra = []; self._res_selected = None   # mở folder mới -> xoá folder cũ ở tab Ảnh đã cắt
            self.path_edit.setText(d)

    def browse_batch(self):
        """Chọn NHIỀU folder cùng lúc (Ctrl/Shift để bôi đen nhiều)."""
        dlg = QFileDialog(self, "Chọn nhiều folder (giữ Ctrl/Shift để chọn nhiều)")
        dlg.setFileMode(QFileDialog.Directory)
        dlg.setOption(QFileDialog.DontUseNativeDialog, True)
        dlg.setOption(QFileDialog.ShowDirsOnly, True)
        cur = self.path_edit.text()
        start = os.path.dirname(cur.rstrip("/\\")) if cur else _start_dir("")
        if start and os.path.isdir(start):
            dlg.setDirectory(start)
        # bật chọn nhiều trên danh sách bên trong hộp thoại
        for view in dlg.findChildren((QListView, QTreeView)):
            view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        if not dlg.exec_():
            return
        picked = [d for d in dlg.selectedFiles() if os.path.isdir(d)]
        # chỉ giữ folder có chứa frame ảnh
        folders = [d for d in picked if list_frames(d)]
        if not folders:
            QMessageBox.warning(self, "Lỗi", "Các folder đã chọn không chứa frame ảnh."); return

        self.queue = folders
        self.qi = 0
        self.done = set()
        self._res_extra = []; self._res_selected = None   # mở bộ folder mới -> xoá folder cũ ở tab Ảnh đã cắt
        self._build_queue_list()
        self._load_queue_current()

    def _build_queue_list(self):
        """Dựng dải thumbnail frame đầu cho các folder trong hàng đợi."""
        self.queue_list.clear()
        for folder in self.queue:
            item = QListWidgetItem(os.path.basename(folder))
            item.setToolTip(folder)
            frames = list_frames(folder)
            if frames:
                img = cv2.imread(frames[0])
                if img is not None:
                    rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    h, w, _ = rgb.shape
                    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
                    item.setIcon(QIcon(QPixmap.fromImage(qimg)))
            self.queue_list.addItem(item)
        self.queue_group.setVisible(bool(self.queue))
        self._refresh_queue_marks()

    def _refresh_queue_marks(self):
        """Cập nhật ✓ (đã xong) / ▶ (đang làm) trên dải thumbnail."""
        for i in range(self.queue_list.count()):
            name = os.path.basename(self.queue[i])
            mark = "✓ " if i in self.done else ("▶ " if i == self.qi else "")
            self.queue_list.item(i).setText(f"{i + 1}. {mark}{name}")
        if 0 <= self.qi < self.queue_list.count():
            self.queue_list.setCurrentRow(self.qi)

    def _on_queue_clicked(self, item):
        self.qi = self.queue_list.row(item)
        self._load_queue_current()

    def _queue_menu(self, pos):
        """Menu chuột phải trên dải folder: bỏ folder này / bỏ tất cả."""
        item = self.queue_list.itemAt(pos)
        menu = QMenu(self)
        act_one = menu.addAction("❌ Bỏ folder này") if item else None
        act_all = menu.addAction("🗑 Bỏ tất cả folder")
        chosen = menu.exec_(self.queue_list.mapToGlobal(pos))
        if chosen is None:
            return
        if chosen is act_one:
            self._remove_queue_item(self.queue_list.row(item))
        elif chosen is act_all:
            self._clear_queue()

    def _remove_queue_item(self, row):
        """Bỏ một folder khỏi hàng đợi (cập nhật lại ✓/▶ và folder đang xem)."""
        if row < 0 or row >= len(self.queue):
            return
        cur_path = self.queue[self.qi] if 0 <= self.qi < len(self.queue) else None
        removed_is_current = (row == self.qi)
        del self.queue[row]
        # dời lại chỉ số "đã xong"
        self.done = {(i - 1 if i > row else i) for i in self.done if i != row}
        if not self.queue:
            self._clear_queue(); return
        # giữ folder đang xem nếu còn, không thì lấy folder kế
        if cur_path in self.queue:
            self.qi = self.queue.index(cur_path)
        else:
            self.qi = min(row, len(self.queue) - 1)
        self._build_queue_list()
        if removed_is_current:
            self._load_queue_current()

    def _clear_queue(self):
        """Bỏ toàn bộ folder đã chọn."""
        self.queue = []; self.qi = 0; self.done = set()
        self.queue_list.clear(); self.queue_group.hide()

    def eventFilter(self, obj, event):
        if (obj is self.queue_list and event.type() == QEvent.KeyPress
                and event.key() == Qt.Key_Delete):
            row = self.queue_list.currentRow()
            if row >= 0:
                self._remove_queue_item(row)
                return True
        return super().eventFilter(obj, event)

    def _load_queue_current(self):
        """Nạp folder hiện tại trong hàng đợi để hiệu chỉnh."""
        self.path_edit.setText(self.queue[self.qi])
        self._refresh_queue_marks()
        self.load_and_detect()

    def _clear_form(self):
        while self.form.count():
            item = self.form.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.id_inputs = []
        self.id_buttons = []

    def _select_box(self, k):
        """Chọn khung k (từ nút bên phải) và đồng bộ với ảnh."""
        self.view.select(k)
        self._highlight_row(k)

    def _highlight_row(self, k):
        for j, btn in enumerate(self.id_buttons):
            btn.setChecked(j == k)

    def _save_current_state(self):
        """Lưu id + khung đã chỉnh của folder ĐANG mở (trong phiên + ra đĩa)."""
        if self._current_folder and self.first_boxes:
            boxes = self.view.get_boxes()
            ids = [e.currentText() for e in self.id_inputs]
            self._cache[self._current_folder] = {
                "first_boxes": self.first_boxes,
                "first_kpts": self.first_kpts,
                "auto_crops": self.auto_crops,
                "boxes": boxes,
                "ids": ids,
                "colors": list(self._suggest_colors),   # giữ màu gợi ý khi quay lại folder
            }
            # bản nhẹ để lưu ra đĩa (id + khung cắt + bbox khung VẼ TAY để khôi phục lại).
            # Khung vẽ tay không tự phát hiện được -> phải lưu bbox của chúng (kpts is None).
            manual = [list(self.first_boxes[i]) for i in range(len(self.first_boxes))
                      if self.first_kpts[i] is None]
            if any(s.strip() for s in ids) or manual:
                rec = {"ids": ids, "boxes": boxes}
                if manual:
                    rec["manual"] = manual
                self._saved_edits[self._current_folder] = rec
                self._refresh_history()
            elif self._current_folder in self._saved_edits:
                # không còn id/khung tay -> bỏ bản lưu cũ (tránh khung tay đã xoá hiện lại)
                del self._saved_edits[self._current_folder]
                self._refresh_history()
            self._save_state_to_disk()

    def load_and_detect(self):
        folder = self.path_edit.text().strip().strip('"')
        if not os.path.isdir(folder):
            QMessageBox.warning(self, "Lỗi", "Đường dẫn thư mục không hợp lệ."); return
        self.frames = list_frames(folder)
        if not self.frames:
            QMessageBox.warning(self, "Lỗi", "Không tìm thấy ảnh trong thư mục."); return

        # Lưu trạng thái folder cũ trước khi chuyển
        if folder != self._current_folder:
            self._save_current_state()
        self._current_folder = folder

        first = cv2.imread(self.frames[0])
        if first is None:
            QMessageBox.warning(self, "Lỗi", "Không đọc được frame đầu."); return

        cached = self._cache.get(folder)
        suggest_colors = None
        if cached:                                   # KHÔI PHỤC trạng thái đã lưu
            self.first_boxes = cached["first_boxes"]
            self.first_kpts = cached["first_kpts"]
            self.auto_crops = cached["auto_crops"]
            saved_boxes = cached["boxes"]
            saved_ids = cached["ids"]
            suggest_colors = cached.get("colors")    # khôi phục màu gợi ý đã lưu
        else:                                        # PHÁT HIỆN mới
            self.statusBar().showMessage("Đang phát hiện người ở frame đầu...")
            QApplication.processEvents()
            try:
                boxes, kpts = detect(first)
            except Exception as e:
                QMessageBox.critical(self, "Lỗi YOLO", str(e)); return
            order = order_boxes(boxes, kpts)
            self.first_boxes = [boxes[i] for i in order]
            self.first_kpts = [kpts[i] for i in order]
            h0, w0 = first.shape[:2]
            self.auto_crops = [list(crop_region(self.first_boxes[i], self.first_kpts[i], w0, h0))
                               for i in range(len(self.first_boxes))]
            # Khôi phục các khung VẼ TAY đã lưu (model không tự phát hiện được) -> nối vào cuối,
            # kpts=None như lúc tạo, để chạy chung pipeline (delta + tracker giữ khung).
            disk = self._saved_edits.get(folder)
            for mb in (disk.get("manual", []) if disk else []):
                self.first_boxes.append([int(v) for v in mb])
                self.first_kpts.append(None)
                self.auto_crops.append(list(crop_region(mb, None, w0, h0)))
            saved_boxes = [list(b) for b in self.auto_crops]
            saved_ids = ["" for _ in self.first_boxes]
            # áp chỉnh sửa đã LƯU RA ĐĨA từ phiên trước (nếu có, khớp số khung gồm cả khung tay)
            if disk:
                db = disk.get("boxes", []); di = disk.get("ids", [])
                if len(db) == len(self.first_boxes):
                    saved_boxes = [list(x) for x in db]
                if len(di) == len(self.first_boxes):
                    saved_ids = list(di)

        # Gợi ý id tự động theo NGOẠI HÌNH (chỉ chạy LẦN ĐẦU: chưa có màu cache, chưa gán id)
        if suggest_colors is None and self._gallery and not any(s.strip() for s in saved_ids):
            saved_ids, suggest_colors = self._suggest_ids(first, saved_boxes)

        self._populate_ui(first, saved_boxes, saved_ids, suggest_colors)

        # Cập nhật cache cho folder hiện tại (giữ detection để lần sau không chạy lại YOLO)
        self._save_current_state()

        self.btn_export.setEnabled(True)
        self.btn_batch.setEnabled(bool(self.queue))
        if self.queue:
            self.btn_export.setText("🚀 Xử lý & folder tiếp →")
            prefix = f"[Folder {self.qi + 1}/{len(self.queue)}: {os.path.basename(self.queue[self.qi])}] "
        else:
            self.btn_export.setText("🚀 Xử lý & xuất tất cả")
            prefix = ""
        self.statusBar().showMessage(
            f"{prefix}{len(self.frames)} frame — phát hiện {len(self.first_boxes)} người ở frame đầu. "
            f"Gán id rồi nhấn nút xanh.")

    def _populate_ui(self, first_img, saved_boxes, saved_ids, suggest_colors=None):
        """Dựng khung + form gán id (dùng chung cho mở folder và tạo bộ nhớ id)."""
        # GIỮ vị trí khung đang xem (đầu/giữa/cuối) qua các folder — KHÔNG ép về "first" nữa.
        pos = self._frame_pos if self._frame_pos in ("first", "mid", "last") else "first"
        labels = [str(i + 1) for i in range(len(self.first_boxes))]
        self.view.set_frame(first_img)
        self.view.set_boxes(saved_boxes, labels)
        self.view.on_select = self._highlight_row

        # đồng bộ danh sách màu gợi ý hiện hành (dùng để LƯU cache + áp lại khi quay lại)
        n = len(self.first_boxes)
        self._suggest_colors = (list(suggest_colors) + [None] * n)[:n] if suggest_colors else [None] * n

        self._clear_form()
        for i in range(len(self.first_boxes)):
            self._add_id_row(i, saved_ids[i] if i < len(saved_ids) else "",
                             self._suggest_colors[i],
                             deletable=self.first_kpts[i] is None)   # None = khung vẽ tay
        if self.id_buttons:
            self._highlight_row(0)

        # tô nút + đổi ảnh nền sang đúng vị trí đã nhớ (giữa/cuối). "first" đã có sẵn first_img.
        for key, btn in self.pos_buttons.items():
            btn.setChecked(key == pos)
        if pos != "first" and len(self.frames) > 1:
            self._show_frame_pos(pos)

    def _add_id_row(self, i, value="", color=None, deletable=False):
        """Dựng 1 hàng gán id (nút chọn khung + ô id) cho khung thứ i, gắn vào form.
        deletable=True (khung VẼ TAY) -> thêm nút ✕ để xoá riêng khung đó."""
        row = QHBoxLayout()
        btn = QPushButton(f"Khung #{i + 1}"); btn.setCheckable(True); btn.setFixedWidth(95)
        btn.setStyleSheet("QPushButton{background:#3a3a3a;padding:5px;}"
                          "QPushButton:checked{background:#c8a000;color:#000;}")
        btn.clicked.connect(lambda _=False, k=i: self._select_box(k))
        self.id_buttons.append(btn); row.addWidget(btn)
        row.addWidget(QLabel("→ id_"))
        edit = self._make_id_combo()
        edit.setCurrentText(value)
        if color:
            edit.setStyleSheet(self._suggest_style(color))
        # khi người dùng tự sửa -> bỏ tô màu gợi ý (UI + cache) về mặc định.
        # Gắn KHÔNG điều kiện để màu khôi phục sau này (nút ↩) cũng tự mất khi sửa tay.
        edit.editTextChanged.connect(lambda _t, e=edit, k=i: self._on_id_edited(k, e))
        self.id_inputs.append(edit); row.addWidget(edit)
        if deletable:                                  # khung vẽ tay -> cho xoá nếu kéo nhầm
            dele = QPushButton("✕"); dele.setFixedWidth(28)
            dele.setToolTip("Xoá khung vẽ tay này (nhỡ kéo nhầm)")
            dele.setStyleSheet("QPushButton{background:#a33;padding:5px;}"
                               "QPushButton:hover{background:#c44;}")
            dele.clicked.connect(lambda _=False, k=i: self._delete_box(k))
            row.addWidget(dele)
        row.addStretch()
        w = QWidget(); w.setLayout(row); self.form.addWidget(w)

    def _suggest_ids(self, first_img, saved_boxes):
        """Gợi ý id theo ngoại hình, khớp 1-1 (mỗi id chỉ gán 1 người). Trả về (ids, màu)."""
        n = len(saved_boxes)
        embs = [gallery.embed(first_img[y1:y2, x1:x2]) for (x1, y1, x2, y2) in saved_boxes]
        pairs = []
        for i, e in enumerate(embs):
            if e is None:
                continue
            for idv, vecs in self._gallery.items():
                s = max(float(np.dot(e, v)) for v in vecs)
                pairs.append((s, i, idv))
        pairs.sort(reverse=True)
        ids = [""] * n; colors = [None] * n
        used_i, used_id = set(), set()
        for s, i, idv in pairs:
            if i in used_i or idv in used_id:
                continue
            if s < 0.45:                       # quá thấp -> để trống, gõ tay
                break
            used_i.add(i); used_id.add(idv)
            ids[i] = idv
            colors[i] = "#28a745" if s >= 0.62 else "#d0a000"   # xanh chắc / vàng ngờ
        return ids, colors

    def _on_id_edited(self, k, edit):
        """Người dùng sửa tay 1 ô id -> bỏ màu gợi ý của ô đó (UI + cache)."""
        if k < len(self._suggest_colors) and self._suggest_colors[k]:
            edit.setStyleSheet("")
            self._suggest_colors[k] = None

    def _restore_suggestions(self):
        """Khôi phục id theo GỢI Ý: chạy lại so khớp ngoại hình trên frame đầu, điền id + tô màu.
        Ghi đè mọi ô (cả ô đã xoá / đã sửa tay) bằng kết quả gợi ý mới nhất."""
        if not self._gallery:
            QMessageBox.information(self, "Chưa có bộ nhớ id",
                "Chưa có bộ nhớ id để gợi ý. Hãy tạo bằng '🧠 Bộ nhớ id' trước."); return
        if not self.frames or not self.id_inputs:
            return
        first = cv2.imread(self.frames[0])
        if first is None:
            QMessageBox.warning(self, "Lỗi", "Không đọc được frame đầu."); return
        boxes = self.view.get_boxes()
        ids, colors = self._suggest_ids(first, boxes)
        restored = 0
        for i, edit in enumerate(self.id_inputs):
            col = colors[i] if i < len(colors) else None
            val = ids[i] if i < len(ids) else ""
            edit.blockSignals(True)                  # tự điền -> không kích _on_id_edited
            edit.setCurrentText(val)
            edit.setStyleSheet(self._suggest_style(col) if col else "")
            edit.blockSignals(False)
            if i < len(self._suggest_colors):
                self._suggest_colors[i] = col
            if val:
                restored += 1
        self._save_current_state()
        self.statusBar().showMessage(
            f"Đã khôi phục {restored} id gợi ý (xanh = chắc, vàng = nên kiểm)." if restored
            else "Không có id nào được gợi ý (ngoại hình không khớp bộ nhớ).")

    def _clear_suggestions(self):
        """Xoá HẾT id của mọi khung (cả id gợi ý lẫn id đã gõ tay) để gán lại từ đầu."""
        cleared = 0
        for i, edit in enumerate(self.id_inputs):
            if edit.currentText().strip():
                cleared += 1
            edit.blockSignals(True)                  # tránh kích editTextChanged khi tự xoá
            edit.setCurrentText("")
            edit.setStyleSheet("")
            edit.blockSignals(False)
            if i < len(self._suggest_colors):
                self._suggest_colors[i] = None
        self._save_current_state()                   # cập nhật cache + đĩa
        self.statusBar().showMessage(
            f"Đã xoá {cleared} id — gán lại từ đầu cho dễ." if cleared
            else "Không có id nào để xoá.")

    def reset_boxes(self):
        """Khôi phục các khung cắt về tự động (bỏ chỉnh tay)."""
        if self.auto_crops:
            labels = [str(i + 1) for i in range(len(self.auto_crops))]
            self.view.set_boxes([list(b) for b in self.auto_crops], labels)

    def _start_add_box(self):
        """Bật chế độ vẽ tay 1 khung cho đối tượng model bỏ sót."""
        if not self.view.has_image():
            QMessageBox.information(self, "Chưa có ảnh",
                "Hãy 'Tải & phát hiện' một folder trước khi thêm khung."); return
        self.btn_nav.setChecked(False)               # tắt chế độ di chuyển nếu đang bật
        self.view.on_new_box = self._on_new_box
        self.view.set_draw_mode(True)
        self.statusBar().showMessage("✏️ Vẽ khung: KÉO chuột trên ảnh để khoanh đối tượng "
                                     "còn thiếu (bấm 1 phát không kéo = huỷ).")

    def _on_new_box(self, box):
        """Thêm 1 khung VẼ TAY (không có keypoint) vào danh sách, kèm 1 ô gán id mới.
        Khung này chạy chung pipeline: IoUTracker bám qua mọi frame, delta giữ đúng vùng cắt."""
        if not self.view.has_image():
            return
        w, h = self.view._img.width(), self.view._img.height()
        # bbox người = vùng vẽ; auto_crop = crop_region(None kpts) để delta khớp khi xuất
        self.first_boxes.append([int(v) for v in box])
        self.first_kpts.append(None)
        self.auto_crops.append(list(crop_region(box, None, w, h)))
        i = len(self.first_boxes) - 1
        self._suggest_colors.append(None)
        self.view.add_box(box, i + 1)                # hiển thị + chọn khung mới
        self._add_id_row(i, "", None, deletable=True)   # ô gán id + nút ✕ (xoá nếu kéo nhầm)
        self._select_box(i)
        # cuộn xuống để thấy ô id mới, bật nút xuất, lưu trạng thái
        QTimer.singleShot(0, lambda: self.scroll.verticalScrollBar().setValue(
            self.scroll.verticalScrollBar().maximum()))
        self.btn_export.setEnabled(True)
        self._save_current_state()
        self.statusBar().showMessage(f"Đã thêm khung #{i + 1}. Gõ id cho khung này rồi xuất.")

    def _delete_box(self, i):
        """Xoá khung thứ i (chỉ dùng cho khung VẼ TAY) khỏi mọi danh sách + dựng lại form."""
        if not (0 <= i < len(self.first_boxes)):
            return
        # gom id + khung hiện tại từ UI trước khi xoá để giữ nguyên các khung còn lại
        ids = [e.currentText() for e in self.id_inputs]
        boxes = self.view.get_boxes()
        colors = list(self._suggest_colors)
        for lst in (self.first_boxes, self.first_kpts, self.auto_crops, ids, boxes, colors):
            if i < len(lst):
                del lst[i]
        self._rebuild_id_rows(boxes, ids, colors)
        if not self.first_boxes:
            self.btn_export.setEnabled(False)
        self._save_current_state()
        self.statusBar().showMessage(f"Đã xoá khung #{i + 1}.")

    def _rebuild_id_rows(self, boxes, ids, colors):
        """Cập nhật khung hiển thị + dựng lại toàn bộ hàng id (đánh số lại) sau khi thêm/xoá.
        KHÔNG đụng ảnh nền / zoom hiện tại."""
        n = len(self.first_boxes)
        self.view.boxes = [[int(v) for v in b] for b in boxes]
        self.view.labels = [str(k + 1) for k in range(n)]
        self.view.selected = 0 if self.view.boxes else None
        self.view.update()
        self._suggest_colors = (list(colors) + [None] * n)[:n]
        self._clear_form()
        for k in range(n):
            self._add_id_row(k, ids[k] if k < len(ids) else "", self._suggest_colors[k],
                             deletable=self.first_kpts[k] is None)
        if self.id_buttons:
            self._highlight_row(0)

    # ---- Bộ nhớ id (gallery theo ngoại hình) ----
    def build_gallery(self):
        """Mở ảnh quy tắc, gán 01–21 một lần để tạo bộ nhớ id."""
        here = os.path.dirname(os.path.abspath(__file__))
        rp = next((os.path.join(here, n) for n in ("rulev2.jpg", "rule.png", "rulepic.png")
                   if os.path.exists(os.path.join(here, n))), None)
        start = rp if rp else _start_dir("")
        path, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh quy tắc (đánh số người)",
                                              start, "Ảnh (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            QMessageBox.warning(self, "Lỗi", "Không đọc được ảnh."); return
        self.statusBar().showMessage("Đang phát hiện người trong ảnh quy tắc..."); QApplication.processEvents()
        try:
            boxes, kpts = detect(img)
        except Exception as e:
            QMessageBox.critical(self, "Lỗi YOLO", str(e)); return
        order = order_boxes(boxes, kpts)
        self._save_current_state()                 # lưu folder thật trước khi chuyển sang chế độ gallery
        self._current_folder = None                # tránh ghi đè state folder
        self.frames = [path]
        self.first_boxes = [boxes[i] for i in order]
        self.first_kpts = [kpts[i] for i in order]
        h0, w0 = img.shape[:2]
        self.auto_crops = [list(crop_region(self.first_boxes[i], self.first_kpts[i], w0, h0))
                           for i in range(len(self.first_boxes))]
        self._gallery_img = img
        self._frame_pos = "first"        # ảnh quy tắc chỉ 1 frame -> luôn xem "đầu"
        self._populate_ui(img, [list(b) for b in self.auto_crops], ["" for _ in self.first_boxes])
        self.btn_savegal.setEnabled(True)
        self.tabs.setCurrentIndex(0)
        QMessageBox.information(self, "Tạo bộ nhớ id",
            "Gán 01–21 cho từng người (theo ảnh quy tắc), chỉnh khung nếu cần, "
            "rồi nhấn '💾 Lưu bộ nhớ id'.")

    def save_gallery_now(self):
        """Tính đặc trưng từng người đã gán id và lưu thành bộ nhớ id."""
        if getattr(self, "_gallery_img", None) is None:
            return
        img = self._gallery_img
        gal = {}
        for i, edit in enumerate(self.id_inputs):
            idv = edit.currentText().strip()
            if not idv:
                continue
            x1, y1, x2, y2 = self.view.boxes[i]
            e = gallery.embed(img[y1:y2, x1:x2])
            if e is not None:
                gal.setdefault(idv, []).append(e)
        if not gal:
            QMessageBox.warning(self, "Chưa gán id", "Hãy gán id cho ít nhất một người."); return
        gallery.save_gallery(gal)
        self._gallery = gallery.load_gallery()
        self._gallery_img = None
        self.btn_savegal.setEnabled(False)
        QMessageBox.information(self, "Đã lưu bộ nhớ id",
            f"Đã ghi nhớ {len(gal)} id. Từ giờ mở folder sẽ TỰ GỢI Ý id "
            f"(viền xanh = chắc, vàng = nên kiểm).")

    def _toggle_nav(self, on):
        """Bật/tắt chế độ di chuyển ảnh (zoom + kéo)."""
        self.view.set_nav(on)
        self.btn_nav.setText("✋ Đang di chuyển" if on else "✋ Di chuyển ảnh")

    def _zoom_btn(self, factor):
        """Nút zoom: phóng quanh tâm khung xem."""
        self.view.zoom_at(factor, self.view.width() / 2, self.view.height() / 2)

    def _show_frame_pos(self, key):
        """Đổi ảnh nền sang frame ĐẦU/GIỮA/CUỐI để so sánh (giữ nguyên khung)."""
        if not self.frames:
            return
        self._frame_pos = key
        for k, btn in self.pos_buttons.items():
            btn.setChecked(k == key)
        idx = {"first": 0, "mid": len(self.frames) // 2, "last": len(self.frames) - 1}[key]
        img = cv2.imread(self.frames[idx])
        if img is not None:
            self.view.set_frame(img)      # set_frame KHÔNG đụng tới khung -> giữ nguyên để so sánh
        self.statusBar().showMessage(
            f"Đang xem frame {os.path.basename(self.frames[idx])} ({ {'first':'đầu','mid':'giữa','last':'cuối'}[key] })")

    # ---- trang Lịch sử ----
    def _build_history_tab(self):
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(16, 16, 16, 16)
        lay.addWidget(QLabel("🕘 Các folder đã/đang sửa — bấm để mở lại:"))
        self.history_list = QListWidget()
        self.history_list.setStyleSheet(
            "QListWidget{background:#262626;border:1px solid #555;border-radius:6px;font-size:13px;}"
            "QListWidget::item{padding:8px;border-bottom:1px solid #333;}"
            "QListWidget::item:selected{background:#3a6ea5;color:#fff;}")
        self.history_list.itemClicked.connect(self._on_history_clicked)
        lay.addWidget(self.history_list, 1)
        self.tabs.addTab(w, "🕘 Lịch sử")

    def _refresh_history(self):
        if not hasattr(self, "history_list"):
            return
        self.history_list.clear()
        for folder, e in self._saved_edits.items():
            nid = sum(1 for s in e.get("ids", []) if s.strip())
            item = QListWidgetItem(f"📁 {os.path.basename(folder)}   —   {nid} id đã gán")
            item.setData(Qt.UserRole, folder)
            item.setToolTip(folder)
            self.history_list.addItem(item)

    def _on_history_clicked(self, item):
        folder = item.data(Qt.UserRole)
        if not folder or not os.path.isdir(folder):
            return
        self._save_current_state()
        if folder in self.queue:
            self.qi = self.queue.index(folder)
            self._load_queue_current()
        else:
            self.path_edit.setText(folder)
            self.load_and_detect()
        self.tabs.setCurrentIndex(0)      # quay về tab Làm việc

    # ---- trang Ảnh đã cắt (duyệt FOLDER ĐÃ CHỌN → id_ → ảnh, xem ngay trong app) ----
    def _build_results_tab(self):
        """Tab xem KẾT QUẢ. Bên trái là CÂY: mỗi folder đã chọn là 1 node, bên trong là các id_;
        bấm 1 id_ thì bên phải hiện ảnh. Đỡ phải mở Windows Explorer để kiểm sau khi cắt."""
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(12, 12, 12, 12); lay.setSpacing(8)
        top = QHBoxLayout()
        top.addWidget(QLabel("📂 Các folder đã chọn — bấm 1 folder để xem ảnh:"))
        top.addStretch()
        b_pick = QPushButton("📂 Xem folder khác..."); b_pick.clicked.connect(self._res_pick)
        top.addWidget(b_pick)
        b_ref = QPushButton("🔄 Làm mới"); b_ref.clicked.connect(self._refresh_results)
        top.addWidget(b_ref)
        lay.addLayout(top)

        hint = QLabel("Bấm 1 folder bên trái → ô chính hiện 3 ảnh (đầu/giữa/cuối) của từng id_ theo thứ tự. "
                      "BẤM ĐÚP ảnh để phóng to / lật ← → cả id_ đó / xoá ảnh hỏng (chuột phải = menu nhanh).")
        hint.setStyleSheet("color:#888;font-weight:normal;"); hint.setWordWrap(True)
        lay.addWidget(hint)

        split = QSplitter(Qt.Horizontal); split.setHandleWidth(8); split.setChildrenCollapsible(False)
        split.setStyleSheet("QSplitter::handle{background:#444;border-radius:3px;margin:2px;}"
                            "QSplitter::handle:hover{background:#3a6ea5;}")
        # trái: CÂY folder đã chọn -> id_
        self.res_tree = QTreeWidget(); self.res_tree.setHeaderHidden(True)
        self.res_tree.setStyleSheet(
            "QTreeWidget{background:#262626;border:1px solid #555;border-radius:6px;font-size:13px;}"
            "QTreeWidget::item{padding:5px;}"
            "QTreeWidget::item:selected{background:#3a6ea5;color:#fff;}")
        self.res_tree.currentItemChanged.connect(lambda *_: self._on_res_tree_select())
        split.addWidget(self.res_tree)
        # phải: lưới thumbnail ảnh trong id_ đang chọn
        self.res_thumbs = QListWidget()
        self.res_thumbs.setViewMode(QListView.IconMode)
        self.res_thumbs.setIconSize(QSize(120, 150))
        self.res_thumbs.setResizeMode(QListView.Adjust)
        self.res_thumbs.setMovement(QListView.Static)
        self.res_thumbs.setSpacing(8)
        self.res_thumbs.setStyleSheet(
            "QListWidget{background:#262626;border:1px solid #555;border-radius:6px;}"
            "QListWidget::item{color:#bbb;}"
            "QListWidget::item:selected{background:#3a6ea5;color:#fff;border-radius:4px;}")
        self.res_thumbs.itemDoubleClicked.connect(lambda _it: self._open_image_viewer())
        self.res_thumbs.setContextMenuPolicy(Qt.CustomContextMenu)
        self.res_thumbs.customContextMenuRequested.connect(self._res_thumb_menu)
        split.addWidget(self.res_thumbs)
        split.setStretchFactor(0, 0); split.setStretchFactor(1, 1)
        split.setSizes([300, 700])
        lay.addWidget(split, 1)

        self.res_status = QLabel(""); self.res_status.setStyleSheet("color:#888;font-weight:normal;")
        lay.addWidget(self.res_status)

        self._results_tab_index = self.tabs.addTab(w, "🖼 Ảnh đã cắt")

    @staticmethod
    def _natkey(name):
        """Khoá sắp xếp TỰ NHIÊN: tách số/chữ để '..._9' đứng TRƯỚC '..._37' (không bị so chuỗi)."""
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', name)]

    @classmethod
    def _list_images(cls, folder):
        """Danh sách đường dẫn ảnh trong 1 folder, sắp theo SỐ tự nhiên (đúng thứ tự frame)."""
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        try:
            names = [n for n in os.listdir(folder) if n.lower().endswith(exts)]
        except OSError:
            return []
        names.sort(key=cls._natkey)
        return [os.path.join(folder, n) for n in names]

    def _result_source_folders(self):
        """Các folder ĐÃ CHỌN để xem kết quả: hàng đợi nếu có, không thì folder đang làm,
        cộng thêm các folder người dùng tự thêm bằng 'Xem folder khác...'. Giữ thứ tự, bỏ trùng."""
        folders = list(self.queue) if self.queue else []
        if not folders:
            cur = self._current_folder or self.path_edit.text().strip().strip('"')
            if cur:
                folders.append(cur)
        folders += list(self._res_extra)
        out, seen = [], set()
        for f in folders:
            if f and os.path.isdir(f) and f not in seen:
                seen.add(f); out.append(f)
        return out

    def _on_tab_changed(self, idx):
        """Sang tab Ảnh đã cắt -> làm mới cây folder đã chọn."""
        if idx == getattr(self, "_results_tab_index", -1):
            self._refresh_results()

    def _res_pick(self):
        """Thêm 1 folder bất kỳ vào cây để xem (ngoài các folder đã chọn)."""
        d = QFileDialog.getExistingDirectory(self, "Chọn folder chứa các id_",
                                             _start_dir(self.path_edit.text()))
        if d:
            if d not in self._res_extra:
                self._res_extra.append(d)
            self._refresh_results()

    def _refresh_results(self):
        """Dựng lại danh sách FOLDER LỚN đã chọn (phẳng, không xổ id_). Bấm 1 folder -> ô chính
        gộp 3 ảnh đầu/giữa/cuối của MỌI id_ bên trong, sắp theo thứ tự id."""
        if not hasattr(self, "res_tree"):
            return
        prev = self._res_selected
        self.res_tree.blockSignals(True)
        self.res_tree.clear(); self.res_thumbs.clear()
        folders = self._result_source_folders()
        reselect = None
        for folder in folders:
            nid = len(self._id_subfolders(folder))
            top = QTreeWidgetItem([f"📁 {os.path.basename(folder)}  ({nid} id)"])
            top.setData(0, Qt.UserRole, folder)
            top.setToolTip(0, folder)
            self.res_tree.addTopLevelItem(top)
            if folder == prev:
                reselect = top
        if reselect is not None:                      # GIỮ folder đang xem (chuyển tab/mở lại vẫn thấy)
            self.res_tree.setCurrentItem(reselect)    # tô sáng lại trong cây
            self.res_tree.blockSignals(False)
            self._res_selected = prev
            self._load_folder_thumbs(prev)            # nạp thẳng ảnh -> chắc chắn hiện lại đúng phần đó
            return
        self.res_tree.blockSignals(False)
        if not folders:
            self.res_status.setText("Chưa có folder nào được chọn — mở folder ở tab Làm việc trước.")
        else:
            self.res_status.setText(f"{len(folders)} folder đã chọn — bấm 1 folder để xem ảnh các id_.")

    @staticmethod
    def _id_sort_key(name):
        """Sắp id_ theo SỐ nếu là số (id_2 trước id_10), không thì theo tên."""
        tail = name[3:]
        return (0, int(tail)) if tail.isdigit() else (1, name)

    def _id_subfolders(self, folder):
        """Tên các folder id_* trong 1 folder lớn, sắp theo số id."""
        try:
            subs = [d for d in os.listdir(folder)
                    if d.startswith("id_") and os.path.isdir(os.path.join(folder, d))]
        except OSError:
            return []
        return sorted(subs, key=self._id_sort_key)

    def _on_res_tree_select(self):
        """Chọn 1 folder lớn -> đổ 3 ảnh đại diện của mọi id_ bên trong ra ô chính."""
        self.res_thumbs.clear()
        item = self.res_tree.currentItem()
        folder = item.data(0, Qt.UserRole) if item else None
        self._res_selected = folder
        if folder:
            self._load_folder_thumbs(folder)

    @staticmethod
    def _pick3(paths):
        """Lấy 3 ảnh ĐẠI DIỆN: đầu / giữa / cuối (ít hơn 3 thì lấy đủ số có)."""
        n = len(paths)
        if n == 0:
            return []
        if n == 1:
            return [(paths[0], "Đầu")]
        if n == 2:
            return [(paths[0], "Đầu"), (paths[-1], "Cuối")]
        return [(paths[0], "Đầu"), (paths[n // 2], "Giữa"), (paths[-1], "Cuối")]

    def _load_folder_thumbs(self, folder):
        """Gộp ảnh của MỌI id_ trong folder lớn: mỗi id_ chỉ 3 ảnh (đầu/giữa/cuối), theo thứ tự id."""
        self.res_thumbs.clear()
        subs = self._id_subfolders(folder)
        total = 0
        for d in subs:
            paths = self._list_images(os.path.join(folder, d))
            total += len(paths)
            for p, tag in self._pick3(paths):
                it = QListWidgetItem(f"{d} · {tag}")
                pm = QPixmap(p)
                if not pm.isNull():
                    it.setIcon(QIcon(pm))
                it.setData(Qt.UserRole, p)
                self.res_thumbs.addItem(it)
        if subs:
            self.res_status.setText(
                f"{os.path.basename(folder)} — {len(subs)} id, tổng {total} ảnh "
                f"(mỗi id hiện 3 ảnh đầu/giữa/cuối; bấm đúp 1 ảnh để xem to / lật hết / xoá).")
        else:
            self.res_status.setText(f"{os.path.basename(folder)} — chưa có folder id_ nào (chưa xuất?).")

    def _open_image_viewer(self):
        """Mở hộp thoại xem ảnh phóng to. Lật được HẾT ảnh trong CHÍNH id_ của ảnh được bấm
        (suy ra folder id_ từ đường dẫn ảnh), mở đúng ở ảnh đã bấm."""
        item = self.res_thumbs.currentItem()
        if not item:
            return
        p = item.data(Qt.UserRole)
        idfolder = os.path.dirname(p)
        paths = self._list_images(idfolder)
        if not paths:
            return
        idx = paths.index(p) if p in paths else 0
        dlg = ImageViewerDialog(paths, idx, self)
        dlg.exec_()
        if dlg.deleted and self._res_selected:        # đã xoá ảnh -> dựng lại ô chính
            self._load_folder_thumbs(self._res_selected)

    def _res_thumb_menu(self, pos):
        """Chuột phải trên 1 thumbnail: xem to / xoá ảnh."""
        item = self.res_thumbs.itemAt(pos)
        if not item:
            return
        menu = QMenu(self)
        act_view = menu.addAction("🔍 Xem ảnh")
        act_del = menu.addAction("🗑 Xoá ảnh này")
        chosen = menu.exec_(self.res_thumbs.mapToGlobal(pos))
        if chosen is act_view:
            self.res_thumbs.setCurrentItem(item); self._open_image_viewer()
        elif chosen is act_del:
            p = item.data(Qt.UserRole)
            if QMessageBox.question(self, "Xoá ảnh", f"Xoá ảnh này?\n{os.path.basename(p)}",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
                return
            try:
                os.remove(p)
            except OSError as e:
                QMessageBox.warning(self, "Lỗi", f"Không xoá được: {e}"); return
            if self._res_selected:
                self._load_folder_thumbs(self._res_selected)

    @staticmethod
    def _suggest_style(color):
        """Stylesheet TÔ NỀN cho ô gợi ý id: nền đậm + chữ đen -> nổi rõ trên nền tối.
        Tô cả ô (không chỉ viền) nên không bị khung con của QComboBox vẽ đè."""
        return (f"QComboBox{{background:{color}; color:#000; font-weight:bold;"
                f" border:1px solid #000; border-radius:5px; padding:3px 4px;}}"
                f"QComboBox:editable{{background:{color};}}"
                f"QComboBox QLineEdit{{background:{color}; color:#000;}}"
                f"QComboBox QAbstractItemView{{background:#2b2b2b; color:#eee;}}")

    def _make_id_combo(self):
        """Ô chọn id 1..21 (vẫn gõ tay được)."""
        cb = QComboBox(); cb.setEditable(True); cb.setFixedWidth(80)
        cb.addItem("")                                     # trống = bỏ qua
        cb.addItems([str(n) for n in range(1, 22)])        # 1..21
        cb.lineEdit().setPlaceholderText("bỏ")
        return cb

    def _collect_export_plan(self):
        """Tách dữ liệu xuất thành 2 nhóm:
          - khung PHÁT HIỆN (có keypoint): chạy IoUTracker + crop_region + delta như cũ.
          - khung VẼ TAY (kpts=None): cắt CỐ ĐỊNH 1 vùng (đúng khung vẽ) trên mọi frame,
            KHÔNG đưa vào tracker (tránh khớp nhầm người khác làm ảnh nhảy/không đều).
        Trả về (det_boxes, det_kpts, mapping{detidx+1:id}, deltas{detidx+1:delta}, manual_crops{id:rect}).
        """
        edited = self.view.get_boxes()
        det_boxes, det_kpts, mapping, deltas, manual_crops = [], [], {}, {}, {}
        for i in range(len(self.first_boxes)):
            val = self.id_inputs[i].currentText().strip() if i < len(self.id_inputs) else ""
            eb = edited[i] if i < len(edited) else self.auto_crops[i]
            if self.first_kpts[i] is None:                 # khung vẽ tay -> cắt cố định
                if val:
                    manual_crops[val] = (int(eb[0]), int(eb[1]), int(eb[2]), int(eb[3]))
            else:                                          # khung phát hiện -> tracking
                di = len(det_boxes)
                det_boxes.append(self.first_boxes[i])
                det_kpts.append(self.first_kpts[i])
                if val:
                    mapping[di + 1] = val
                    ab = self.auto_crops[i]
                    deltas[di + 1] = (eb[0] - ab[0], eb[1] - ab[1], eb[2] - ab[2], eb[3] - ab[3])
        return det_boxes, det_kpts, mapping, deltas, manual_crops

    @staticmethod
    def _boxes_from_record(rec):
        """Danh sách KHUNG CẮT đã chỉnh tay của 1 folder (mọi slot, theo thứ tự gốc), từ bản lưu
        (cache phiên hoặc app_state). KHÔNG lấy id riêng của folder — id sẽ gán lại theo MẪU."""
        return [[int(v) for v in b] for b in (rec.get("boxes") or [])]

    @staticmethod
    def _fit_box(x1, y1, x2, y2, w, h):
        """Dời khung vào trong ảnh nhưng GIỮ NGUYÊN kích thước nếu được (chỉ co lại khi to hơn ảnh).
        Nhờ vậy mọi ảnh trong 1 id giữ đúng W×H -> đều nhau, không bị cắt cụt méo ở mép."""
        bw, bh = x2 - x1, y2 - y1
        if bw >= w:
            x1, x2 = 0, w
        elif x1 < 0:
            x1, x2 = 0, bw
        elif x2 > w:
            x1, x2 = w - bw, w
        if bh >= h:
            y1, y2 = 0, h
        elif y1 < 0:
            y1, y2 = 0, bh
        elif y2 > h:
            y1, y2 = h - bh, h
        return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

    def _export_folder(self, folder, frames, first_boxes, first_kpts, mapping, deltas,
                       manual_crops=None, on_frame=None):
        """Cắt & lưu một folder. mapping/deltas key theo seed id (chỉ số + 1).
        manual_crops {id: (x1,y1,x2,y2)} = khung VẼ TAY, cắt CỐ ĐỊNH cùng vùng mọi frame.
        Trả về counts."""
        manual_crops = manual_crops or {}
        all_ids = set(mapping.values()) | set(manual_crops.keys())
        counts = {idv: 0 for idv in all_ids}
        for idv in all_ids:
            os.makedirs(os.path.join(folder, f"id_{idv}"), exist_ok=True)
        tracker = IoUTracker(first_boxes)
        last = {}        # idv -> (x1,y1,x2,y2) khung lần gần nhất, để GIỮ KHUNG khi frame mất dấu
        fixed_size = {}  # tid -> (W,H) chốt ở frame đầu: mọi ảnh trong 1 id ĐỀU kích thước
        for fi, fpath in enumerate(frames):
            img = cv2.imread(fpath)
            if img is not None:
                h, w = img.shape[:2]
                fname = os.path.basename(fpath)
                # KHUNG VẼ TAY: cắt cố định 1 vùng cho mọi frame (model không thấy -> giữ ĐỀU).
                for idv, rect in manual_crops.items():
                    x1, y1, x2, y2 = rect
                    x1 = max(0, min(int(x1), w)); x2 = max(0, min(int(x2), w))
                    y1 = max(0, min(int(y1), h)); y2 = max(0, min(int(y2), h))
                    if x2 - x1 >= 5 and y2 - y1 >= 5:
                        crop = img[y1:y2, x1:x2]
                        if crop.size:
                            cv2.imwrite(os.path.join(folder, f"id_{idv}", fname), crop)
                            counts[idv] += 1
                if fi == 0:
                    dets, dkpts = first_boxes, first_kpts
                    matches = {i: i + 1 for i in range(len(dets))}
                elif mapping:
                    dets, dkpts = detect(img)
                    matches = tracker.update(dets)
                else:
                    dets, dkpts, matches = [], [], {}   # chỉ cắt khung cố định -> khỏi chạy YOLO
                done = set()
                for di, tid in matches.items():
                    idv = mapping.get(tid)
                    if idv is None:
                        continue
                    # khung "tự nhiên" theo pose frame này (+ chỉnh tay của người dùng)
                    x1, y1, x2, y2 = crop_region(dets[di], dkpts[di], w, h)
                    dl = deltas.get(tid)
                    if dl:
                        x1 += dl[0]; y1 += dl[1]; x2 += dl[2]; y2 += dl[3]
                    if fi == 0:
                        # frame đầu = khung tham chiếu -> CHỐT kích thước cố định cho id này
                        fixed_size[tid] = (x2 - x1, y2 - y1)
                    else:
                        # giữ nguyên W×H frame đầu, đặt TÂM theo khung tự nhiên (bám người).
                        # chốt x2=x1+wf để mọi ảnh trong id ĐỀU đúng kích thước (không lệch 1px).
                        wf, hf = fixed_size.get(tid, (x2 - x1, y2 - y1))
                        cx = (x1 + x2) / 2.0; cy = (y1 + y2) / 2.0
                        x1 = int(round(cx - wf / 2.0)); x2 = x1 + wf
                        y1 = int(round(cy - hf / 2.0)); y2 = y1 + hf
                    # dời khung vào trong ảnh nhưng GIỮ NGUYÊN kích thước (không cắt cụt ở mép)
                    x1, y1, x2, y2 = self._fit_box(x1, y1, x2, y2, w, h)
                    if x2 - x1 < 5 or y2 - y1 < 5:
                        continue
                    crop = img[y1:y2, x1:x2]
                    if crop.size:
                        cv2.imwrite(os.path.join(folder, f"id_{idv}", fname), crop)
                        counts[idv] += 1
                        last[idv] = (x1, y1, x2, y2)
                        done.add(idv)
                # ĐỦ 60 ẢNH: id nào mất dấu ở frame này -> cắt theo khung lần gần nhất
                for idv in set(mapping.values()):
                    if idv not in done and idv in last:
                        x1, y1, x2, y2 = last[idv]
                        crop = img[y1:y2, x1:x2]
                        if crop.size:
                            cv2.imwrite(os.path.join(folder, f"id_{idv}", fname), crop)
                            counts[idv] += 1
            if on_frame:
                on_frame(fi, len(frames))
        return counts

    def export_all(self):
        if not self.frames or not self.first_boxes:
            return
        det_boxes, det_kpts, mapping, deltas, manual_crops = self._collect_export_plan()
        if not mapping and not manual_crops:
            QMessageBox.warning(self, "Chưa gán id", "Hãy gõ id cho ít nhất một khung."); return

        folder = self.path_edit.text().strip().strip('"')
        self.progress.setMaximum(len(self.frames)); self.progress.setValue(0)
        self.btn_export.setEnabled(False)

        def on_frame(fi, total):
            self.progress.setValue(fi + 1)
            if fi % 3 == 0:
                self.statusBar().showMessage(f"Đang xử lý frame {fi + 1}/{total}...")
                QApplication.processEvents()

        counts = self._export_folder(folder, self.frames, det_boxes, det_kpts,
                                     mapping, deltas, manual_crops, on_frame)
        self.btn_export.setEnabled(True)
        summary = "\n".join(f"  id_{k}: {v} ảnh" for k, v in sorted(counts.items()))

        # cập nhật cây tab "Ảnh đã cắt" để sang tab là kiểm được folder vừa xuất ngay
        self._refresh_results()

        if self.queue:
            self.done.add(self.qi)
            self._refresh_queue_marks()

        # KHÔNG tự nhảy sang folder kế: ở lại folder hiện tại để chỉnh tiếp.
        # Người dùng tự bấm folder khác trong danh sách bên trái khi muốn chuyển.
        remaining = [i for i in range(len(self.queue)) if i not in self.done]
        extra = (f"\n\nCòn {len(remaining)} folder chưa làm — bấm folder trong danh sách bên trái "
                 f"khi muốn chuyển (KHÔNG tự chuyển nữa)." if remaining else
                 (f"\n\nĐã xong tất cả {len(self.queue)} folder." if self.queue else ""))
        QMessageBox.information(self, "Xong folder này",
            f"Đã xuất vào:\n{folder}\n\nSố ảnh mỗi id:\n{summary}{extra}")
        self.statusBar().showMessage(
            f"Đã xuất {len(counts)} id, tổng {sum(counts.values())} ảnh — ở lại folder này để chỉnh tiếp.")

    @staticmethod
    def _match_boxes(new_boxes, ref_boxes, max_shift=1.2):
        """Khớp 1-1 khung folder mới với khung mẫu theo KHOẢNG CÁCH TÂM (bền hơn IoU khi người dịch
        nhẹ). Chỉ ghép khi tâm cách nhau <= max_shift x bề rộng người mẫu. Trả {new_idx: ref_idx}."""
        def cen(b):
            return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
        pairs = []
        for ni, nb in enumerate(new_boxes):
            cn = cen(nb)
            for ri, rb in enumerate(ref_boxes):
                cr = cen(rb)
                d = math.hypot(cn[0] - cr[0], cn[1] - cr[1])
                if d <= (rb[2] - rb[0]) * max_shift:
                    pairs.append((d, ni, ri))
        pairs.sort()                       # gần nhất trước
        used_n, used_r, out = set(), set(), {}
        for d, ni, ri in pairs:
            if ni in used_n or ri in used_r:
                continue
            used_n.add(ni); used_r.add(ri); out[ni] = ri
        return out

    def batch_all(self):
        """Xuất TẤT CẢ folder: mỗi folder dùng KHUNG CẮT của chính nó, ID luôn gán theo MẪU
        (KHÔNG dùng id riêng của folder).
          - Folder ĐÃ CHỈNH → giữ khung đã chỉnh của nó; id remap theo mẫu (khớp tâm crop box).
          - Folder CHƯA chỉnh → phát hiện CỦA CHÍNH NÓ + crop_region/delta riêng; id theo mẫu (khớp tâm
            khung người). Khung VẼ TAY của mẫu (camera cố định) áp cố định cho folder chưa chỉnh."""
        if not self.queue:
            QMessageBox.warning(self, "Chưa có nhiều folder",
                "Hãy dùng '📁 Chọn nhiều folder...' trước, rồi gán id cho 1 folder làm mẫu."); return
        if not self.first_boxes:
            return
        # Mẫu: khung phát hiện (để khớp tâm + delta) và khung vẽ tay (cắt cố định).
        ref_boxes, _ref_kpts, ref_map, ref_dlt, manual_crops = self._collect_export_plan()
        ref_ids = {k - 1: v for k, v in ref_map.items()}      # det_idx (0-based) -> id
        ref_deltas = {k - 1: v for k, v in ref_dlt.items()}   # det_idx (0-based) -> delta
        if not ref_ids and not manual_crops:
            QMessageBox.warning(self, "Chưa gán id",
                "Hãy gán id cho folder hiện tại để làm MẪU áp cho tất cả."); return
        # lưu chỉnh sửa folder hiện tại vào cache trước, để vòng lặp đọc đúng khung mới nhất của nó
        self._save_current_state()
        ref_boxes = [list(b) for b in ref_boxes]
        # KHUNG CẮT của mẫu (đã chỉnh) + id, để gán lại id theo MẪU cho folder đã chỉnh (khớp tâm crop).
        edited = self.view.get_boxes()
        ref_crop_boxes, ref_crop_ids = [], []
        for i in range(len(self.first_boxes)):
            val = self.id_inputs[i].currentText().strip() if i < len(self.id_inputs) else ""
            if val:
                ref_crop_boxes.append([int(v) for v in edited[i]])
                ref_crop_ids.append(val)

        # Chọn CỤ THỂ folder áp theo số thứ tự (vd: 1-8  hoặc  8,9  hoặc  1,3,5-7)
        cur_no = self.qi + 1
        text, ok = QInputDialog.getText(
            self, "Áp cho folder nào",
            f"Nhập số thứ tự folder cần áp (1–{len(self.queue)}).\n"
            f"VD: 1-8  hoặc  8,9  hoặc  1,3,5-7. Để trống = tất cả.\n"
            f"(Mỗi folder dùng KHUNG của chính nó; ID luôn gán theo MẪU #{cur_no}, không dùng id riêng.)",
            text=f"1-{len(self.queue)}")
        if not ok:
            return
        targets = self._parse_ranges(text, len(self.queue))
        if not targets:
            QMessageBox.warning(self, "Lỗi", "Phạm vi không hợp lệ."); return

        self.btn_export.setEnabled(False); self.btn_batch.setEnabled(False)
        self.progress.setMaximum(len(targets)); self.progress.setValue(0)

        grand = {}
        for step, k in enumerate(targets):
            folder = self.queue[k]
            self.statusBar().showMessage(f"[{step + 1}/{len(targets)}] #{k + 1} {os.path.basename(folder)}...")
            QApplication.processEvents()
            frames = list_frames(folder)
            if not frames:
                self.progress.setValue(step + 1); self._refresh_queue_marks(); continue
            rec = self._cache.get(folder) or self._saved_edits.get(folder)
            own_boxes = self._boxes_from_record(rec) if rec else []
            if own_boxes and ref_crop_boxes:
                # folder ĐÃ CHỈNH -> dùng KHUNG của chính nó, nhưng ID gán LẠI theo MẪU (khớp tâm crop
                # box với mẫu), KHÔNG dùng id riêng của folder. Cắt cố định (nhanh, không YOLO).
                m = self._match_boxes(own_boxes, ref_crop_boxes)   # own_idx -> ref_crop_idx
                fixed = {}
                for ni, ri in m.items():
                    fixed[ref_crop_ids[ri]] = tuple(own_boxes[ni])
                counts = self._export_folder(folder, frames, [], [], {}, {}, manual_crops=fixed) if fixed else {}
            else:
                # folder CHƯA chỉnh -> phát hiện CỦA CHÍNH NÓ, gán id theo mẫu, crop_region riêng.
                first = cv2.imread(frames[0])
                counts = {}
                if first is not None:
                    nb, nk = detect(first)
                    order = order_boxes(nb, nk)
                    nb = [nb[i] for i in order]; nk = [nk[i] for i in order]
                    match = self._match_boxes(nb, ref_boxes)     # new_idx -> ref_idx
                    mapping, deltas = {}, {}
                    for ni, ri in match.items():
                        if ri in ref_ids:
                            mapping[ni + 1] = ref_ids[ri]
                            deltas[ni + 1] = ref_deltas.get(ri, (0, 0, 0, 0))
                    if mapping or manual_crops:
                        counts = self._export_folder(folder, frames, nb, nk, mapping, deltas, manual_crops)
            for idv, c in counts.items():
                grand[idv] = grand.get(idv, 0) + c
            if counts:
                self.done.add(k)
            self.progress.setValue(step + 1)
            self._refresh_queue_marks()
            QApplication.processEvents()

        self.btn_export.setEnabled(True); self.btn_batch.setEnabled(True)
        summary = "\n".join(f"  id_{k}: {v} ảnh" for k, v in sorted(grand.items()))
        QMessageBox.information(self, "Hoàn tất",
            f"Đã xuất {len(targets)} folder (mỗi folder dùng KHUNG của chính nó; ID theo MẪU).\n\n"
            f"Tổng ảnh mỗi id:\n{summary}")
        self.statusBar().showMessage(
            f"Hoàn tất {len(targets)} folder — tổng {sum(grand.values())} ảnh.")

    @staticmethod
    def _parse_ranges(text, maxn):
        """Chuỗi '1-8, 10, 12-13' -> danh sách chỉ số 0-based (đã lọc trong 1..maxn). Trống = tất cả."""
        text = (text or "").strip()
        if not text:
            return list(range(maxn))
        out = set()
        for tok in text.replace(" ", "").split(","):
            if not tok:
                continue
            if "-" in tok:
                try:
                    a, b = tok.split("-", 1); a, b = int(a), int(b)
                except ValueError:
                    continue
                if a > b:
                    a, b = b, a
                for v in range(a, b + 1):
                    if 1 <= v <= maxn:
                        out.add(v - 1)
            elif tok.isdigit():
                v = int(tok)
                if 1 <= v <= maxn:
                    out.add(v - 1)
        return sorted(out)
