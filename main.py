# -*- coding: utf-8 -*-
"""
AUTO CROP theo ID - Tạo dataset từng người từ folder frame đã băm sẵn.

Cấu trúc dự án (đã tách nhỏ cho gọn):
  - config.py     : thông số tinh chỉnh + nạp model YOLO
  - detection.py  : phát hiện người, khử trùng, cắt khung, tracking, liệt kê file
  - gui.py        : giao diện PyQt5
  - main.py       : file này - chỉ khởi chạy

Luồng làm việc:
  1) Chọn folder chứa ~60 frame (đã băm sẵn từ video).
  2) App phát hiện người ở FRAME ĐẦU, đánh số tạm #1, #2... lên ảnh.
  3) Bạn gõ id thật cho từng khung theo BỘ QUY TẮC (để trống = bỏ qua / người mới vào).
  4) App tự bám (track) từng người qua tất cả frame, cắt theo thân (kể cả khi
     không thấy mặt), xuất vào: <folder>/id_<N>/<tên_frame>.jpg

Yêu cầu: pip install opencv-python numpy PyQt5 ultralytics
Chạy: python main.py
"""

import sys
from PyQt5.QtWidgets import QApplication

from gui import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
