==================================================
HƯỚNG DẪN CÀI ĐẶT VÀ CHẠY DỰ ÁN QUẢN LÝ TỒN KHO
==================================================

1. CẤU TRÚC THƯ MỤC:
- Source/: Chứa mã nguồn chính của dự án (các file .py hoặc .ipynb). Các file rác/cache đã được dọn dẹp.
- Tools/: Chứa các công cụ hỗ trợ cho dự án (nếu có).
- requirements.txt: Chứa danh sách các thư viện Python cần thiết để chạy dự án.
- readme.txt: Hướng dẫn này.

2. HƯỚNG DẪN CÀI ĐẶT:
- Yêu cầu môi trường: Python 3.8+.
- Mở Terminal hoặc Command Prompt tại thư mục 'source_code'.
- Chạy lệnh sau để cài đặt toàn bộ thư viện cần thiết:
  pip install -r requirements.txt

3. HƯỚNG DẪN CHẠY DỰ ÁN:
- Di chuyển vào thư mục Source/.
- Chạy mã nguồn theo thứ tự luồng công việc:
  1. Tiền xử lý dữ liệu (Data Preprocessing).
  2. Huấn luyện mô hình Machine Learning dự báo nhu cầu (XGBoost, LightGBM, LSTM).
  3. Chạy mô hình Reinforcement Learning (PPO) để tối ưu hóa tồn kho.
