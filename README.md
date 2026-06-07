<div align="center">

<img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/XGBoost-ML_Forecasting-F7931E?style=for-the-badge"/>
<img src="https://img.shields.io/badge/LightGBM-Gradient_Boosting-2980B9?style=for-the-badge"/>
<img src="https://img.shields.io/badge/LSTM-Deep_Learning-EE4C2C?style=for-the-badge"/>
<img src="https://img.shields.io/badge/PPO-Reinforcement_Learning-6DB33F?style=for-the-badge"/>

# 📦 Hệ thống Dự báo & Điều phối Tồn kho

**Inventory Forecasting & Coordination · Machine Learning + Deep Learning + Reinforcement Learning**

---

> Dự án xây dựng hệ thống thông minh kết hợp **Học máy**, **Học sâu** và **Học tăng cường**  
> để dự báo nhu cầu và tối ưu hóa chiến lược điều phối tồn kho đa mặt hàng

</div>

---

## 👥 Thành viên nhóm

| Họ và Tên |
|---|
| Tôn Quốc Thái |
| Trần Thiên Hưng |


---

## 📑 Mục lục

- [Tổng quan](#-tổng-quan)
- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [Công nghệ sử dụng](#-công-nghệ-sử-dụng)
- [Cài đặt & Chạy](#-cài-đặt--chạy)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Dependencies](#-dependencies)

---

## 🎯 Tổng quan

Hệ thống giải quyết bài toán **quản lý tồn kho** thông minh qua hai giai đoạn:

| Giai đoạn | Phương pháp | Mục tiêu |
|---|---|---|
| **Dự báo nhu cầu** | XGBoost · LightGBM · LSTM · Two-Stage | Dự đoán lượng hàng cần nhập |
| **Điều phối tồn kho** | PPO (Reinforcement Learning) | Tối ưu hóa chiến lược đặt hàng |

---

## 🏗️ Kiến trúc hệ thống

```
┌──────────────────────────────────────────────────────────┐
│                    DATA PIPELINE                         │
│                                                          │
│  📂 Raw Data (link_data.txt)                            │
│        ↓                                                 │
│  🔄 Tiền xử lý  ←  data_preprocessing.py               │
│        ↓                                                 │
│  ✂️  Train / Val / Test Split  ←  data_split.py         │
└─────────────────────┬────────────────────────────────────┘
                      │
┌─────────────────────▼────────────────────────────────────┐
│                   MODEL PIPELINE                         │
│                                                          │
│  📈 ML Forecasting                                       │
│     XGBoost · LightGBM · LSTM · Two-Stage Model         │
│        ↓                                                 │
│  🤖 Reinforcement Learning                               │
│     PPO · FairWarehouseEnv · MultiItemWarehouseEnv      │
│        ↓                                                 │
│  📊 Evaluation  ←  EOQ · Metrics                        │
└─────────────────────┬────────────────────────────────────┘
                      │
┌─────────────────────▼────────────────────────────────────┐
│  🚀 main.py  —  Entry Point / Full Pipeline              │
└──────────────────────────────────────────────────────────┘
```

---

## 🛠️ Công nghệ sử dụng

| Thành phần | Công nghệ |
|---|---|
| Forecasting Models | XGBoost, LightGBM, LSTM, Two-Stage |
| RL Framework | PPO (Stable-Baselines3 / TRL) |
| RL Environments | FairWarehouseEnv, MultiItemWarehouseEnv |
| Evaluation | EOQ (Economic Order Quantity) |
| Language | Python 3.8+ |
| Notebooks | Jupyter Notebook |

---

## 🚀 Cài đặt & Chạy

### Yêu cầu hệ thống

```
Python  ≥ 3.8
RAM     ≥ 8 GB (khuyến nghị)
```

### Bước 1 — Clone repository

```bash
git clone https://github.com/Tonthai0607/DuAnCNTT.git
cd DuAnCNTT
```

### Bước 2 — Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Bước 3 — Tải dữ liệu

Xem link Google Drive trong file `link_data.txt`, tải về và đặt folder `data/` vào trong thư mục `Source/`:

```
Source/
└── data/       ← đặt data vào đây
```

### Bước 4 — Chạy pipeline

```bash
cd Source
python main.py
```

> ⚙️ **Lưu ý**: Mở `main.py` để bỏ comment các bước muốn chạy (ML, RL) và kiểm tra lại `data_path` cho khớp với máy.

---

## 📁 Cấu trúc thư mục

```
DuAnCNTT/
│
├── Source/
│   ├── data_preprocessing.py     # Tiền xử lý dữ liệu thô
│   ├── data_split.py             # Chia tập Train / Val / Test
│   ├── ml_forecasting.py         # Mô hình ML: XGBoost, LightGBM, LSTM, Two-Stage
│   ├── rl_environments.py        # Môi trường RL: FairWarehouseEnv, MultiItemWarehouseEnv
│   ├── rl_training.py            # Huấn luyện RL: PPO
│   ├── evaluation.py             # Tính toán EOQ và đánh giá mô hình
│   └── main.py                   # Entry point — chạy toàn bộ pipeline
│
├── Tools/
│   └── utils.py                  # Các hàm tiện ích hỗ trợ
│
├── notebook/                     # Jupyter notebooks phân tích & thực nghiệm
│
├── .gitignore
├── link_data.txt                 # Link Google Drive chứa dataset
├── requirements.txt              # Danh sách thư viện Python
└── README.md
```

---

## 📦 Dependencies

```bash
pip install -r requirements.txt
```

Các thư viện chính bao gồm:

```
# Machine Learning
xgboost · lightgbm · scikit-learn

# Deep Learning
torch · tensorflow (hoặc keras)

# Reinforcement Learning
stable-baselines3 · gymnasium

# Data Processing
pandas · numpy · matplotlib · seaborn

# Notebook
jupyter
```

---

## 📄 Giấy phép

Dự án được phát triển cho mục đích **học thuật và nghiên cứu**.

---

<div align="center">

**Dự án CNTT · Dự báo & Điều phối Tồn kho**  
Machine Learning · Deep Learning · Reinforcement Learning

</div>
