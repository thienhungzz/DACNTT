import os
import sys
import importlib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Khởi chạy chương trình quản lý tồn kho...")

# Khai báo các modules
preprocessing = importlib.import_module("01_data_preprocessing")
data_split = importlib.import_module("01b_data_split")
ml_forecasting = importlib.import_module("02_ml_forecasting")
rl_training = importlib.import_module("04_rl_training")
evaluation = importlib.import_module("05_evaluation")

# Bước 1 & 2
print("\n1. Tiền xử lý dữ liệu...")
# ml_df, m5_df = preprocessing.preprocess_data(data_path="../data/")
print("--> OK!")

print("\n2. Chia tập dữ liệu Train/Val/Test...")
# X_train, y_train, X_val, y_val, X_test, y_test = data_split.split_train_val_test(ml_df)
print("--> OK!")

print("\n3. Huấn luyện mô hình ML...")
# model_xgb = ml_forecasting.train_xgboost(X_train, y_train, X_val, y_val, model_dir="../MoHinh/")
print("--> OK!")

print("\n4a. Huấn luyện RL (Single Item)...")
# model_ppo_single = rl_training.train_ppo_single(env_kwargs={...})
print("--> OK!")

print("\n4b. Huấn luyện RL (Multi Item - Joint Replenishment)...")
# model_ppo_multi = rl_training.train_ppo_multi(env_kwargs={...})
print("--> OK!")

print("\nHoàn thành toàn bộ quy trình kiểm tra import!")