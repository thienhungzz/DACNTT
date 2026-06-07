import pandas as pd
import numpy as np

def split_train_val_test(ml_full_df, test_days=28, val_days=28):
    """Chia dữ liệu thành Train, Val, Test theo thờ gian"""
    print("Đang chia tập dữ liệu...")
    max_date = pd.to_datetime(ml_full_df['date']).max()
    test_start = max_date - pd.Timedelta(days=test_days)
    val_start = test_start - pd.Timedelta(days=val_days)

    dates = pd.to_datetime(ml_full_df['date'])
    train_mask = dates <= val_start
    val_mask = (dates > val_start) & (dates <= test_start)
    test_mask = dates > test_start

    features = [c for c in ml_full_df.columns if c not in ['date', 'sales']]
    target = 'sales'

    X_train, y_train = ml_full_df.loc[train_mask, features], ml_full_df.loc[train_mask, target]
    X_val, y_val = ml_full_df.loc[val_mask, features], ml_full_df.loc[val_mask, target]
    X_test, y_test = ml_full_df.loc[test_mask, features], ml_full_df.loc[test_mask, target]

    print(f"Train size: {len(X_train):,}, Val size: {len(X_val):,}, Test size: {len(X_test):,}")
    return X_train, y_train, X_val, y_val, X_test, y_test
